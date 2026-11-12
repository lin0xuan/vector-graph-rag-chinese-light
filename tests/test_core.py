"""Local integration checks with deterministic chat responses."""
import json
import os
import tempfile
import unittest
from urllib.error import URLError
from urllib.request import Request, urlopen

from vector_graph_rag_chinese_light import ChineseLightRAG
from vector_graph_rag_chinese_light.clients import COLLECTIONS
from vector_graph_rag_chinese_light.graph import query_entities
from vector_graph_rag_chinese_light.index import Passage


class FixtureChat:
    def __init__(self):
        self.fail = False
        self.relation_rankings = []
        self.fail_relation_ranking = False
        self.calls = []

    def __call__(self, system, data, max_tokens, timeout=25):
        self.calls.append((data, max_tokens))
        if self.fail:
            raise ValueError("Simulated extraction failure")
        if "items" in data:
            items = []
            for item in data["items"]:
                text = item["text"]
                triples = []
                if "重大变更必须" in text:
                    triples.append(["示例项目的重大变更", "事项", "由其审批", "示例值守岗", "岗位"])
                if "由示例成员担任" in text:
                    triples.append(["示例值守岗", "岗位", "担任者", "示例成员", "人员"])
                items.append({"id": item["id"], "triplets": triples})
            return json.dumps({"items": items}, ensure_ascii=False)
        if "entities" in system:
            return '{"entities":["示例值守岗"]}'
        if "relations" in data:
            self.relation_rankings.append(data["relations"])
            if self.fail_relation_ranking:
                return '{"ids":[999]}'
            return json.dumps({"ids": [item["id"] for item in data["relations"]]})
        if "candidates" in data:
            return json.dumps({"ids": [item["id"] for item in data["candidates"][:8]]})
        return "示例项目的重大变更由示例值守岗审批，该岗位由示例成员担任。"


class TextTests(unittest.TestCase):
    def test_balanced_candidates_deduplicate_without_losing_title_context(self):
        rag = object.__new__(ChineseLightRAG)
        passages = {
            'a': Passage('a', 'first', '', '收入为十万元。', '2023年'),
            'copy': Passage('copy', 'reprint', '', '收入为十万元。', '2023年'),
            'year': Passage('year', 'other-year', '', '收入为十万元。', '2024年'),
            'bridge': Passage('bridge', 'roster', '', '负责人为李某。', '名单'),
            'last': Passage('last', 'policy', '', '该规定不适用于分公司。', '制度'),
        }
        self.assertEqual(rag._merge_candidates(
            passages, ['a', 'copy', 'year'], ['a', 'a', 'bridge', 'last'],
        ), ['a', 'bridge', 'year', 'last'])

    def test_balanced_candidates_bound_both_branches_and_fill_after_duplicates(self):
        rag = object.__new__(ChineseLightRAG)
        passages = {f'{branch}{i}': Passage(f'{branch}{i}', 'news', '', f'{branch}独有事实{i}')
                    for branch in ('d', 'g') for i in range(30)}
        dense = [f'd{i}' for i in range(12)]
        graph = [f'g{i}' for i in range(30)]
        self.assertEqual(rag._merge_candidates(passages, dense, graph),
                         [pid for i in range(12) for pid in (f'd{i}', f'g{i}')])
        passages.update({f'copy{i}': Passage(f'copy{i}', 'news', '', 'd独有事实0') for i in range(30)})
        self.assertEqual(rag._merge_candidates(passages, [], [f'copy{i}' for i in range(30)] + graph),
                         ['copy0'] + graph[:23])

    def test_entity_extraction_timeout(self):
        def timeout(*args, **kwargs):
            raise URLError(TimeoutError())

        trace = {}
        self.assertEqual(query_entities(timeout, "谁负责审批？", trace=trace), [])
        self.assertEqual(trace, {"entity_extraction_status": "failed", "entity_extraction_error": "TimeoutError"})


@unittest.skipUnless(os.getenv("VGRAG_TEST_CHROMA_URL"), "Set VGRAG_TEST_CHROMA_URL to run integration checks")
class IntegrationTests(unittest.TestCase):
    def test_public_api_update_delete_and_rollback(self):
        chat = FixtureChat()
        with tempfile.TemporaryDirectory() as directory:
            chroma_url = os.environ["VGRAG_TEST_CHROMA_URL"]
            rag = ChineseLightRAG(directory, chroma_url=chroma_url, chat=chat, retrieval_mode="single-stage")
            try:
                rag.upsert_text("policy", "示例项目的重大变更必须由示例值守岗审批。", title="示例制度")
                self.assertEqual(rag.index.passages()[0].title, "示例制度")
                citation = rag.index.passages()[0].as_dict()
                self.assertEqual(citation["source_type"], "txt")
                self.assertEqual(citation["end_offset"], len(citation["text"]))
                self.assertGreater(citation["token_count"], 0)
                rag.upsert_text("roster", "示例值守岗由示例成员担任。")
                before_query = len(chat.calls)
                result = rag.query("示例值守岗负责什么，由谁担任？")
                self.assertEqual(len(chat.calls) - before_query, 3)
                self.assertEqual(chat.calls[-1][1], 1000)
                self.assertEqual(result.status, "answered")
                self.assertEqual({p.source for p in result.citations}, {"policy", "roster"})
                self.assertIn("示例成员", result.answer)
                self.assertEqual(chat.relation_rankings, [])

                two_stage = ChineseLightRAG(directory, chroma_url=chroma_url, chat=chat,
                                            retrieval_mode="two-stage")
                question = "示例值守岗负责什么，由谁担任？"
                trace = {}
                before_query = len(chat.calls)
                result = two_stage.query(question, trace=trace)
                self.assertEqual(len(chat.calls) - before_query, 4)
                self.assertEqual(chat.calls[-1][1], 1000)
                self.assertEqual(trace["effective_mode"], "two-stage")
                self.assertEqual({p["source"] for p in trace["candidates"]}, {"policy", "roster"})
                self.assertEqual(result.status, "answered")
                self.assertEqual({p.source for p in result.citations}, {"policy", "roster"})
                self.assertEqual(len(chat.relation_rankings), 1)
                self.assertEqual({item["predicate"] for item in chat.relation_rankings[0]},
                                 {"由其审批", "担任者"})
                chat.fail_relation_ranking = True
                with self.assertLogs("vector_graph_rag_chinese_light.rag", level="WARNING") as logs:
                    self.assertEqual(two_stage.search(question, trace=trace), rag.search(question))
                self.assertEqual(trace["effective_mode"], "single-stage")
                self.assertEqual(trace["fallback_reason"], "relation_rerank_error")
                self.assertTrue(any("Relation reranking fallback" in line for line in logs.output))

                old = rag.index.passages()
                rag.upsert_text("roster", "示例值守岗由示例成员担任。")
                self.assertEqual(rag.index.passages(), old)
                chat.fail = True
                with self.assertRaises(ValueError):
                    rag.upsert_text("roster", "示例值守岗现在由新成员担任。")
                self.assertEqual(rag.index.passages(), old)
                chat.fail = False
                rag.upsert_text("roster", "示例值守岗现在由新成员担任。")
                self.assertIn("新成员", next(p.text for p in rag.index.passages() if p.source == "roster"))
                rag.remove_source("roster")
                self.assertEqual({p.source for p in rag.search("示例值守岗审批重大变更。")}, {"policy"})
                rag.remove_source("policy")
                self.assertEqual(rag.query("示例值守岗").status, "no_evidence")
            finally:
                for kind in ("passages", "graph"):
                    collection = f"{COLLECTIONS}/vgrag-zh-{rag.index.namespace}-{kind}"
                    with urlopen(Request(rag.chroma.url + collection, method="DELETE"), timeout=15):
                        pass


if __name__ == "__main__":
    unittest.main()
