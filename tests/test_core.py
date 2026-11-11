"""Local integration checks with deterministic chat responses."""
import json
import os
import tempfile
import unittest
from urllib.error import URLError
from urllib.request import Request, urlopen

from vector_graph_rag_chinese_light import ChineseLightRAG
from vector_graph_rag_chinese_light.clients import COLLECTIONS
from vector_graph_rag_chinese_light.documents import chunks, text_blocks
from vector_graph_rag_chinese_light.graph import query_entities


class FixtureChat:
    def __init__(self):
        self.fail = False
        self.relation_rankings = []
        self.fail_relation_ranking = False

    def __call__(self, system, data, max_tokens, timeout=25):
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
    def test_locations_and_chunk_limit(self):
        self.assertEqual(list(text_blocks("一\r\n\r\n二")), [("第 1 行", "一"), ("第 3 行", "二")])
        self.assertTrue(all(len(part) <= 300 for part in chunks("示例文字" * 250)))


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
                rag.upsert_text("roster", "示例值守岗由示例成员担任。")
                result = rag.query("示例值守岗负责什么，由谁担任？")
                self.assertEqual(result.status, "answered")
                self.assertEqual({p.source for p in result.citations}, {"policy", "roster"})
                self.assertIn("示例成员", result.answer)
                self.assertEqual(chat.relation_rankings, [])

                two_stage = ChineseLightRAG(directory, chroma_url=chroma_url, chat=chat,
                                            retrieval_mode="two-stage")
                question = "示例值守岗负责什么，由谁担任？"
                trace = {}
                result = two_stage.query(question, trace=trace)
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
