"""Exercise benchmark checkpointing against local Chroma and BGE."""
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from evaluation import run
from test_core import FixtureChat
from vector_graph_rag_chinese_light import clients


@unittest.skipUnless(os.getenv("VGRAG_TEST_CHROMA_URL"), "Set VGRAG_TEST_CHROMA_URL for local integration checks")
class EvaluationIntegrationTests(unittest.TestCase):
    def test_shared_index_fallback_and_resume(self):
        fixture = FixtureChat()
        request_json = clients.request_json
        events = []
        scoring = {"interrupt": False, "rate_limit": False, "timeouts": 3}
        faults = {"relations": False, "entities": "http"}
        fallback_question = "示例值守岗由谁担任，负责什么？"
        normal_question = "示例值守岗由谁担任、负责什么、联络渠道是什么？"

        def respond(url, body=None, headers=None, timeout=25):
            if not url.startswith("https://api.siliconflow.cn/"):
                return request_json(url, body, headers, timeout)
            system, payload = body["messages"]
            data = json.loads(payload["content"])
            events.append(data)
            if "reference" in data:
                if body["model"] == "invalid":
                    raise HTTPError(url, 404, "invalid model", {}, None)
                if data["question"] == fallback_question and scoring["interrupt"]:
                    scoring["interrupt"] = False
                    raise KeyboardInterrupt
                if data["question"] == fallback_question and scoring["rate_limit"]:
                    scoring["rate_limit"] = False
                    raise HTTPError(url, 429, "limited", {"Retry-After": "0"}, None)
                if data["question"] == fallback_question and scoring["timeouts"]:
                    scoring["timeouts"] -= 1
                    raise TimeoutError
                content = '{"accuracy":"correct","grounded":true}'
            elif "entities" in system["content"]:
                if data["question"] == fallback_question and faults["entities"] == "http":
                    raise HTTPError(url, 503, "unavailable", {}, None)
                if data["question"] == fallback_question and faults["entities"] == "invalid":
                    content = '{"entities":"示例值守岗"}'
                elif data["question"] == normal_question:
                    content = '{"entities":[]}'
                else:
                    content = fixture(system["content"], data, body["max_tokens"])
            elif "relations" in data and data["question"] == fallback_question and faults["relations"]:
                content = '{"ids":[999]}'
            elif "candidates" in data and data["question"] == fallback_question:
                content = json.dumps({"ids": [data["candidates"][0]["id"]]})
            else:
                content = fixture(system["content"], data, body["max_tokens"])
            if "excerpts" in data and not faults["relations"]:
                content += "（恢复后生成）"
            return {"choices": [{"message": {"content": content}}]}

        chroma_url = os.environ["VGRAG_TEST_CHROMA_URL"]
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
            "SILICONFLOW_API_KEY": "fixture-key", "VGRAG_CHAT_MODEL": "fixture-chat", "VGRAG_CHROMA_URL": chroma_url,
        }), patch.object(clients, "request_json", side_effect=respond), patch.object(run, "sleep") as wait:
            root = Path(directory)
            documents = root / "documents.jsonl"
            questions = root / "questions.jsonl"
            index = root / "index"
            two = root / "two.jsonl"
            single = root / "single.jsonl"
            sources = ["a" * 64, "b" * 64, "c" * 64]
            with documents.open("w", encoding="utf-8") as stream:
                for source, body in zip(sources, ("示例项目的重大变更必须由示例值守岗审批。",
                                                  "示例值守岗由示例成员担任。", "联络渠道为内部值班电话。")):
                    run.append(stream, {"source": source, "text": body, "title": "中文新闻"})
            with questions.open("w", encoding="utf-8") as stream:
                for i, question in enumerate((fallback_question, normal_question), start=1):
                    run.append(stream, {"id": f"q{i}", "question": question, "reference": "示例参考",
                                        "sources": sources[:i + 1]})

            def invoke(mode, output, *extra, judge="fixture-judge"):
                argv = ["run.py", "--documents", str(documents), "--questions", str(questions),
                        "--data-dir", str(index), "--judge-model", judge, "--retrieval-mode", mode,
                        "--output", str(output), *extra]
                with patch.object(sys, "argv", argv), redirect_stdout(StringIO()) as stdout:
                    run.main()
                return json.loads(stdout.getvalue())

            with self.assertRaises(run.ModelConfigurationError):
                invoke("single-stage", single, judge="invalid")
            self.assertFalse(index.exists())

            def manifest(path):
                return json.loads(path.with_suffix(".jsonl.manifest.json").read_text())

            try:
                with self.assertRaises(SystemExit) as incomplete, redirect_stderr(StringIO()):
                    invoke("single-stage", single)
                self.assertEqual(incomplete.exception.code, 1)
                checkpoint = run.previous_rows(single)
                self.assertEqual(checkpoint["q1"]["grade_error"], "TimeoutError")
                self.assertEqual(checkpoint["q1"]["entity_extraction_error"], "HTTP 503")
                self.assertIsNotNone(checkpoint["q2"]["grade"])
                answers_sent = sum("excerpts" in data for data in events)
                single_summary = invoke("single-stage", single, "--resume")
                self.assertEqual(single_summary["questions"], 2)
                self.assertEqual(single_summary["complete_answers"], 2)
                self.assertEqual(sum("excerpts" in data for data in events), answers_sent)
                self.assertEqual(single_summary["effective_modes"], {"single-stage": 2})
                self.assertEqual(single_summary["entity_extraction_failures"], 1)
                graph_payloads = [data for data in events if "items" in data]
                self.assertTrue(graph_payloads)
                self.assertTrue(all(item["text"].startswith("中文新闻\n") for data in graph_payloads
                                    for item in data["items"]))

                wait.reset_mock()
                faults.update(relations=True, entities="invalid")
                scoring.update(interrupt=True, rate_limit=True)
                before_two = len(events)
                with self.assertRaises(KeyboardInterrupt), self.assertLogs(
                    "vector_graph_rag_chinese_light.rag", level="WARNING",
                ):
                    invoke("two-stage", two, "--question-id", "q1")
                interrupted = list(run.rows(two))
                self.assertEqual(len(interrupted), 1)
                self.assertEqual(interrupted[0]["effective_mode"], "single-stage")
                self.assertEqual(interrupted[0]["fallback_reason"], "relation_rerank_error")
                self.assertEqual(interrupted[0]["entity_extraction_status"], "failed")
                self.assertEqual(interrupted[0]["entity_extraction_error"], "invalid_response")
                self.assertIsNone(interrupted[0]["grade"])
                self.assertEqual(interrupted[0]["candidate_matched_sources"], sources[:2])
                payload = next(data for data in events[before_two:] if "candidates" in data)
                self.assertEqual(["中文新闻\n" + part["text"] for part in interrupted[0]["candidates"]],
                                 [part["text"] for part in payload["candidates"]])

                with two.open("ab") as stream:
                    stream.write(b'{"incomplete":')
                with self.assertRaises(SystemExit), redirect_stderr(StringIO()):
                    invoke("single-stage", two, "--resume")
                two_summary = invoke("two-stage", two, "--resume")
                self.assertEqual(two_summary["questions"], 2)
                self.assertEqual(two_summary["effective_modes"], {"single-stage": 1, "two-stage": 1})
                self.assertEqual(two_summary["by_gold_source_count"]["2"]["sources_lost_after_candidates"], 1)
                self.assertEqual(two_summary["by_gold_source_count"]["3"]["all_sources"], 1)
                self.assertEqual(two_summary["complete_answers"], 2)
                self.assertEqual(len(list(run.rows(two))), 4)
                self.assertEqual(sum(data.get("question") == fallback_question and "excerpts" in data
                                     for data in events[before_two:]), 1)
                wait.assert_called_once()
                before_skip = len(events)
                self.assertEqual(invoke("two-stage", two, "--resume"), two_summary)
                self.assertEqual(len(events), before_skip)

                previous = run.previous_rows(two)
                self.assertEqual(previous["q2"]["entity_extraction_status"], "empty")
                self.assertIsNone(previous["q2"]["entity_extraction_error"])
                self.assertEqual(two_summary["entity_extraction_failures"], 1)
                self.assertEqual(two_summary["by_gold_source_count"]["2"]["entity_extraction_failures"], 1)
                self.assertEqual(two_summary["by_gold_source_count"]["3"]["entity_extraction_failures"], 0)
                self.assertEqual(single_summary["index_fingerprints"], two_summary["index_fingerprints"])
                self.assertEqual(two_summary["index_fingerprints"]["passages"]["records"], 3)
                for key in ("passage_collection", "graph_collection"):
                    self.assertEqual(manifest(two)[key], manifest(single)[key])

                faults.update(relations=False, entities="ok")
                before_rerun = len(events)
                rerun = invoke("two-stage", two, "--resume", "--question-id", "q1", "--rerun-fallback")
                latest = run.previous_rows(two)
                self.assertEqual(rerun["questions"], 2)
                self.assertEqual(rerun["complete_answers"], 2)
                self.assertEqual(rerun["effective_modes"], {"two-stage": 2})
                self.assertEqual(rerun["entity_extraction_failures"], 0)
                self.assertIsNone(latest["q1"]["fallback_reason"])
                self.assertEqual(latest["q1"]["entity_extraction_status"], "extracted")
                self.assertEqual(latest["q2"], previous["q2"])
                self.assertEqual(len(list(run.rows(two))), 6)
                self.assertNotEqual(latest["q1"]["answer"], previous["q1"]["answer"])
                rerun_events = events[before_rerun:]
                self.assertFalse(any(data.get("question") == normal_question for data in rerun_events))
                self.assertTrue(any("relations" in data for data in rerun_events))
                self.assertEqual([data["answer"] for data in rerun_events if data.get("question") == fallback_question
                                  and "reference" in data], [latest["q1"]["answer"]])
                self.assertEqual(sum("items" in data for data in events), len(graph_payloads))
                self.assertTrue(all(item["title"] == "中文新闻" for data in events if "excerpts" in data
                                    for item in data["excerpts"]))
                with self.assertRaises(SystemExit), redirect_stderr(StringIO()) as errors:
                    invoke("two-stage", two, "--resume", "--question-id", "q2", "--rerun-fallback")
                self.assertIn("q2 does not have a scored fallback result", errors.getvalue())

                before_mutation = len(events)
                graph_path = manifest(two)["graph_collection"]
                passage_path = manifest(two)["passage_collection"]
                chroma = clients.Chroma(chroma_url)

                graph = chroma(graph_path + "/get", {"limit": 1, "include": ["embeddings", "metadatas"]})
                self.assertTrue(graph["ids"])
                chroma(graph_path + "/delete", {"ids": graph["ids"]})
                with self.assertRaisesRegex(run.IndexIntegrityError, "fingerprint mismatch: graph"):
                    invoke("two-stage", two, "--resume")
                chroma(graph_path + "/upsert", {key: graph[key] for key in ("ids", "embeddings", "metadatas")})
                self.assertEqual(invoke("two-stage", two, "--resume")["index_fingerprints"], rerun["index_fingerprints"])

                passage = chroma(passage_path + "/get", {"limit": 1, "include": ["embeddings", "metadatas"]})
                new_output = root / "changed.jsonl"
                vector = passage["embeddings"][0].copy()
                vector[0] += 0.5
                count = request_json(chroma_url + passage_path + "/count")
                chroma(passage_path + "/update", {"ids": passage["ids"], "embeddings": [vector]})
                self.assertEqual(request_json(chroma_url + passage_path + "/count"), count)
                with self.assertRaisesRegex(run.IndexIntegrityError, "fingerprint mismatch: passages"):
                    invoke("single-stage", new_output)
                chroma(passage_path + "/update", {"ids": passage["ids"], "embeddings": passage["embeddings"],
                                                   "metadatas": [{"graph_ready": False}]})
                self.assertEqual(request_json(chroma_url + passage_path + "/count"), count)
                with self.assertRaisesRegex(run.IndexIntegrityError, "fingerprint mismatch: passages"):
                    invoke("single-stage", single, "--resume")
                self.assertFalse(any(data.get("question") in {fallback_question, normal_question}
                                     or "items" in data for data in events[before_mutation:]))
            finally:
                manifest_path = two.with_suffix(".jsonl.manifest.json")
                if manifest_path.exists():
                    namespace = json.loads(manifest_path.read_text())["namespace"]
                    for kind in ("passages", "graph"):
                        name = f"vgrag-zh-{namespace}-{kind}"
                        with urlopen(Request(chroma_url + clients.COLLECTIONS + "/" + name, method="DELETE")):
                            pass
