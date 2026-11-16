"""Offline regression tests. These do not establish model/benchmark quality."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "starter"))
import day07
from evidence import evidence_grade, strict_hit, validate_cases, write_json, fingerprint
from agentic import run_agent
from ollama_controller import OllamaController


def chunk(cid, text):
    return {"id": cid, "text": text, "source_id": "doc", "source": "book", "page": 1, "metadata": {}}


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [chunk("a", "Adopt the method."), chunk("b", "Reward the worker.")]
        original = {"source_id": "doc", "page": 1, "evidence": "Adopt the method. Reward the worker."}
        self.case = {"id": "q", "split": "dev", "question": "Both?", "answerable": True,
                     "expected_answer": "Adopt and reward", "gold": [original],
                     "evidence_groups": [[{**original, "evidence": c["text"], "chunk_id": c["id"]}
                                           for c in self.chunks]]}

    def test_required_parts_are_and_not_or(self):
        validate_cases(self.chunks, [self.case], "dev")
        self.assertEqual(evidence_grade(self.chunks[:1], self.case), 1)
        self.assertEqual(evidence_grade(self.chunks, self.case), 2)
        self.assertFalse(strict_hit(self.chunks, self.case))
        self.assertEqual(evidence_grade([], self.case), 0)

    def test_alternative_groups_are_or(self):
        alternative = chunk("c", "Alternative full support.")
        case = copy.deepcopy(self.case)
        case["evidence_groups"].append([{"source_id": "doc", "page": 1,
                                         "evidence": alternative["text"]}])
        self.assertEqual(evidence_grade([alternative], case), 2)

    def test_cannot_silently_shorten_original_gold(self):
        self.case["evidence_groups"][0].pop()
        with self.assertRaises(ValueError):
            validate_cases(self.chunks, [self.case], "dev")

    def test_wrong_page_does_not_support_evidence(self):
        rows = [{**c, "page": 2} for c in self.chunks]
        self.assertEqual(evidence_grade(rows, self.case), 0)


class ControllerTests(unittest.TestCase):
    def test_real_transport_contract_and_gold_is_not_added(self):
        controller = OllamaController()
        action = {"action": "search", "query": "question", "answer": "", "citations": [], "reason": "find source"}
        response = {"done": True, "message": {"content": json.dumps(action)}, "prompt_eval_count": 12, "eval_count": 5}
        with patch.object(controller, "request", return_value=response) as transport:
            self.assertEqual(controller.decide("question", []), action)
        payload = transport.call_args.args[1]
        self.assertFalse(payload["stream"])
        user_payload = json.loads(payload["messages"][1]["content"])
        self.assertEqual(set(user_payload), {"question", "retrieval_history", "output_schema"})
        self.assertEqual(controller.usage()["prompt_tokens"], 12)

    def test_nonlocal_endpoint_and_incomplete_response_rejected(self):
        with self.assertRaises(ValueError):
            OllamaController(base_url="https://example.org")
        controller = OllamaController()
        with patch.object(controller, "request", return_value={"done": True, "done_reason": "length"}):
            with self.assertRaises(ValueError):
                controller.decide("q", [])
        self.assertIn("error", controller.calls[0])

    def test_renderer_only_formats_model_selected_context_ids(self):
        for cid, expected in (("a", "A fact. [a]"), ("not-in-context", "A fact.")):
            controller = OllamaController()
            raw = {"action": "answer", "query": "", "answer": "A fact.", "citations": [cid], "reason": ""}
            response = {"done": True, "message": {"content": json.dumps(raw)}}
            with patch.object(controller, "request", return_value=response):
                action = controller.decide("q", [{"context_ids": ["a"], "context": "A fact."}])
            self.assertEqual(action["answer"], expected)
            self.assertEqual(json.loads(controller.calls[0]["response"]["message"]["content"])["answer"], "A fact.")


class AgentSafetyTests(unittest.TestCase):
    def run_actions(self, actions, event_log=None):
        class Controller:
            def decide(self, question, history): return next(actions)
        class Retriever:
            def search(self, query, **kwargs): return [chunk("a", "A fact.")]
        return run_agent("q", Controller(), Retriever(), event_log=event_log)

    def test_deliberate_abstention_is_distinct(self):
        result = self.run_actions(iter([{"action": "search", "query": "q"},
                                      {"action": "abstain", "reason": "No amount specified"}]))
        self.assertEqual(result["status"], "abstain")
        self.assertEqual(result["events"][-1]["event"], "deliberate_abstention")

    def test_extra_unlisted_citation_is_rejected(self):
        result = self.run_actions(iter([{"action": "search", "query": "q"},
                    {"action": "answer", "answer": "A fact [a] and fabricated [fake]", "citations": ["a"]}]))
        self.assertEqual(result["status"], "abstain")

    def test_repeated_query_cannot_spin_on_identical_results(self):
        result = self.run_actions(iter([{"action": "search", "query": "q"},
                                       {"action": "search", "query": "q"}]))
        self.assertEqual(result["events"][-1]["event"], "duplicate_query_stop")
        self.assertEqual(sum(e["event"] == "search" for e in result["events"]), 1)

    def test_partial_trace_survives_later_failure(self):
        log = []
        with self.assertRaises(StopIteration):
            self.run_actions(iter([{"action": "search", "query": "q"}]), event_log=log)
        self.assertEqual(log[0]["event"], "search")


class TestIsolationTests(unittest.TestCase):
    def test_freeze_detects_changed_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            benchmark = root / "benchmark"
            write_json(benchmark / "test.json", [{"question": "held out"}])
            write_json(benchmark / "freeze.json", {"files": {"benchmark/test.json": fingerprint(benchmark / "test.json")}})
            with patch.object(day07, "ROOT", root), patch.object(day07, "BENCHMARK", benchmark):
                day07.verify_freeze()
                write_json(benchmark / "test.json", [{"question": "changed"}])
                with self.assertRaisesRegex(ValueError, "Frozen input changed"):
                    day07.verify_freeze()

    def test_test_run_cannot_skip_dev_selection(self):
        with patch.object(day07, "verify_freeze", return_value="hash"), \
             patch.object(day07, "verify_selection", side_effect=ValueError("No dev selection")), \
             patch.object(day07, "check_runtime") as runtime:
            with self.assertRaisesRegex(ValueError, "No dev selection"):
                day07.run("test", ["B0"], "model", "http://localhost:11434", 1)
            runtime.assert_not_called()


if __name__ == "__main__":
    unittest.main()
