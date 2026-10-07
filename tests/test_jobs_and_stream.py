import unittest
from unittest.mock import patch

from jobs import JobStore
from rag_engine import NO_INFO_MESSAGE, RAGEngine


class JobStoreTests(unittest.TestCase):
    def test_duplicate_active_job_rejected(self):
        store = JobStore()
        first = store.create()
        self.assertTrue(store.has_active())
        with self.assertRaises(RuntimeError):
            store.create()
        store.update(first.id, status="done")
        second = store.create()
        self.assertNotEqual(first.id, second.id)

    def test_missing_job_after_restart_behavior(self):
        store = JobStore()
        self.assertIsNone(store.get("missing"))
        self.assertFalse(store.has_active())


class StreamDoneTests(unittest.TestCase):
    def test_no_info_path_emits_normalized_done_answer(self):
        engine = RAGEngine()
        engine.vectorstore = None
        events = list(engine.answer_stream("anything"))
        types = [event["type"] for event in events]
        self.assertEqual(types, ["meta", "token", "done"])
        self.assertEqual(events[-1]["answer"], NO_INFO_MESSAGE)

    def test_llm_stream_done_uses_stripped_answer(self):
        engine = RAGEngine()
        engine.vectorstore = object()
        engine.llm = type("L", (), {"stream": lambda self, prompt: ["  hello ", "world  "]})()
        with patch.object(engine, "retrieve", return_value=[
            (
                type("D", (), {"page_content": "ctx", "metadata": {"filename": "a.pdf", "page_number": 1}})(),
                0.1,
            )
        ]), patch.object(engine, "_collection_count", return_value=1):
            events = list(engine.answer_stream("q"))
        done = [event for event in events if event["type"] == "done"][0]
        self.assertEqual(done["answer"], "hello world")


if __name__ == "__main__":
    unittest.main()
