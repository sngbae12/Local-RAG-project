import io
import unittest
from unittest.mock import patch

import app as app_module
from jobs import JobStore


class AppApiTests(unittest.TestCase):
    def setUp(self):
        self.store = JobStore()
        self.job_patch = patch.object(app_module, "job_store", self.store)
        self.job_patch.start()
        self.client = app_module.app.test_client()

    def tearDown(self):
        self.job_patch.stop()

    def test_empty_chat_rejected(self):
        with patch.object(app_module, "get_engine") as engine:
            engine.return_value.llm_ready = True
            res = self.client.post("/api/chat", json={"message": "   "})
        self.assertEqual(res.status_code, 400)
        self.assertIn("질문을 입력하세요", res.get_json()["error"])

    def test_retry_models_skips_when_ready(self):
        with patch.object(app_module, "get_engine") as engine:
            inst = engine.return_value
            inst.ready = True
            inst.llm_ready = True
            inst.embeddings_ready = True
            inst.initializing = False
            inst.error = None
            inst.embedding_error = None
            inst.llm_error = None
            inst.list_files.return_value = []
            res = self.client.post("/api/retry-models")
        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.get_json()["ok"])
        inst.initialize.assert_not_called()

    def test_retry_models_avoids_duplicate_initialize(self):
        with patch.object(app_module, "get_engine") as engine:
            inst = engine.return_value
            inst.ready = False
            inst.initializing = True
            inst.llm_ready = False
            inst.embeddings_ready = False
            inst.error = "busy"
            inst.embedding_error = None
            inst.llm_error = None
            inst.list_files.return_value = []
            res = self.client.post("/api/retry-models")
        self.assertEqual(res.status_code, 200)
        inst.initialize.assert_not_called()

    def test_upload_rejects_non_pdf_without_job_run(self):
        data = {"files": (io.BytesIO(b"hello"), "note.txt")}
        res = self.client.post("/api/upload", data=data, content_type="multipart/form-data")
        self.assertEqual(res.status_code, 400)
        body = res.get_json()
        self.assertFalse(body["ok"])
        self.assertEqual(body["status"], "failed")

    def test_missing_job_is_not_success(self):
        res = self.client.get("/api/jobs/does-not-exist")
        self.assertEqual(res.status_code, 404)
        self.assertTrue(res.get_json()["missing"])


if __name__ == "__main__":
    unittest.main()
