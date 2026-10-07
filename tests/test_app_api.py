import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import app as app_module
from jobs import JobStore

TEXT_PDF = b"""%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
trailer<</Root 1 0 R>>
%%EOF
"""


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

    def test_mkdir_fail_then_next_upload_not_409(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        tmp_upload = Path(tmp.name) / ".tmp"
        tmp_upload.mkdir()
        original_mkdir = Path.mkdir

        def boom(self, *args, **kwargs):
            if str(self).startswith(str(tmp_upload)):
                raise OSError("disk full")
            return original_mkdir(self, *args, **kwargs)

        with patch.object(app_module, "TMP_UPLOAD_DIR", tmp_upload), patch.object(Path, "mkdir", boom):
            res = self.client.post(
                "/api/upload",
                data={"files": (io.BytesIO(TEXT_PDF), "note.pdf")},
                content_type="multipart/form-data",
            )
        self.assertEqual(res.status_code, 500)
        self.assertIn("업로드 준비에 실패했습니다", res.get_json()["error"])
        self.assertFalse(self.store.has_active())

        res2 = self.client.post(
            "/api/upload",
            data={"files": (io.BytesIO(b"hello"), "note.txt")},
            content_type="multipart/form-data",
        )
        self.assertEqual(res2.status_code, 400)
        self.assertNotEqual(res2.status_code, 409)
        self.assertFalse(self.store.has_active())

    def test_thread_start_fail_releases_lock(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        tmp_upload = Path(tmp.name) / ".tmp"
        tmp_upload.mkdir()
        with patch.object(app_module, "TMP_UPLOAD_DIR", tmp_upload), patch(
            "app.threading.Thread", side_effect=RuntimeError("cannot start thread")
        ):
            res = self.client.post(
                "/api/upload",
                data={"files": (io.BytesIO(TEXT_PDF), "note.pdf")},
                content_type="multipart/form-data",
            )
        self.assertEqual(res.status_code, 500)
        self.assertIn("문서 처리를 시작하지 못했습니다", res.get_json()["error"])
        self.assertFalse(self.store.has_active())

        res2 = self.client.post(
            "/api/upload",
            data={"files": (io.BytesIO(b"hello"), "note.txt")},
            content_type="multipart/form-data",
        )
        self.assertEqual(res2.status_code, 400)
        self.assertFalse(self.store.has_active())

    def test_save_fail_releases_lock(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        tmp_upload = Path(tmp.name) / ".tmp"
        tmp_upload.mkdir()
        with patch.object(app_module, "TMP_UPLOAD_DIR", tmp_upload), patch(
            "werkzeug.datastructures.FileStorage.save", side_effect=OSError("save failed")
        ):
            res = self.client.post(
                "/api/upload",
                data={"files": (io.BytesIO(TEXT_PDF), "note.pdf")},
                content_type="multipart/form-data",
            )
        self.assertEqual(res.status_code, 500)
        self.assertFalse(self.store.has_active())
        res2 = self.client.post(
            "/api/upload",
            data={"files": (io.BytesIO(b"hello"), "note.txt")},
            content_type="multipart/form-data",
        )
        self.assertEqual(res2.status_code, 400)


if __name__ == "__main__":
    unittest.main()
