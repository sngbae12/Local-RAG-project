import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.documents import Document

from rag_engine import RAGEngine


TEXT_PDF = b"""%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj
3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj
4 0 obj<</Length 68>>stream
BT /F1 12 Tf 72 720 Td (Safety replacement document alpha) Tj ET
endstream
endobj
5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj
trailer<</Root 1 0 R>>
%%EOF
"""


class FakeStore:
    def __init__(self):
        self.ids = []
        self.metadatas = []
        self.documents = []
        self.fail_add = False
        self.fail_delete = False
        self._collection = self

    def add_documents(self, chunks, ids=None):
        if self.fail_add:
            raise RuntimeError("embed failed")
        ids = ids or [f"auto-{len(self.ids) + i}" for i in range(len(chunks))]
        for chunk, doc_id in zip(chunks, ids):
            self.ids.append(doc_id)
            self.metadatas.append(dict(chunk.metadata or {}))
            self.documents.append(chunk.page_content)

    def get(self, where=None, ids=None, include=None):
        selected = []
        for idx, doc_id in enumerate(self.ids):
            meta = self.metadatas[idx]
            if ids is not None and doc_id not in ids:
                continue
            if where and meta.get("filename") != where.get("filename"):
                continue
            selected.append((doc_id, meta, self.documents[idx]))
        return {
            "ids": [item[0] for item in selected],
            "metadatas": [item[1] for item in selected],
            "documents": [item[2] for item in selected],
        }

    def delete(self, ids=None):
        if self.fail_delete:
            raise RuntimeError("delete failed")
        keep = [(i, m, d) for i, m, d in zip(self.ids, self.metadatas, self.documents) if i not in set(ids or [])]
        self.ids = [item[0] for item in keep]
        self.metadatas = [item[1] for item in keep]
        self.documents = [item[2] for item in keep]

    def update(self, ids=None, metadatas=None):
        for doc_id, meta in zip(ids or [], metadatas or []):
            idx = self.ids.index(doc_id)
            merged = dict(self.metadatas[idx])
            merged.update(meta)
            self.metadatas[idx] = merged


class IngestSafetyTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)
        self.uploads = self.root / "uploads"
        self.uploads.mkdir()
        self.engine = RAGEngine()
        self.store = FakeStore()
        self.engine.vectorstore = self.store
        self.engine.embeddings_ready = True
        self.patchers = [
            patch("rag_engine.UPLOAD_DIR", self.uploads),
            patch("rag_engine.TMP_UPLOAD_DIR", self.uploads / ".tmp"),
        ]
        for item in self.patchers:
            item.start()

    def tearDown(self):
        for item in self.patchers:
            item.stop()
        self.tmpdir.cleanup()

    def _seed(self, filename, text):
        doc = Document(page_content=text, metadata={"filename": filename, "source": filename, "ingest_status": "committed", "ingest_id": "old"})
        self.store.add_documents([doc], ids=["old-1"])
        dest = self.uploads / filename
        dest.write_bytes(TEXT_PDF)
        return dest

    def test_invalid_pdf_keeps_existing_file_and_chunks(self):
        dest = self._seed("note.pdf", "original chunk")
        bad = self.uploads / ".tmp" / "note.pdf"
        bad.parent.mkdir(parents=True, exist_ok=True)
        bad.write_bytes(b"not a pdf")
        result = self.engine.ingest_pdfs([(bad, "note.pdf")])
        self.assertEqual(result["added"], [])
        self.assertTrue(result["skipped"])
        self.assertEqual(dest.read_bytes(), TEXT_PDF)
        self.assertEqual(self.store.documents, ["original chunk"])

    def test_add_failure_keeps_old_chunks(self):
        dest = self._seed("note.pdf", "original chunk")
        temp = self.uploads / ".tmp" / "note.pdf"
        temp.parent.mkdir(parents=True, exist_ok=True)
        temp.write_bytes(TEXT_PDF)
        self.store.fail_add = True
        with self.assertRaises(RuntimeError):
            self.engine._commit_chunks_and_file(
                "note.pdf",
                [Document(page_content="new", metadata={})],
                temp,
                dest,
            )
        self.assertEqual(self.store.documents, ["original chunk"])
        self.assertEqual(dest.read_bytes(), TEXT_PDF)

    def test_successful_replace_swaps_file_and_keeps_only_new_ids(self):
        dest = self._seed("note.pdf", "original chunk")
        temp = self.uploads / ".tmp" / "note.pdf"
        temp.parent.mkdir(parents=True, exist_ok=True)
        temp.write_bytes(TEXT_PDF)
        self.engine._commit_chunks_and_file(
            "note.pdf",
            [Document(page_content="new chunk", metadata={"page_number": 1})],
            temp,
            dest,
        )
        self.assertTrue(dest.exists())
        self.assertFalse(any(doc_id.startswith("old-") for doc_id in self.store.ids))
        self.assertTrue(all(meta.get("ingest_status") == "committed" for meta in self.store.metadatas))
        self.assertTrue(all(meta.get("ingest_id") != "old" for meta in self.store.metadatas))
        self.assertIn("new chunk", self.store.documents)
        self.assertNotIn("original chunk", self.store.documents)

    def test_partial_multi_upload_keeps_success(self):
        self._seed("keep.pdf", "keep me")
        good = self.uploads / ".tmp" / "new.pdf"
        bad = self.uploads / ".tmp" / "bad.pdf"
        good.parent.mkdir(parents=True, exist_ok=True)
        good.write_bytes(TEXT_PDF)
        bad.write_bytes(b"nope")
        self.engine._commit_chunks_and_file(
            "new.pdf",
            [Document(page_content="fresh", metadata={"page_number": 1})],
            good,
            self.uploads / "new.pdf",
        )
        result = self.engine.ingest_pdfs([(bad, "bad.pdf")])
        self.assertEqual(result["added"], [])
        self.assertEqual(len(result["skipped"]), 1)
        self.assertTrue((self.uploads / "keep.pdf").exists())
        self.assertTrue((self.uploads / "new.pdf").exists())
        self.assertIn("keep me", self.store.documents)
        self.assertIn("fresh", self.store.documents)

    def test_pending_chunks_hidden_from_listing(self):
        self.store.add_documents(
            [Document(page_content="hidden", metadata={"filename": "tmp.pdf", "ingest_status": "pending"})],
            ids=["pending-1"],
        )
        self.store.add_documents(
            [Document(page_content="shown", metadata={"filename": "ok.pdf", "ingest_status": "committed"})],
            ids=["ok-1"],
        )
        names = [item["name"] for item in self.engine.list_files()]
        self.assertEqual(names, ["ok.pdf"])


if __name__ == "__main__":
    unittest.main()
