import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.documents import Document

from ingest_journal import list_records, read_record, write_record
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
        self.fail_update = False
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
        if self.fail_update:
            raise RuntimeError("update failed")
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
        self.swap = self.uploads / ".swap"
        self.versions = self.uploads / ".versions"
        self.patchers = [
            patch("rag_engine.UPLOAD_DIR", self.uploads),
            patch("rag_engine.TMP_UPLOAD_DIR", self.uploads / ".tmp"),
            patch("rag_engine.VERSION_DIR", self.versions),
            patch("ingest_journal.SWAP_DIR", self.swap),
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

    def _temp_pdf(self, name="note.pdf"):
        temp = self.uploads / ".tmp" / name
        temp.parent.mkdir(parents=True, exist_ok=True)
        temp.write_bytes(TEXT_PDF)
        return temp

    def _assert_old_preserved(self, dest, text="original chunk"):
        self.assertEqual(dest.read_bytes(), TEXT_PDF)
        self.assertEqual(self.store.documents, [text])
        self.assertFalse(list_records())
        self.assertFalse(read_record(dest.name))

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
        self._assert_old_preserved(dest)

    def test_metadata_update_fail_keeps_old(self):
        dest = self._seed("note.pdf", "original chunk")
        temp = self._temp_pdf()
        self.store.fail_update = True
        with self.assertRaises(RuntimeError):
            self.engine._commit_chunks_and_file(
                "note.pdf",
                [Document(page_content="new", metadata={})],
                temp,
                dest,
            )
        self._assert_old_preserved(dest)

    def test_os_replace_fail_keeps_old(self):
        dest = self._seed("note.pdf", "original chunk")
        temp = self._temp_pdf()
        real_replace = os.replace

        def fail_dest_only(src, dst, *args, **kwargs):
            if Path(dst) == dest:
                raise OSError("replace failed")
            return real_replace(src, dst, *args, **kwargs)

        with patch("os.replace", side_effect=fail_dest_only):
            with self.assertRaises(RuntimeError):
                self.engine._commit_chunks_and_file(
                    "note.pdf",
                    [Document(page_content="new", metadata={})],
                    temp,
                    dest,
                )
        self._assert_old_preserved(dest)

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

    def test_reconcile_file_swapped_promotes(self):
        dest = self._seed("note.pdf", "original chunk")
        dest.write_bytes(b"%PDF-1.4 new-version")
        self.store.add_documents(
            [Document(page_content="new chunk", metadata={"filename": "note.pdf", "ingest_status": "pending", "ingest_id": "new"})],
            ids=["new-1"],
        )
        backup = self.versions
        backup.mkdir(parents=True, exist_ok=True)
        bak = backup / "note.pdf.new.bak.pdf"
        bak.write_bytes(TEXT_PDF)
        write_record(
            "note.pdf",
            {
                "filename": "note.pdf",
                "ingest_id": "new",
                "phase": "file_swapped",
                "dest": str(dest),
                "backup_path": str(bak),
                "new_ids": ["new-1"],
            },
        )
        stats = self.engine._reconcile_pending()
        self.assertEqual(stats["promoted"], 1)
        self.assertEqual(self.store.documents, ["new chunk"])
        self.assertEqual(self.store.metadatas[0]["ingest_status"], "committed")
        self.assertFalse(read_record("note.pdf"))
        self.assertFalse(bak.exists())
        self.assertEqual(dest.read_bytes(), b"%PDF-1.4 new-version")

    def test_reconcile_pending_added_rolls_back(self):
        dest = self._seed("note.pdf", "original chunk")
        self.store.add_documents(
            [Document(page_content="new chunk", metadata={"filename": "note.pdf", "ingest_status": "pending", "ingest_id": "new"})],
            ids=["new-1"],
        )
        self.versions.mkdir(parents=True, exist_ok=True)
        bak = self.versions / "note.pdf.new.bak.pdf"
        bak.write_bytes(TEXT_PDF)
        dest.write_bytes(b"%PDF-1.4 half-replaced")
        write_record(
            "note.pdf",
            {
                "filename": "note.pdf",
                "ingest_id": "new",
                "phase": "pending_added",
                "dest": str(dest),
                "backup_path": str(bak),
                "new_ids": ["new-1"],
            },
        )
        stats = self.engine._reconcile_pending()
        self.assertEqual(stats["rolled_back"], 1)
        self._assert_old_preserved(dest)

    def test_reconcile_pending_without_pdf_does_not_promote(self):
        self.store.add_documents(
            [Document(page_content="orphan", metadata={"filename": "missing.pdf", "ingest_status": "pending", "ingest_id": "x"})],
            ids=["pending-1"],
        )
        stats = self.engine._reconcile_pending()
        self.assertEqual(stats["orphans_removed"], 1)
        self.assertEqual(self.engine.list_files(), [])
        self.assertEqual(self.store.ids, [])
        self.assertFalse((self.uploads / "missing.pdf").exists())

    def test_reconcile_pending_with_dest_no_journal_does_not_promote(self):
        dest = self.uploads / "note.pdf"
        dest.write_bytes(TEXT_PDF)
        self.store.add_documents(
            [Document(page_content="pending only", metadata={"filename": "note.pdf", "ingest_status": "pending", "ingest_id": "x"})],
            ids=["pending-1"],
        )
        stats = self.engine._reconcile_pending()
        self.assertEqual(stats["orphans_removed"], 1)
        self.assertEqual(self.engine.list_files(), [])
        self.assertEqual(self.store.ids, [])
        self.assertTrue(dest.exists())

    def test_reconcile_idempotent(self):
        dest = self._seed("note.pdf", "original chunk")
        dest.write_bytes(b"%PDF-1.4 new-version")
        self.store.add_documents(
            [Document(page_content="new chunk", metadata={"filename": "note.pdf", "ingest_status": "pending", "ingest_id": "new"})],
            ids=["new-1"],
        )
        write_record(
            "note.pdf",
            {
                "filename": "note.pdf",
                "ingest_id": "new",
                "phase": "file_swapped",
                "dest": str(dest),
                "backup_path": None,
                "new_ids": ["new-1"],
            },
        )
        first = self.engine._reconcile_pending()
        second = self.engine._reconcile_pending()
        self.assertEqual(first["promoted"], 1)
        self.assertEqual(second["promoted"], 0)
        self.assertEqual(self.store.documents, ["new chunk"])
        self.assertEqual(self.engine.list_files()[0]["name"], "note.pdf")


if __name__ == "__main__":
    unittest.main()
