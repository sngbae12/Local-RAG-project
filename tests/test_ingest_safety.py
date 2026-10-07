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

NEW_PDF = TEXT_PDF.replace(
    b"Safety replacement document alpha",
    b"Replacement document beta v2!!!!!",
)


class FakeStore:
    def __init__(self):
        self.ids = []
        self.metadatas = []
        self.documents = []
        self.fail_add = False
        self.fail_delete = False
        self.fail_update = False
        self.fail_delete_ids = set()
        self.partial_delete = False
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
        ids = list(ids or [])
        if self.fail_delete:
            raise RuntimeError("delete failed")
        if self.fail_delete_ids and any(doc_id in self.fail_delete_ids for doc_id in ids):
            raise RuntimeError("delete failed")
        if self.partial_delete and ids:
            self.partial_delete = False
            keep = [(i, m, d) for i, m, d in zip(self.ids, self.metadatas, self.documents) if i != ids[0]]
            self.ids = [item[0] for item in keep]
            self.metadatas = [item[1] for item in keep]
            self.documents = [item[2] for item in keep]
            raise RuntimeError("partial delete failed")
        keep = [(i, m, d) for i, m, d in zip(self.ids, self.metadatas, self.documents) if i not in set(ids)]
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

    def _temp_pdf(self, name="note.pdf", data=None):
        temp = self.uploads / ".tmp" / name
        temp.parent.mkdir(parents=True, exist_ok=True)
        temp.write_bytes(data if data is not None else NEW_PDF)
        return temp

    def _visible_texts(self):
        confirmed = self.engine._confirmed_ingest_map()
        return [
            text
            for meta, text in zip(self.store.metadatas, self.store.documents)
            if self.engine._is_visible_meta(meta, confirmed)
        ]

    def _assert_old_preserved(self, dest, text="original chunk"):
        self.assertEqual(dest.read_bytes(), TEXT_PDF)
        self.assertEqual(self._visible_texts(), [text])
        self.assertFalse(list_records())
        self.assertFalse(read_record(dest.name))

    def _assert_new_usable(self, dest, text="new chunk", pdf_bytes=None):
        self.assertEqual(dest.read_bytes(), pdf_bytes or NEW_PDF)
        self.assertEqual(self._visible_texts(), [text])
        names = [item["name"] for item in self.engine.list_files()]
        self.assertEqual(names, [dest.name])
        self.assertNotIn("original chunk", self._visible_texts())

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
        temp = self._temp_pdf()
        warning = self.engine._commit_chunks_and_file(
            "note.pdf",
            [Document(page_content="new chunk", metadata={"page_number": 1})],
            temp,
            dest,
        )
        self.assertIsNone(warning)
        self._assert_new_usable(dest)
        self.assertFalse(any(doc_id.startswith("old-") for doc_id in self.store.ids))
        self.assertTrue(all(meta.get("ingest_status") == "committed" for meta in self.store.metadatas))
        self.assertTrue(all(meta.get("ingest_id") != "old" for meta in self.store.metadatas))
        self.assertFalse(read_record("note.pdf"))

    def test_partial_multi_upload_keeps_success(self):
        self._seed("keep.pdf", "keep me")
        good = self.uploads / ".tmp" / "new.pdf"
        bad = self.uploads / ".tmp" / "bad.pdf"
        good.parent.mkdir(parents=True, exist_ok=True)
        good.write_bytes(TEXT_PDF)
        bad.write_bytes(b"nope")
        good.write_bytes(NEW_PDF)
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

    def test_reconcile_file_swapped_rolls_back(self):
        dest = self._seed("note.pdf", "original chunk")
        dest.write_bytes(NEW_PDF)
        self.store.add_documents(
            [Document(page_content="new chunk", metadata={"filename": "note.pdf", "ingest_status": "pending", "ingest_id": "new"})],
            ids=["new-1"],
        )
        self.versions.mkdir(parents=True, exist_ok=True)
        bak = self.versions / "note.pdf.new.bak.pdf"
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
        self.assertEqual(stats["rolled_back"], 1)
        self._assert_old_preserved(dest)

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
        dest.write_bytes(NEW_PDF)
        self.store.add_documents(
            [Document(page_content="new chunk", metadata={"filename": "note.pdf", "ingest_status": "committed", "ingest_id": "new"})],
            ids=["new-1"],
        )
        write_record(
            "note.pdf",
            {
                "filename": "note.pdf",
                "ingest_id": "new",
                "phase": "accepted",
                "dest": str(dest),
                "backup_path": None,
                "new_ids": ["new-1"],
                "old_ids": ["old-1"],
            },
        )
        first = self.engine._reconcile_pending()
        second = self.engine._reconcile_pending()
        self.assertEqual(first["cleaned"], 1)
        self.assertEqual(second["cleaned"], 0)
        self.assertEqual(second["cleanup_pending"], 0)
        self._assert_new_usable(dest)
        self.assertFalse(read_record("note.pdf"))

    def test_accept_record_write_fail_keeps_old(self):
        dest = self._seed("note.pdf", "original chunk")
        temp = self._temp_pdf()
        real_write = write_record

        def fail_accepted(filename, record):
            if record.get("phase") == "accepted":
                raise OSError("journal write failed")
            return real_write(filename, record)

        with patch("rag_engine.write_record", side_effect=fail_accepted):
            with self.assertRaises(RuntimeError) as caught:
                self.engine._commit_chunks_and_file(
                    "note.pdf",
                    [Document(page_content="new chunk", metadata={})],
                    temp,
                    dest,
                )
        self.assertIn("이전 문서를 유지합니다", str(caught.exception))
        self._assert_old_preserved(dest)

    def test_old_chunk_delete_fail_keeps_new(self):
        dest = self._seed("note.pdf", "original chunk")
        temp = self._temp_pdf()
        self.store.fail_delete_ids = {"old-1"}
        warning = self.engine._commit_chunks_and_file(
            "note.pdf",
            [Document(page_content="new chunk", metadata={})],
            temp,
            dest,
        )
        self.assertIsNotNone(warning)
        self.assertIn("새 버전으로 저장", warning)
        self.assertTrue(read_record("note.pdf"))
        self.assertEqual(read_record("note.pdf")["phase"], "cleanup")
        self._assert_new_usable(dest)
        self.assertIn("original chunk", self.store.documents)
        self.assertNotIn("original chunk", self._visible_texts())

    def test_partial_old_delete_fail_keeps_new(self):
        dest = self._seed("note.pdf", "original chunk")
        self.store.add_documents(
            [Document(page_content="original extra", metadata={"filename": "note.pdf", "source": "note.pdf", "ingest_status": "committed", "ingest_id": "old"})],
            ids=["old-2"],
        )
        temp = self._temp_pdf()
        self.store.partial_delete = True
        warning = self.engine._commit_chunks_and_file(
            "note.pdf",
            [Document(page_content="new chunk", metadata={})],
            temp,
            dest,
        )
        self.assertIsNotNone(warning)
        self._assert_new_usable(dest)
        leftover_old = [text for text in self.store.documents if text.startswith("original")]
        self.assertTrue(leftover_old)
        self.assertEqual(self._visible_texts(), ["new chunk"])
        record = read_record("note.pdf")
        self.assertEqual(record["phase"], "cleanup")
        self.assertTrue(record.get("old_ids"))

    def test_backup_delete_fail_keeps_new(self):
        dest = self._seed("note.pdf", "original chunk")
        temp = self._temp_pdf()
        original_unlink = Path.unlink

        def fail_backup(self, missing_ok=False):
            if self.name.endswith(".bak.pdf"):
                raise OSError("backup locked")
            return original_unlink(self, missing_ok=missing_ok)

        with patch.object(Path, "unlink", fail_backup):
            warning = self.engine._commit_chunks_and_file(
                "note.pdf",
                [Document(page_content="new chunk", metadata={})],
                temp,
                dest,
            )
        self.assertIsNotNone(warning)
        self._assert_new_usable(dest)
        record = read_record("note.pdf")
        self.assertEqual(record["phase"], "cleanup")
        self.assertTrue(record.get("backup_path"))
        self.assertTrue(Path(record["backup_path"]).exists())

    def test_journal_clear_fail_keeps_new(self):
        dest = self._seed("note.pdf", "original chunk")
        temp = self._temp_pdf()
        with patch("rag_engine.clear_record", side_effect=OSError("journal locked")):
            warning = self.engine._commit_chunks_and_file(
                "note.pdf",
                [Document(page_content="new chunk", metadata={})],
                temp,
                dest,
            )
        self.assertIsNotNone(warning)
        self._assert_new_usable(dest)
        self.assertTrue(read_record("note.pdf"))
        self.assertNotIn("original chunk", self.store.documents)

    def test_reconcile_accepted_continues_cleanup(self):
        dest = self._seed("note.pdf", "original chunk")
        dest.write_bytes(NEW_PDF)
        self.store.add_documents(
            [Document(page_content="new chunk", metadata={"filename": "note.pdf", "ingest_status": "committed", "ingest_id": "new"})],
            ids=["new-1"],
        )
        write_record(
            "note.pdf",
            {
                "filename": "note.pdf",
                "ingest_id": "new",
                "phase": "accepted",
                "dest": str(dest),
                "backup_path": None,
                "new_ids": ["new-1"],
                "old_ids": ["old-1"],
            },
        )
        stats = self.engine._reconcile_pending()
        self.assertEqual(stats["cleaned"], 1)
        self._assert_new_usable(dest)
        self.assertFalse(read_record("note.pdf"))

    def test_cleanup_retry_and_repeat_reconcile(self):
        dest = self._seed("note.pdf", "original chunk")
        dest.write_bytes(NEW_PDF)
        self.store.add_documents(
            [Document(page_content="new chunk", metadata={"filename": "note.pdf", "ingest_status": "committed", "ingest_id": "new"})],
            ids=["new-1"],
        )
        self.versions.mkdir(parents=True, exist_ok=True)
        bak = self.versions / "note.pdf.new.bak.pdf"
        bak.write_bytes(TEXT_PDF)
        write_record(
            "note.pdf",
            {
                "filename": "note.pdf",
                "ingest_id": "new",
                "phase": "cleanup",
                "dest": str(dest),
                "backup_path": str(bak),
                "new_ids": ["new-1"],
                "old_ids": ["old-1"],
                "cleanup_error": "old_chunks",
            },
        )
        self.store.fail_delete_ids = {"old-1"}
        first = self.engine._reconcile_pending()
        self.assertEqual(first["cleanup_pending"], 1)
        self._assert_new_usable(dest)
        self.assertTrue(read_record("note.pdf"))

        self.store.fail_delete_ids = set()
        second = self.engine._reconcile_pending()
        self.assertEqual(second["cleaned"], 1)
        third = self.engine._reconcile_pending()
        self.assertEqual(third["cleaned"], 0)
        self._assert_new_usable(dest)
        self.assertFalse(read_record("note.pdf"))
        self.assertFalse(bak.exists())

    def test_ingest_reports_cleanup_warning_not_skip(self):
        dest = self._seed("note.pdf", "original chunk")
        temp = self._temp_pdf()
        self.store.fail_delete_ids = {"old-1"}

        def ingest_without_loader(temp_path, filename, progress=None):
            warning = self.engine._commit_chunks_and_file(
                filename,
                [Document(page_content="new chunk", metadata={"page_number": 1})],
                temp_path,
                dest,
            )
            return 1, warning

        with patch.object(self.engine, "_ingest_one", side_effect=ingest_without_loader):
            result = self.engine.ingest_pdfs([(temp, "note.pdf")])
        self.assertEqual(result["added"], ["note.pdf"])
        self.assertEqual(result["skipped"], [])
        self.assertTrue(result["warnings"])
        self.assertIn("새 버전으로 저장", result["warnings"][0]["reason"])
        self._assert_new_usable(dest)


if __name__ == "__main__":
    unittest.main()
