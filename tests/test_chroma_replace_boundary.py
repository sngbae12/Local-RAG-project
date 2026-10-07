import gc
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_chroma import Chroma
from langchain_core.documents import Document

from ingest_journal import read_record, write_record
from rag_engine import RAGEngine

try:
    from test_ingest_safety import NEW_PDF, TEXT_PDF
except ImportError:
    from tests.test_ingest_safety import NEW_PDF, TEXT_PDF


class HashEmbeddings:
    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        values = [0.0] * 8
        for idx, byte in enumerate((text or "").encode("utf-8")):
            values[idx % 8] += (byte + 1) / 256.0
        norm = sum(item * item for item in values) ** 0.5 or 1.0
        return [item / norm for item in values]


class ChromaReplaceBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.root = Path(self.tmpdir.name)
        self.uploads = self.root / "uploads"
        self.chroma = self.root / "chroma_db"
        self.uploads.mkdir()
        self.chroma.mkdir()
        self.swap = self.uploads / ".swap"
        self.versions = self.uploads / ".versions"
        self.engine = RAGEngine()
        self.engine.embeddings_ready = True
        self.engine.vectorstore = Chroma(
            collection_name="pdf_rag_test",
            embedding_function=HashEmbeddings(),
            persist_directory=str(self.chroma),
            collection_metadata={"hnsw:space": "cosine"},
        )
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
        self.engine.vectorstore = None
        gc.collect()
        self.tmpdir.cleanup()

    def _seed(self, filename="note.pdf", text="original chunk"):
        dest = self.uploads / filename
        dest.write_bytes(TEXT_PDF)
        self.engine.vectorstore.add_documents(
            [Document(page_content=text, metadata={"filename": filename, "source": filename, "ingest_status": "committed", "ingest_id": "old"})],
            ids=["old-1"],
        )
        return dest

    def _temp_pdf(self):
        temp = self.uploads / ".tmp" / "note.pdf"
        temp.parent.mkdir(parents=True, exist_ok=True)
        temp.write_bytes(NEW_PDF)
        return temp

    def _visible_texts(self):
        data = self.engine.vectorstore.get(include=["documents", "metadatas"])
        confirmed = self.engine._confirmed_ingest_map()
        return [
            text
            for text, meta in zip(data.get("documents") or [], data.get("metadatas") or [])
            if self.engine._is_visible_meta(meta, confirmed)
        ]

    def test_accept_write_fail_does_not_empty_chroma(self):
        dest = self._seed()
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
        self.assertEqual(dest.read_bytes(), TEXT_PDF)
        self.assertGreater(self.engine.vectorstore._collection.count(), 0)
        self.assertEqual(self._visible_texts(), ["original chunk"])
        self.assertEqual(self.engine.list_files()[0]["name"], "note.pdf")

    def test_old_delete_fail_keeps_new_version_only_in_search(self):
        dest = self._seed()
        temp = self._temp_pdf()
        real_delete = self.engine.vectorstore.delete

        def fail_old(ids=None):
            ids = list(ids or [])
            if any(doc_id.startswith("old-") for doc_id in ids):
                raise RuntimeError("delete failed")
            return real_delete(ids=ids)

        self.engine.vectorstore.delete = fail_old
        warning = self.engine._commit_chunks_and_file(
            "note.pdf",
            [Document(page_content="new chunk", metadata={"page_number": 1})],
            temp,
            dest,
        )
        self.assertIsNotNone(warning)
        self.assertIn("새 버전으로 저장", warning)
        self.assertEqual(dest.read_bytes(), NEW_PDF)
        self.assertEqual(self._visible_texts(), ["new chunk"])
        self.assertNotIn("original chunk", self._visible_texts())
        self.assertGreaterEqual(self.engine.vectorstore._collection.count(), 2)
        names = [item["name"] for item in self.engine.list_files()]
        self.assertEqual(names, ["note.pdf"])
        self.assertEqual(self.engine.list_files()[0]["chunks"], 1)

    def test_restart_cleanup_after_accepted(self):
        dest = self._seed()
        dest.write_bytes(NEW_PDF)
        self.engine.vectorstore.add_documents(
            [Document(page_content="new chunk", metadata={"filename": "note.pdf", "source": "note.pdf", "ingest_status": "committed", "ingest_id": "new"})],
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
        self.assertEqual(dest.read_bytes(), NEW_PDF)
        self.assertEqual(self._visible_texts(), ["new chunk"])
        self.assertIsNone(read_record("note.pdf"))


if __name__ == "__main__":
    unittest.main()
