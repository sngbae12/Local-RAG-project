import logging
import os
import shutil
import site
import threading
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from langchain_chroma import Chroma
from langchain_classic.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    DISTANCE_THRESHOLD,
    EMBEDDING_PATH,
    LLM_PATH,
    NO_INFO_MESSAGE,
    RETRIEVE_K,
    SEPARATORS,
    SWAP_DIR,
    TMP_UPLOAD_DIR,
    UPLOAD_DIR,
    VERSION_DIR,
)
from ingest_journal import clear_record, list_records, read_record, write_record

logger = logging.getLogger(__name__)

PRE_CONFIRM_PHASES = {"started", "pending_added", "file_swapped"}
CONFIRMED_PHASES = {"accepted", "cleanup", "committed"}
CLEANUP_WARNING = (
    "문서는 새 버전으로 저장됐지만 이전 데이터 정리에 실패했습니다. "
    "서버를 다시 시작하면 정리를 이어서 시도합니다."
)

SYSTEM_PROMPT = (
    "당신은 PDF 문서 기반 질의응답 도우미입니다. "
    "반드시 아래 [참고 문서] 내용만으로 답하세요. "
    "참고 문서에 질문의 답이 없으면 다른 말을 하지 말고 정확히 다음 한 문장만 출력하세요: "
    f"{NO_INFO_MESSAGE} "
    "추측, 일반 지식, 문서에 없는 설명을 금지합니다. "
    "한국어로 간결하고 정확하게 답하세요."
)


def _prepare_llama_dll_path() -> None:
    """Windows에서 llama.cpp DLL이 같은 폴더의 ggml*.dll을 찾도록 PATH를 보강한다."""
    candidates: List[Path] = []
    for sp in list(site.getsitepackages()) + [site.getusersitepackages()]:
        lib = Path(sp) / "llama_cpp" / "lib"
        if lib.exists():
            candidates.append(lib)
    appdata = Path(os.environ.get("APPDATA", "")) / "Python"
    if appdata.exists():
        for lib in appdata.glob("Python*/site-packages/llama_cpp/lib"):
            if lib.exists():
                candidates.append(lib)

    for lib in candidates:
        lib_str = str(lib)
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(lib_str)
        os.environ["PATH"] = lib_str + os.pathsep + os.environ.get("PATH", "")


def _page_number(metadata: Dict[str, Any]) -> int:
    page = metadata.get("page_number")
    if page is not None:
        try:
            return int(page)
        except (TypeError, ValueError):
            pass
    page = metadata.get("page", 0)
    try:
        return int(page) + 1
    except (TypeError, ValueError):
        return 1


def _excerpt(text: str, limit: int = 180) -> str:
    cleaned = " ".join((text or "").split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit].rstrip() + "…"


def _format_history(messages: List[Any]) -> str:
    if not messages:
        return "(없음)"
    lines: List[str] = []
    for msg in messages[-8:]:
        role = getattr(msg, "type", "")
        content = getattr(msg, "content", "")
        if role == "human":
            lines.append(f"사용자: {content}")
        elif role == "ai":
            lines.append(f"도우미: {content}")
    return "\n".join(lines) if lines else "(없음)"


def _build_prompt(question: str, context: str, history: str) -> str:
    user_block = (
        f"[이전 대화]\n{history}\n\n"
        f"[참고 문서]\n{context}\n\n"
        f"[질문]\n{question}"
    )
    return (
        f"[|system|]{SYSTEM_PROMPT}[|endofturn|]"
        f"[|user|]{user_block}[|endofturn|]"
        "[|assistant|]"
    )


class RAGEngine:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.embeddings: Optional[HuggingFaceEmbeddings] = None
        self.llm: Optional[LlamaCpp] = None
        self.vectorstore: Optional[Chroma] = None
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=SEPARATORS,
            length_function=len,
        )
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
        )
        self.ready = False
        self.llm_ready = False
        self.embeddings_ready = False
        self.error: Optional[str] = None
        self.embedding_error: Optional[str] = None
        self.llm_error: Optional[str] = None
        self.initializing = False
        self._store_lock = threading.RLock()

    def initialize(self) -> None:
        with self._lock:
            if self.ready:
                return
            if self.initializing:
                return
            self.initializing = True
        try:
            self._initialize_body()
        finally:
            with self._lock:
                self.initializing = False

    def _initialize_body(self) -> None:
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        TMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        SWAP_DIR.mkdir(parents=True, exist_ok=True)
        VERSION_DIR.mkdir(parents=True, exist_ok=True)
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)

        if self.embeddings is None:
            try:
                self._load_embeddings()
                self.embeddings_ready = True
                self.embedding_error = None
            except Exception as exc:
                self.embeddings_ready = False
                self.embedding_error = f"임베딩 모델 로딩 실패: {exc}"
                self.error = self.embedding_error
                raise
        if self.vectorstore is None:
            self._load_vectorstore()
            stats = self._reconcile_pending()
            logger.info("ingest reconcile %s", stats)

        if self.llm is None:
            try:
                self._load_llm()
                self.llm_ready = True
                self.llm_error = None
            except Exception as exc:
                self.llm_ready = False
                detail = str(exc)
                if "llama.dll" in detail or "shared library" in detail.lower():
                    self.llm_error = (
                        "llama-cpp-python이 필요한 DLL을 찾지 못해 GGUF LLM을 불러오지 못했습니다. "
                        "CPU용 휠을 설치하세요: pip install llama-cpp-python==0.3.33 "
                        "--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu"
                    )
                else:
                    self.llm_error = f"LLM 로딩 실패: {detail}"
                self.error = self.llm_error
                self.ready = False
                return

        self.ready = True
        self.error = None

    def _load_embeddings(self) -> None:
        if not EMBEDDING_PATH.exists():
            raise FileNotFoundError(f"임베딩 모델을 찾을 수 없습니다: {EMBEDDING_PATH}")
        # bge-m3는 HuggingFace/SentenceTransformer 형식이다.
        # llama-cpp-python의 LlamaCppEmbeddings는 GGUF만 로드할 수 있어
        # 로컬 bge-m3 폴더를 HuggingFaceEmbeddings로 로드한다.
        self.embeddings = HuggingFaceEmbeddings(
            model_name=str(EMBEDDING_PATH),
            model_kwargs={"device": "cpu", "local_files_only": True},
            encode_kwargs={"normalize_embeddings": True},
        )

    def _load_vectorstore(self) -> None:
        self.vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=str(CHROMA_DIR),
            collection_metadata={"hnsw:space": "cosine"},
        )

    def _load_llm(self) -> None:
        if not LLM_PATH.exists():
            raise FileNotFoundError(f"LLM GGUF 파일을 찾을 수 없습니다: {LLM_PATH}")
        _prepare_llama_dll_path()
        self.llm = LlamaCpp(
            model_path=str(LLM_PATH),
            n_ctx=4096,
            n_batch=256,
            n_gpu_layers=0,
            max_tokens=512,
            temperature=0.2,
            top_p=0.9,
            repeat_penalty=1.1,
            stop=["[|endofturn|]", "[|user|]", "[|system|]"],
            verbose=False,
            streaming=True,
        )

    def _collection_count(self) -> int:
        if self.vectorstore is None:
            return 0
        try:
            return int(self.vectorstore._collection.count())
        except Exception:
            return 0

    def _confirmed_ingest_map(self) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        for record in list_records():
            phase = record.get("phase")
            filename = record.get("filename")
            ingest_id = record.get("ingest_id")
            if phase in CONFIRMED_PHASES and filename and ingest_id:
                mapping[filename] = ingest_id
        return mapping

    def _is_visible_meta(
        self,
        meta: Optional[Dict[str, Any]],
        confirmed: Optional[Dict[str, str]] = None,
    ) -> bool:
        meta = meta or {}
        status = meta.get("ingest_status") or "committed"
        if status != "committed":
            return False
        filename = meta.get("filename") or Path(str(meta.get("source", ""))).name
        if confirmed is None:
            confirmed = self._confirmed_ingest_map()
        active = confirmed.get(filename) if filename else None
        if active:
            return meta.get("ingest_id") == active
        return True

    def list_files(self) -> List[Dict[str, Any]]:
        if self.vectorstore is None:
            return []
        with self._store_lock:
            try:
                data = self.vectorstore.get(include=["metadatas"])
            except Exception:
                return []
            confirmed = self._confirmed_ingest_map()
        counts: Dict[str, int] = {}
        for meta in data.get("metadatas") or []:
            if not self._is_visible_meta(meta, confirmed):
                continue
            name = (meta or {}).get("filename") or Path((meta or {}).get("source", "")).name
            if name:
                counts[name] = counts.get(name, 0) + 1
        return [{"name": name, "chunks": counts[name]} for name in sorted(counts)]

    def ingest_pdfs(
        self,
        items: List[Tuple[Path, str]],
        progress: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        if self.vectorstore is None:
            raise RuntimeError("벡터 DB가 아직 준비되지 않았습니다.")

        added_files: List[str] = []
        skipped: List[Dict[str, str]] = []
        warnings: List[Dict[str, str]] = []
        total_chunks = 0

        for temp_path, filename in items:
            try:
                chunk_count, warning = self._ingest_one(temp_path, filename, progress)
                added_files.append(filename)
                total_chunks += chunk_count
                if warning:
                    warnings.append({"name": filename, "reason": warning})
            except Exception as exc:
                skipped.append({"name": filename, "reason": str(exc)})

        return {
            "added": added_files,
            "skipped": skipped,
            "warnings": warnings,
            "chunk_count": total_chunks,
            "files": self.list_files(),
        }

    def _notify(self, progress: Optional[Callable[[str], None]], stage: str) -> None:
        if progress:
            progress(stage)

    def _ingest_one(
        self,
        temp_path: Path,
        filename: str,
        progress: Optional[Callable[[str], None]] = None,
    ) -> Tuple[int, Optional[str]]:
        self._notify(progress, "extract")
        loader = PyPDFLoader(str(temp_path))
        pages = loader.load()
        if not pages or not any((p.page_content or "").strip() for p in pages):
            raise RuntimeError("텍스트를 추출할 수 없는 PDF입니다.")

        for page in pages:
            page.metadata["filename"] = filename
            page.metadata["source"] = filename
            page.metadata["page_number"] = _page_number(page.metadata)

        self._notify(progress, "chunk")
        chunks = self.splitter.split_documents(pages)
        chunks = [c for c in chunks if (c.page_content or "").strip()]
        if not chunks:
            raise RuntimeError("분할할 텍스트가 없습니다.")

        dest = UPLOAD_DIR / filename
        warning = self._commit_chunks_and_file(filename, chunks, temp_path, dest, progress)
        return len(chunks), warning

    def _commit_chunks_and_file(
        self,
        filename: str,
        chunks: List[Document],
        temp_path: Path,
        dest: Path,
        progress: Optional[Callable[[str], None]] = None,
    ) -> Optional[str]:
        assert self.vectorstore is not None
        ingest_id = uuid.uuid4().hex
        new_ids = [f"ing-{ingest_id}-{idx:06d}" for idx in range(len(chunks))]
        for chunk in chunks:
            chunk.metadata["filename"] = filename
            chunk.metadata["source"] = filename
            chunk.metadata["ingest_id"] = ingest_id
            chunk.metadata["ingest_status"] = "pending"

        record: Dict[str, Any] = {
            "filename": filename,
            "ingest_id": ingest_id,
            "phase": "started",
            "dest": str(dest),
            "backup_path": None,
            "new_ids": new_ids,
            "old_ids": [],
        }
        write_record(filename, record)
        file_swapped = False

        with self._store_lock:
            try:
                if dest.exists():
                    VERSION_DIR.mkdir(parents=True, exist_ok=True)
                    backup = VERSION_DIR / f"{filename}.{ingest_id}.bak.pdf"
                    shutil.copy2(dest, backup)
                    record["backup_path"] = str(backup)
                    write_record(filename, record)

                self._notify(progress, "embed")
                self.vectorstore.add_documents(chunks, ids=new_ids)
                record["phase"] = "pending_added"
                write_record(filename, record)

                self._notify(progress, "store")
                dest.parent.mkdir(parents=True, exist_ok=True)
                os.replace(temp_path, dest)
                file_swapped = True
                record["phase"] = "file_swapped"
                write_record(filename, record)

                self._mark_committed(new_ids, filename, ingest_id)
                old_ids = self._leftover_ids(filename, ingest_id)
                record["old_ids"] = old_ids
                record["phase"] = "accepted"
                write_record(filename, record)
            except Exception as exc:
                try:
                    self._rollback_replace(filename, new_ids, record, dest, file_swapped)
                except Exception:
                    raise
                raise RuntimeError(f"문서 교체에 실패해 이전 문서를 유지합니다: {exc}") from exc

            try:
                return self._cleanup_accepted(filename, record)
            except Exception as exc:
                logger.exception("post-accept cleanup failed for %s", filename)
                record["phase"] = "cleanup"
                record["cleanup_error"] = str(exc)
                try:
                    write_record(filename, record)
                except Exception:
                    logger.exception("could not persist cleanup record for %s", filename)
                return CLEANUP_WARNING

    def _delete_ids_quiet(self, ids: List[str]) -> None:
        if not ids or self.vectorstore is None:
            return
        self.vectorstore.delete(ids=ids)

    def _rollback_replace(
        self,
        filename: str,
        new_ids: List[str],
        record: Dict[str, Any],
        dest: Path,
        file_swapped: bool,
    ) -> None:
        pending_error = None
        file_error = None
        try:
            self._delete_ids_quiet(new_ids)
        except Exception as exc:
            pending_error = exc

        if file_swapped:
            backup = record.get("backup_path")
            if backup and Path(backup).exists():
                try:
                    os.replace(backup, dest)
                except Exception as exc:
                    file_error = exc
                    record["phase"] = "rollback_failed"
                    record["rollback_error"] = "file_restore_failed"
                    write_record(filename, record)
            elif dest.exists() and not backup:
                try:
                    dest.unlink()
                except Exception as exc:
                    file_error = exc
                    record["phase"] = "rollback_failed"
                    record["rollback_error"] = "new_file_delete_failed"
                    write_record(filename, record)

        if pending_error:
            record["phase"] = "rollback_failed"
            record["rollback_error"] = "pending_delete_failed"
            write_record(filename, record)
            raise RuntimeError(
                f"문서 교체 실패 후 새 chunk를 정리하지 못했습니다. 교체 기록을 보존합니다: {pending_error}"
            ) from pending_error
        if file_error:
            backup = record.get("backup_path")
            raise RuntimeError(
                "검색 데이터는 이전 문서로 되돌렸지만 PDF 복구에 실패했습니다. "
                f"백업 경로를 보존합니다: {backup}"
            ) from file_error

        backup = record.get("backup_path")
        if backup and Path(backup).exists() and dest.exists() and not file_swapped:
            Path(backup).unlink(missing_ok=True)
        if record.get("phase") != "rollback_failed":
            clear_record(filename)

    def _ids_for_ingest(self, filename: str, ingest_id: str) -> List[str]:
        assert self.vectorstore is not None
        try:
            existing = self.vectorstore.get(where={"filename": filename}, include=["metadatas"])
        except Exception:
            return []
        ids = existing.get("ids") or []
        metas = existing.get("metadatas") or []
        return [doc_id for doc_id, meta in zip(ids, metas) if (meta or {}).get("ingest_id") == ingest_id]

    def _leftover_ids(self, filename: str, ingest_id: str) -> List[str]:
        assert self.vectorstore is not None
        try:
            existing = self.vectorstore.get(where={"filename": filename}, include=["metadatas"])
        except Exception:
            return []
        ids = existing.get("ids") or []
        metas = existing.get("metadatas") or []
        leftover = []
        for doc_id, meta in zip(ids, metas):
            if (meta or {}).get("ingest_id") != ingest_id:
                leftover.append(doc_id)
        return leftover

    def _existing_ids_for_file(self, filename: str, exclude_ids: Optional[set] = None) -> List[str]:
        assert self.vectorstore is not None
        exclude_ids = exclude_ids or set()
        try:
            existing = self.vectorstore.get(where={"filename": filename})
        except Exception:
            return []
        ids = existing.get("ids") or []
        return [doc_id for doc_id in ids if doc_id not in exclude_ids]

    def _mark_committed(self, ids: List[str], filename: str, ingest_id: str) -> None:
        self._update_metas(ids, ingest_status="committed", filename=filename, ingest_id=ingest_id)

    def _mark_superseded(self, ids: List[str]) -> None:
        self._update_metas(ids, ingest_status="superseded")

    def _update_metas(
        self,
        ids: List[str],
        ingest_status: str,
        filename: Optional[str] = None,
        ingest_id: Optional[str] = None,
    ) -> None:
        assert self.vectorstore is not None
        if not ids:
            return
        try:
            data = self.vectorstore.get(ids=ids, include=["metadatas"])
            found_ids = data.get("ids") or []
            metas = data.get("metadatas") or []
        except Exception:
            found_ids = ids
            metas = [None] * len(ids)
        if not found_ids:
            return
        updated = []
        for meta in metas:
            item = dict(meta or {})
            if filename:
                item["filename"] = filename
                item["source"] = filename
            if ingest_id:
                item["ingest_id"] = ingest_id
            item["ingest_status"] = ingest_status
            updated.append(item)
        self.vectorstore._collection.update(ids=found_ids, metadatas=updated)

    def _cleanup_accepted(self, filename: str, record: Dict[str, Any]) -> Optional[str]:
        ingest_id = record.get("ingest_id") or ""
        leftover: List[str] = []
        remaining_old = self._leftover_ids(filename, ingest_id)
        record["old_ids"] = remaining_old

        if remaining_old:
            try:
                self._mark_superseded(remaining_old)
            except Exception as exc:
                leftover.append(f"supersede:{exc}")
            try:
                self._delete_ids_quiet(remaining_old)
                remaining_old = self._leftover_ids(filename, ingest_id)
                record["old_ids"] = remaining_old
                if remaining_old:
                    leftover.append("old_chunks")
            except Exception as exc:
                remaining_old = self._leftover_ids(filename, ingest_id)
                record["old_ids"] = remaining_old
                leftover.append(f"old_chunks:{exc}")

        backup = record.get("backup_path")
        if backup:
            backup_path = Path(backup)
            try:
                backup_path.unlink(missing_ok=True)
                if backup_path.exists():
                    leftover.append("backup")
                else:
                    record["backup_path"] = None
            except Exception as exc:
                leftover.append(f"backup:{exc}")

        if leftover:
            record["phase"] = "cleanup"
            record["cleanup_error"] = ";".join(leftover)
            write_record(filename, record)
            return f"{CLEANUP_WARNING} ({'; '.join(leftover)})"

        try:
            clear_record(filename)
        except Exception as exc:
            record["phase"] = "cleanup"
            record["cleanup_error"] = f"journal:{exc}"
            try:
                write_record(filename, record)
            except Exception:
                logger.exception("could not keep cleanup journal for %s", filename)
            return f"{CLEANUP_WARNING} (journal:{exc})"
        return None

    def _finish_confirmed_replace(self, record: Dict[str, Any]) -> Optional[str]:
        filename = record["filename"]
        ingest_id = record["ingest_id"]
        dest = Path(record.get("dest") or (UPLOAD_DIR / filename))
        if not dest.exists():
            raise RuntimeError("확정된 교체의 PDF가 없습니다. 교체 기록과 백업을 보존합니다.")
        new_ids = record.get("new_ids") or self._ids_for_ingest(filename, ingest_id)
        if new_ids:
            self._mark_committed(new_ids, filename, ingest_id)
        elif not self._ids_for_ingest(filename, ingest_id):
            raise RuntimeError("확정된 교체의 새 chunk를 찾지 못했습니다. 교체 기록을 보존합니다.")
        return self._cleanup_accepted(filename, record)

    def _reconcile_pending(self) -> Dict[str, int]:
        stats = {
            "promoted": 0,
            "rolled_back": 0,
            "cleaned": 0,
            "cleanup_pending": 0,
            "orphans_removed": 0,
            "unchanged": 0,
            "errors": 0,
        }
        if self.vectorstore is None:
            return stats
        with self._store_lock:
            for record in list_records():
                filename = record.get("filename") or ""
                phase = record.get("phase")
                try:
                    if phase in CONFIRMED_PHASES:
                        warning = self._finish_confirmed_replace(record)
                        if warning:
                            stats["cleanup_pending"] += 1
                        else:
                            stats["cleaned"] += 1
                    elif phase in PRE_CONFIRM_PHASES or phase == "rollback_failed":
                        new_ids = record.get("new_ids") or self._ids_for_ingest(filename, record.get("ingest_id", ""))
                        dest = Path(record.get("dest") or (UPLOAD_DIR / filename))
                        backup = record.get("backup_path")
                        should_restore = bool(backup and Path(backup).exists())
                        self._rollback_replace(filename, new_ids, record, dest, file_swapped=should_restore)
                        stats["rolled_back"] += 1
                    else:
                        stats["unchanged"] += 1
                except Exception:
                    stats["errors"] += 1
                    logger.exception("ingest reconcile failed for %s", filename)

            try:
                data = self.vectorstore.get(include=["metadatas"])
            except Exception:
                return stats
            ids = data.get("ids") or []
            metas = data.get("metadatas") or []
            by_file: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
            for doc_id, meta in zip(ids, metas):
                meta = meta or {}
                name = meta.get("filename") or Path(str(meta.get("source", ""))).name
                if not name:
                    continue
                by_file.setdefault(name, []).append((doc_id, meta))
            for filename, items in by_file.items():
                if read_record(filename):
                    stats["unchanged"] += 1
                    continue
                pending_ids = [
                    doc_id
                    for doc_id, meta in items
                    if (meta.get("ingest_status") == "pending")
                ]
                if not pending_ids:
                    continue
                try:
                    self._delete_ids_quiet(pending_ids)
                    stats["orphans_removed"] += 1
                    logger.info("removed orphan pending chunks filename=%s count=%s", filename, len(pending_ids))
                except Exception:
                    stats["errors"] += 1
                    logger.exception("orphan pending cleanup failed for %s", filename)
        return stats

    def _search_query(self, question: str) -> str:
        messages = self.memory.buffer_as_messages
        last_human = ""
        for msg in reversed(messages):
            if getattr(msg, "type", "") == "human":
                last_human = getattr(msg, "content", "")
                break
        if last_human:
            return f"{last_human}\n{question}"
        return question

    def retrieve(self, question: str) -> List[Tuple[Document, float]]:
        if self.vectorstore is None or self._collection_count() == 0:
            return []
        query = self._search_query(question)
        with self._store_lock:
            results = self.vectorstore.similarity_search_with_score(query, k=max(RETRIEVE_K * 3, RETRIEVE_K))
            confirmed = self._confirmed_ingest_map()
        visible = [
            (doc, float(score))
            for doc, score in results
            if self._is_visible_meta(doc.metadata, confirmed) and float(score) <= DISTANCE_THRESHOLD
        ]
        return visible[:RETRIEVE_K]

    def sources_payload(self, pairs: List[Tuple[Document, float]]) -> List[Dict[str, Any]]:
        payload: List[Dict[str, Any]] = []
        for doc, score in pairs:
            meta = doc.metadata or {}
            payload.append(
                {
                    "filename": meta.get("filename") or Path(str(meta.get("source", ""))).name,
                    "page": _page_number(meta),
                    "excerpt": _excerpt(doc.page_content),
                    "content": doc.page_content,
                    "score": round(float(score), 4),
                }
            )
        return payload

    def reset_memory(self) -> None:
        self.memory.clear()

    def answer_stream(self, question: str) -> Iterator[Dict[str, Any]]:
        pairs = self.retrieve(question)
        sources = self.sources_payload(pairs)

        if not pairs:
            self.memory.save_context({"input": question}, {"output": NO_INFO_MESSAGE})
            yield {"type": "meta", "sources": [], "no_info": True}
            yield {"type": "token", "text": NO_INFO_MESSAGE}
            yield {"type": "done", "answer": NO_INFO_MESSAGE}
            return

        if self.llm is None:
            raise RuntimeError("LLM이 아직 준비되지 않았습니다.")

        context_blocks = []
        for idx, (doc, _) in enumerate(pairs, start=1):
            meta = doc.metadata or {}
            filename = meta.get("filename") or "문서"
            page = _page_number(meta)
            context_blocks.append(
                f"[문서 {idx}] 파일: {filename} / 페이지: {page}\n{doc.page_content.strip()}"
            )
        context = "\n\n".join(context_blocks)
        history = _format_history(self.memory.buffer_as_messages)
        prompt = _build_prompt(question, context, history)

        yield {"type": "meta", "sources": sources, "no_info": False}

        collected = []
        for chunk in self.llm.stream(prompt):
            text = chunk if isinstance(chunk, str) else str(chunk)
            if not text:
                continue
            collected.append(text)
            yield {"type": "token", "text": text}

        answer = "".join(collected).strip() or NO_INFO_MESSAGE
        if "정보가 없어서 답변할 수 없습니다" in answer and len(answer) < 40:
            answer = NO_INFO_MESSAGE
        self.memory.save_context({"input": question}, {"output": answer})
        yield {"type": "done", "answer": answer}


_engine: Optional[RAGEngine] = None
_engine_lock = threading.Lock()


def get_engine() -> RAGEngine:
    global _engine
    with _engine_lock:
        if _engine is None:
            _engine = RAGEngine()
        return _engine
