import os
import site
import threading
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

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
    UPLOAD_DIR,
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

    def initialize(self) -> None:
        with self._lock:
            if self.ready:
                return
            UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
            CHROMA_DIR.mkdir(parents=True, exist_ok=True)

            if self.embeddings is None:
                try:
                    self._load_embeddings()
                    self.embeddings_ready = True
                except Exception as exc:
                    self.error = f"임베딩 모델 로딩 실패: {exc}"
                    raise
            if self.vectorstore is None:
                self._load_vectorstore()

            if self.llm is None:
                try:
                    self._load_llm()
                    self.llm_ready = True
                    self.error = None
                except Exception as exc:
                    self.llm_ready = False
                    detail = str(exc)
                    if "llama.dll" in detail or "shared library" in detail.lower():
                        self.error = (
                            "llama-cpp-python이 필요한 DLL을 찾지 못해 GGUF LLM을 불러오지 못했습니다. "
                            "CPU용 휠을 설치하세요: pip install llama-cpp-python==0.3.33 "
                            "--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu"
                        )
                    else:
                        self.error = f"LLM 로딩 실패: {detail}"
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

    def list_files(self) -> List[Dict[str, Any]]:
        if self.vectorstore is None:
            return []
        try:
            data = self.vectorstore.get(include=["metadatas"])
        except Exception:
            return []
        counts: Dict[str, int] = {}
        for meta in data.get("metadatas") or []:
            name = (meta or {}).get("filename") or Path((meta or {}).get("source", "")).name
            if name:
                counts[name] = counts.get(name, 0) + 1
        return [{"name": name, "chunks": counts[name]} for name in sorted(counts)]

    def ingest_pdfs(self, file_paths: List[Path]) -> Dict[str, Any]:
        if self.vectorstore is None:
            raise RuntimeError("벡터 DB가 아직 준비되지 않았습니다.")

        added_files: List[str] = []
        skipped: List[Dict[str, str]] = []
        total_chunks = 0

        for path in file_paths:
            filename = path.name
            try:
                loader = PyPDFLoader(str(path))
                pages = loader.load()
                if not pages or not any((p.page_content or "").strip() for p in pages):
                    skipped.append({"name": filename, "reason": "텍스트를 추출할 수 없는 PDF입니다."})
                    continue

                for page in pages:
                    page.metadata["filename"] = filename
                    page.metadata["source"] = filename
                    page.metadata["page_number"] = _page_number(page.metadata)

                chunks = self.splitter.split_documents(pages)
                chunks = [c for c in chunks if (c.page_content or "").strip()]
                if not chunks:
                    skipped.append({"name": filename, "reason": "분할할 텍스트가 없습니다."})
                    continue

                self._replace_file_chunks(filename, chunks)
                added_files.append(filename)
                total_chunks += len(chunks)
            except Exception as exc:
                skipped.append({"name": filename, "reason": str(exc)})

        return {
            "added": added_files,
            "skipped": skipped,
            "chunk_count": total_chunks,
            "files": self.list_files(),
        }

    def _replace_file_chunks(self, filename: str, chunks: List[Document]) -> None:
        assert self.vectorstore is not None
        try:
            existing = self.vectorstore.get(where={"filename": filename})
            ids = existing.get("ids") or []
            if ids:
                self.vectorstore.delete(ids=ids)
        except Exception:
            pass
        self.vectorstore.add_documents(chunks)

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
        results = self.vectorstore.similarity_search_with_score(query, k=RETRIEVE_K)
        relevant = [(doc, float(score)) for doc, score in results if float(score) <= DISTANCE_THRESHOLD]
        return relevant

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
