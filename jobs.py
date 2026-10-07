import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


STAGE_LABELS = {
    "extract": "텍스트 추출",
    "chunk": "청킹",
    "embed": "임베딩",
    "store": "벡터 저장",
}


@dataclass
class UploadJob:
    id: str
    status: str = "queued"
    stage: Optional[str] = None
    error: Optional[str] = None
    added: List[str] = field(default_factory=list)
    skipped: List[Dict[str, str]] = field(default_factory=list)
    warnings: List[Dict[str, str]] = field(default_factory=list)
    rejected: List[Dict[str, str]] = field(default_factory=list)
    files: List[Dict[str, Any]] = field(default_factory=list)
    chunk_count: int = 0
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "stage": self.stage,
            "stage_label": STAGE_LABELS.get(self.stage) if self.stage else None,
            "error": self.error,
            "added": self.added,
            "skipped": self.skipped,
            "warnings": self.warnings,
            "rejected": self.rejected,
            "files": self.files,
            "chunk_count": self.chunk_count,
        }


class JobStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: Dict[str, UploadJob] = {}
        self._active_id: Optional[str] = None

    def has_active(self) -> bool:
        with self._lock:
            return self._active_id is not None

    def active_job(self) -> Optional[UploadJob]:
        with self._lock:
            if not self._active_id:
                return None
            return self._jobs.get(self._active_id)

    def create(self, rejected: Optional[List[Dict[str, str]]] = None) -> UploadJob:
        with self._lock:
            if self._active_id:
                raise RuntimeError("이미 문서 처리를 진행 중입니다.")
            job = UploadJob(id=uuid.uuid4().hex, rejected=list(rejected or []))
            self._jobs[job.id] = job
            self._active_id = job.id
            return job

    def get(self, job_id: str) -> Optional[UploadJob]:
        with self._lock:
            return self._jobs.get(job_id)

    def update(self, job_id: str, **fields: Any) -> Optional[UploadJob]:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            for key, value in fields.items():
                setattr(job, key, value)
            if job.status in {"done", "failed"} and self._active_id == job_id:
                self._active_id = None
            return job

    def fail(self, job_id: str, error: str, **fields: Any) -> Optional[UploadJob]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.status = "failed"
                job.stage = None
                job.error = error
                for key, value in fields.items():
                    setattr(job, key, value)
            if self._active_id == job_id:
                self._active_id = None
            return job


job_store = JobStore()
