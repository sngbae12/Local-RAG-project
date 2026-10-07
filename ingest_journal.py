import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import SWAP_DIR

logger = logging.getLogger(__name__)


def _safe_name(filename: str) -> str:
    return Path(filename).name.replace("\\", "_").replace("/", "_")


def record_path(filename: str) -> Path:
    return SWAP_DIR / f"{_safe_name(filename)}.json"


def write_record(filename: str, record: Dict[str, Any]) -> None:
    SWAP_DIR.mkdir(parents=True, exist_ok=True)
    path = record_path(filename)
    tmp = path.with_suffix(".json.tmp")
    payload = dict(record)
    payload["filename"] = Path(filename).name
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def read_record(filename: str) -> Optional[Dict[str, Any]]:
    path = record_path(filename)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("swap record unreadable: %s", path.name)
        return None


def clear_record(filename: str) -> None:
    path = record_path(filename)
    path.unlink(missing_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.unlink(missing_ok=True)


def list_records() -> List[Dict[str, Any]]:
    if not SWAP_DIR.exists():
        return []
    records = []
    for path in SWAP_DIR.glob("*.json"):
        try:
            records.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            logger.warning("swap record skipped: %s", path.name)
    return records
