import json
import logging
import re
import shutil
import threading
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request, stream_with_context
from werkzeug.exceptions import RequestEntityTooLarge

from config import ALLOWED_EXTENSIONS, MAX_CONTENT_LENGTH, NO_INFO_MESSAGE, TMP_UPLOAD_DIR, UPLOAD_DIR
from jobs import job_store
from rag_engine import get_engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
app.secret_key = "local-rag-chatbot"


def _is_pdf(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def _safe_pdf_name(filename: str) -> str:
    name = Path(filename).name.replace("\x00", "")
    name = re.sub(r'[\\/:*?"<>|]', "_", name).strip()
    if not name.lower().endswith(".pdf") or name.lower() in {"pdf", ".pdf"}:
        return ""
    return name


def _preload_models() -> None:
    engine = get_engine()
    try:
        engine.initialize()
    except Exception:
        pass


def _public_status():
    engine = get_engine()
    active = job_store.active_job()
    return {
        "ready": engine.ready,
        "llm_ready": engine.llm_ready,
        "embeddings_ready": engine.embeddings_ready,
        "initializing": engine.initializing,
        "error": engine.error,
        "embedding_error": engine.embedding_error,
        "llm_error": engine.llm_error,
        "files": engine.list_files() if engine.embeddings_ready else [],
        "upload_job": active.to_dict() if active else None,
    }


def _cleanup_job_dir(job_dir: Path) -> None:
    shutil.rmtree(job_dir, ignore_errors=True)


def _run_upload_job(job_id: str, items: list, job_dir: Path) -> None:
    engine = get_engine()
    try:
        if not engine.embeddings_ready:
            engine.initialize()
        if not engine.embeddings_ready or engine.vectorstore is None:
            raise RuntimeError(engine.embedding_error or "임베딩 모델이 아직 준비되지 않았습니다.")

        def progress(stage: str) -> None:
            job_store.update(job_id, status="running", stage=stage)

        job_store.update(job_id, status="running", stage="extract")
        result = engine.ingest_pdfs(items, progress=progress)
        status = "done"
        error = None
        if not result.get("added") and result.get("skipped"):
            status = "failed"
            error = "파일을 처리하지 못했습니다. " + " / ".join(
                f"{item['name']}: {item['reason']}" for item in result["skipped"]
            )
        job_store.update(
            job_id,
            status=status,
            stage=None,
            error=error,
            added=result.get("added") or [],
            skipped=result.get("skipped") or [],
            files=result.get("files") or [],
            chunk_count=result.get("chunk_count") or 0,
        )
    except Exception as exc:
        logger.exception("upload job failed")
        job_store.update(job_id, status="failed", stage=None, error=str(exc))
    finally:
        _cleanup_job_dir(job_dir)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def status():
    return jsonify(_public_status())


@app.route("/api/retry-models", methods=["POST"])
def retry_models():
    engine = get_engine()
    if engine.ready:
        return jsonify({"ok": True, **_public_status()})
    if engine.initializing:
        return jsonify({"ok": True, "message": "이미 모델을 준비하고 있습니다.", **_public_status()})
    try:
        engine.initialize()
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc), **_public_status()}), 503
    payload = _public_status()
    payload["ok"] = bool(engine.ready or engine.embeddings_ready)
    return jsonify(payload)


@app.errorhandler(RequestEntityTooLarge)
def too_large(_exc):
    return jsonify({"ok": False, "error": "파일이 너무 큽니다. 50MB 이하 PDF만 업로드하세요."}), 413


@app.route("/api/upload", methods=["POST"])
def upload():
    engine = get_engine()
    files = request.files.getlist("files") or list(request.files.values())
    logger.info("upload received: %s", [f.filename for f in files])
    if not files:
        return jsonify({"ok": False, "error": "PDF 파일이 전달되지 않았습니다. 버튼을 눌러 다시 선택하세요."}), 400

    rejected = []
    items = []
    try:
        job = job_store.create()
    except RuntimeError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 409
    job_dir = TMP_UPLOAD_DIR / job.id
    job_dir.mkdir(parents=True, exist_ok=True)

    try:
        for file in files:
            original = file.filename or ""
            if not original:
                continue
            if not _is_pdf(original):
                rejected.append({"name": original, "reason": "PDF 파일만 업로드할 수 있습니다."})
                continue
            filename = _safe_pdf_name(original)
            if not filename:
                rejected.append({"name": original, "reason": "PDF 파일만 업로드할 수 있습니다."})
                continue
            temp_path = job_dir / filename
            file.save(temp_path)
            header = temp_path.read_bytes()[:1024]
            if b"%PDF" not in header:
                temp_path.unlink(missing_ok=True)
                rejected.append({"name": original, "reason": "올바른 PDF 파일이 아닙니다."})
                continue
            items.append((temp_path, filename))
    except Exception:
        job_store.update(job.id, status="failed", error="업로드 파일을 저장하지 못했습니다.")
        _cleanup_job_dir(job_dir)
        raise

    job_store.update(job.id, rejected=rejected)
    if not items:
        job_store.update(
            job.id,
            status="failed",
            error="허용되지 않은 파일입니다." if rejected else "PDF 파일이 전달되지 않았습니다. 버튼을 눌러 다시 선택하세요.",
            skipped=rejected,
        )
        _cleanup_job_dir(job_dir)
        return jsonify({"ok": False, "job_id": job.id, **job_store.get(job.id).to_dict()}), 400

    threading.Thread(target=_run_upload_job, args=(job.id, items, job_dir), daemon=True).start()
    return jsonify({"ok": True, "job_id": job.id, **job.to_dict()})


@app.route("/api/jobs/<job_id>")
def job_status(job_id: str):
    job = job_store.get(job_id)
    if not job:
        return jsonify({
            "ok": False,
            "missing": True,
            "error": "작업 상태를 찾을 수 없습니다. 서버가 재시작되었을 수 있습니다.",
        }), 404
    return jsonify({"ok": True, **job.to_dict()})


@app.route("/api/files")
def files():
    engine = get_engine()
    return jsonify({"files": engine.list_files()})


@app.route("/api/reset", methods=["POST"])
def reset_chat():
    engine = get_engine()
    engine.reset_memory()
    return jsonify({"ok": True})


@app.route("/api/chat", methods=["POST"])
def chat():
    engine = get_engine()
    if not engine.llm_ready:
        try:
            engine.initialize()
        except Exception as exc:
            return jsonify({"ok": False, "error": f"모델 로딩 실패: {exc}"}), 503
    if not engine.llm_ready:
        return jsonify(
            {"ok": False, "error": engine.llm_error or engine.error or "LLM이 아직 준비되지 않았습니다."}
        ), 503

    payload = request.get_json(silent=True) or {}
    question = (payload.get("message") or "").strip()
    if not question:
        return jsonify({"ok": False, "error": "질문을 입력하세요."}), 400

    def generate():
        try:
            for event in engine.answer_stream(question):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as exc:
            fail = {
                "type": "error",
                "error": str(exc),
                "text": NO_INFO_MESSAGE,
            }
            yield f"data: {json.dumps(fail, ensure_ascii=False)}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    threading.Thread(target=_preload_models, daemon=True).start()
    app.run(host="127.0.0.1", port=5050, debug=False, threaded=True)
