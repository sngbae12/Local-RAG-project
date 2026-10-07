import json
import logging
import re
import threading
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request, stream_with_context
from werkzeug.exceptions import RequestEntityTooLarge

from config import ALLOWED_EXTENSIONS, MAX_CONTENT_LENGTH, NO_INFO_MESSAGE, UPLOAD_DIR
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


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def status():
    engine = get_engine()
    return jsonify(
        {
            "ready": engine.ready,
            "llm_ready": engine.llm_ready,
            "embeddings_ready": engine.embeddings_ready,
            "error": engine.error,
            "files": engine.list_files() if engine.embeddings_ready else [],
        }
    )


@app.errorhandler(RequestEntityTooLarge)
def too_large(_exc):
    return jsonify({"ok": False, "error": "파일이 너무 큽니다. 50MB 이하 PDF만 업로드하세요."}), 413


@app.route("/api/upload", methods=["POST"])
def upload():
    engine = get_engine()
    if not engine.embeddings_ready:
        try:
            engine.initialize()
        except Exception as exc:
            return jsonify({"ok": False, "error": f"모델 로딩 실패: {exc}"}), 503
    if not engine.embeddings_ready or engine.vectorstore is None:
        return jsonify({"ok": False, "error": "임베딩 모델이 아직 준비되지 않았습니다. 잠시 후 다시 시도하세요."}), 503

    files = request.files.getlist("files") or list(request.files.values())
    logger.info("upload received: %s", [f.filename for f in files])
    if not files:
        return jsonify({"ok": False, "error": "PDF 파일이 전달되지 않았습니다. 버튼을 눌러 다시 선택하세요."}), 400

    saved_paths = []
    rejected = []
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

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
        dest = UPLOAD_DIR / filename
        file.save(dest)
        header = dest.read_bytes()[:1024]
        if b"%PDF" not in header:
            dest.unlink(missing_ok=True)
            rejected.append({"name": original, "reason": "올바른 PDF 파일이 아닙니다."})
            continue
        saved_paths.append(dest)

    if not saved_paths and rejected:
        return jsonify({"ok": False, "error": "허용되지 않은 파일입니다.", "skipped": rejected}), 400
    if not saved_paths:
        return jsonify({"ok": False, "error": "PDF 파일이 전달되지 않았습니다. 버튼을 눌러 다시 선택하세요."}), 400

    try:
        result = engine.ingest_pdfs(saved_paths)
    except Exception as exc:
        logger.exception("ingest failed")
        return jsonify({"ok": False, "error": f"문서 처리 중 오류: {exc}"}), 500
    result["ok"] = True
    result["rejected"] = rejected
    if not result.get("added") and (result.get("skipped") or rejected):
        reasons = result.get("skipped") or rejected
        result["ok"] = False
        result["error"] = "파일을 저장했지만 텍스트를 읽지 못했습니다. " + " / ".join(
            f"{item['name']}: {item['reason']}" for item in reasons
        )
        return jsonify(result), 400
    return jsonify(result)


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
            {"ok": False, "error": engine.error or "LLM이 아직 준비되지 않았습니다."}
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
