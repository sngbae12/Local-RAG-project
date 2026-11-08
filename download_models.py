"""Download the specific local models used by this app, or check their structure.

Only --check is fully offline. These checks do not load either model and do not
establish file integrity or CPU compatibility. Hugging Face handles downloads.
"""

import argparse
import json
from pathlib import Path
import sys

from config import BASE_DIR, LLM_PATH

LLM_REPO = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct-GGUF"
LLM_FILENAME = LLM_PATH.name
EMBEDDING_REPO = "BAAI/bge-m3"
EMBEDDING_FILES = (
    "config.json",
    "modules.json",
    "config_sentence_transformers.json",
    "sentence_bert_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "sentencepiece.bpe.model",
    "1_Pooling/config.json",
    "pytorch_model.bin",
)


def model_files(base_dir: Path):
    yield LLM_REPO, base_dir / "models" / "exaone_2.4b", LLM_FILENAME
    for filename in EMBEDDING_FILES:
        yield EMBEDDING_REPO, base_dir / "models" / "bge-m3", filename


def check_models(base_dir: Path = BASE_DIR) -> list[str]:
    """Return actionable errors without importing ML packages or using a network."""
    errors = []
    configs = {}
    for _repo, folder, filename in model_files(base_dir):
        path = folder / filename
        label = str(path.relative_to(base_dir))
        try:
            if not path.is_file() or path.stat().st_size == 0:
                errors.append(f"Missing or empty: {label}")
                continue
            if filename.endswith(".json"):
                configs[filename] = json.loads(path.read_text(encoding="utf-8"))
            elif filename.endswith((".gguf", ".bin")):
                with path.open("rb") as stream:
                    header = stream.read(128)
                if filename.endswith(".gguf") and not header.startswith(b"GGUF"):
                    errors.append(f"Invalid GGUF header: {label}")
                if path.stat().st_size < 1024 * 1024 or header.startswith(b"version https://git-lfs"):
                    errors.append(f"Weight file is too small or is a Git LFS pointer: {label}")
        except (OSError, ValueError) as exc:
            errors.append(f"Cannot read {label}: {exc}")

    config = configs.get("config.json")
    if "config.json" in configs and (not isinstance(config, dict) or config.get("model_type") != "xlm-roberta"):
        errors.append("BGE-M3 config.json must contain model_type: xlm-roberta")
    modules = configs.get("modules.json")
    if "modules.json" in configs:
        if not isinstance(modules, list) or not any(
            isinstance(item, dict)
            and item.get("type") == "sentence_transformers.models.Transformer"
            and item.get("path") == ""
            for item in modules
        ) or not any(
            isinstance(item, dict)
            and item.get("type") == "sentence_transformers.models.Pooling"
            and item.get("path") == "1_Pooling"
            for item in modules
        ):
            errors.append("BGE-M3 modules.json is not the expected Transformer/Pooling configuration")
    return errors


def download_models(base_dir: Path = BASE_DIR, *, repair_config=False, force=False):
    # Keep --check usable before dependencies are installed.
    from huggingface_hub import hf_hub_download

    for repo, folder, filename in model_files(base_dir):
        print(f"Downloading/checking {repo}: {filename}", flush=True)
        hf_hub_download(
            repo_id=repo,
            filename=filename,
            local_dir=str(folder),
            force_download=force or (repair_config and filename.endswith(".json")),
        )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--check", action="store_true", help="Check local files without network or model loading")
    group.add_argument("--repair-config", action="store_true", help="Re-download JSON files; reuse cached weights")
    group.add_argument("--force", action="store_true", help="Re-download every required file, including large weights")
    args = parser.parse_args(argv)
    if not args.check:
        print("Model sources and terms: models/README.md", flush=True)
        try:
            download_models(repair_config=args.repair_config, force=args.force)
        except KeyboardInterrupt:
            print("\nDownload interrupted. Run the same command to retry.", file=sys.stderr)
            return 130
        except Exception as exc:
            print(f"Download failed: {exc}", file=sys.stderr)
            print("See docs/troubleshooting.md; no PDFs or vector data were removed.", file=sys.stderr)
            return 1
    errors = check_models()
    if errors:
        for error in errors:
            print(f"[ERROR] {error}", file=sys.stderr)
        print("Run download_models.py, or use --repair-config for incorrect JSON files.", file=sys.stderr)
        return 1
    print("Basic model file checks passed. Actual model loading is checked by the app.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
