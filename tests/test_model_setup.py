"""Offline regression tests for distributing the required model files.

Small sparse fixtures are structural test doubles, not executable models.
"""

import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import download_models as setup


def official_config(filename):
    if filename == "config.json":
        return {"model_type": "xlm-roberta"}
    if filename == "modules.json":
        return [
            {"type": "sentence_transformers.models.Transformer", "path": ""},
            {"type": "sentence_transformers.models.Pooling", "path": "1_Pooling"},
        ]
    return {}


def write_fixture(path, filename):
    path.parent.mkdir(parents=True, exist_ok=True)
    if filename.endswith(".json"):
        path.write_text(json.dumps(official_config(filename)), encoding="utf-8")
    elif filename.endswith((".gguf", ".bin")):
        with path.open("wb") as stream:
            stream.write(b"GGUF" if filename.endswith(".gguf") else b"fixture")
            stream.truncate(1024 * 1024)
    else:
        path.write_bytes(b"fixture tokenizer")


class ModelSetupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="local rag ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for _repo, folder, filename in setup.model_files(self.root):
            write_fixture(folder / filename, filename)
        for folder in ("uploads", "chroma_db"):
            target = self.root / folder / "keep.dat"
            target.parent.mkdir()
            target.write_bytes(b"user data")

    def assert_user_data_preserved(self):
        for folder in ("uploads", "chroma_db"):
            self.assertEqual((self.root / folder / "keep.dat").read_bytes(), b"user data")

    def test_missing_sentence_transformer_files_are_reported(self):
        bge = self.root / "models" / "bge-m3"
        (bge / "modules.json").unlink()
        (bge / "1_Pooling" / "config.json").unlink()
        errors = setup.check_models(self.root)
        self.assertTrue(any("modules.json" in error for error in errors))
        self.assertTrue(any("1_Pooling" in error for error in errors))
        self.assert_user_data_preserved()

    def test_invalid_model_type_and_null_configuration_are_rejected(self):
        bge = self.root / "models" / "bge-m3"
        for content in ("{}", "null", "[]", '{"model_type": "wrong"}'):
            with self.subTest(content=content):
                (bge / "config.json").write_text(content, encoding="utf-8")
                self.assertTrue(any("model_type" in error for error in setup.check_models(self.root)))
        (bge / "modules.json").write_text("null", encoding="utf-8")
        self.assertTrue(any("modules.json" in error for error in setup.check_models(self.root)))

    def test_corrupt_json_and_pointer_weights_are_rejected(self):
        bge = self.root / "models" / "bge-m3"
        (bge / "tokenizer.json").write_text("<html>error</html>", encoding="utf-8")
        (bge / "pytorch_model.bin").write_bytes(b"version https://git-lfs.github.com/spec/v1\n")
        llm = self.root / "models" / "exaone_2.4b" / setup.LLM_FILENAME
        with llm.open("r+b") as stream:
            stream.write(b"HTML")
        errors = setup.check_models(self.root)
        self.assertTrue(any("tokenizer.json" in error for error in errors))
        self.assertTrue(any("pointer" in error for error in errors))
        self.assertTrue(any("GGUF header" in error for error in errors))
        self.assert_user_data_preserved()

    def test_repair_config_restores_json_without_forcing_large_weights(self):
        bge = self.root / "models" / "bge-m3"
        (bge / "config.json").write_text("{}", encoding="utf-8")
        (bge / "modules.json").unlink()
        calls = []

        def fake_download(**kwargs):
            calls.append(kwargs)
            path = Path(kwargs["local_dir"]) / kwargs["filename"]
            if kwargs["force_download"] or not path.exists():
                write_fixture(path, kwargs["filename"])
            return str(path)

        with patch.dict(sys.modules, {"huggingface_hub": SimpleNamespace(hf_hub_download=fake_download)}):
            with contextlib.redirect_stdout(io.StringIO()):
                setup.download_models(self.root, repair_config=True)
        self.assertEqual(setup.check_models(self.root), [])
        self.assertTrue(all(not c["force_download"] for c in calls if c["filename"].endswith((".bin", ".gguf"))))
        self.assertTrue(all(c["force_download"] for c in calls if c["filename"].endswith(".json")))
        self.assertFalse(any("onnx" in c["filename"] for c in calls))
        self.assert_user_data_preserved()

    def test_download_failure_does_not_report_success_or_remove_user_data(self):
        output = io.StringIO()
        with patch.object(setup, "download_models", side_effect=OSError("network unavailable")):
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                result = setup.main([])
        self.assertEqual(result, 1)
        self.assertIn("Download failed", output.getvalue())
        self.assertNotIn("checks passed", output.getvalue())
        self.assert_user_data_preserved()

    def test_offline_check_from_other_directory_needs_no_installed_packages(self):
        # -S removes site-packages; the check must still explain missing files.
        result = subprocess.run(
            [sys.executable, "-S", str(Path(setup.__file__).resolve()), "--check"],
            cwd=self.root,
            capture_output=True,
            text=True,
        )
        self.assertIn(result.returncode, (0, 1))
        self.assertNotIn("ModuleNotFoundError", result.stderr)
        self.assertTrue("Basic model file checks passed" in result.stdout or "Missing or empty" in result.stderr)

    @unittest.skipUnless(sys.platform == "win32", "Requires Windows cmd.exe")
    def test_launcher_explains_missing_environment_on_windows(self):
        script = self.root / "start.bat"
        script.write_bytes((Path(setup.__file__).parent / "start.bat").read_bytes())
        result = subprocess.run(
            ["cmd.exe", "/d", "/c", str(script)], input="\n", cwd=self.root,
            capture_output=True, text=True, timeout=15,
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("Run setup.bat first", result.stdout)


if __name__ == "__main__":
    unittest.main()
