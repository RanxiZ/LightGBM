import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location("npu_install", Path(__file__).parents[1] / "install.py")
INSTALL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(INSTALL)


class InstallationTests(unittest.TestCase):
    def test_frozen_qlib_extracts_without_network_and_keeps_user_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "qlib"
            INSTALL.ensure_qlib(root)
            marker = json.loads((root / ".npu-qlib-source.json").read_text())
            self.assertEqual(marker["source_commit"], INSTALL.QLIB_COMMIT)
            self.assertTrue((root / "examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml").is_file())
            extra = root / "user_model.py"
            extra.write_text("preserve this")
            INSTALL.ensure_qlib(root)
            self.assertEqual(extra.read_text(), "preserve this")

    def test_corrupted_qlib_archive_is_rejected_before_extraction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake = root / "repo/npu"
            (fake / "reference").mkdir(parents=True)
            (fake / "reference/qlib_source.json").write_text(json.dumps({"archive_sha256": "0" * 64}))
            (fake / "qlib_source.tar.gz").write_bytes(b"corrupted archive")
            original = INSTALL.REPO
            try:
                INSTALL.REPO = fake.parent
                with self.assertRaisesRegex(RuntimeError, "SHA256 mismatch"):
                    INSTALL.ensure_qlib(root / "qlib")
                self.assertFalse((root / "qlib").exists())
            finally:
                INSTALL.REPO = original

    def test_qlib_commit_guard_preserves_unrelated_local_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def git(*args):
                return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.DEVNULL, text=True).strip()
            git("init")
            git("config", "user.name", "Installer Test")
            git("config", "user.email", "installer@example.invalid")
            file = root / "user_model.py"
            file.write_text("original")
            git("add", ".")
            git("commit", "-m", "fixture")
            head = git("rev-parse", "HEAD")
            file.write_text("user modification")
            original = INSTALL.QLIB_COMMIT
            try:
                INSTALL.QLIB_COMMIT = "0" * 40
                with self.assertRaises(RuntimeError):
                    INSTALL.ensure_qlib(root)
                INSTALL.QLIB_COMMIT = head
                INSTALL.ensure_qlib(root)
                self.assertEqual(git("rev-parse", "HEAD"), head)
                self.assertEqual(file.read_text(), "user modification")
            finally:
                INSTALL.QLIB_COMMIT = original

    def test_rollback_restores_contents_modes_and_removes_new_entry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / "qrun"
            original.write_text("original\n")
            original.chmod(0o740)
            untouched = root / "other_model.py"
            untouched.write_text("preserve user code\n")
            added = root / "new_qrun"
            state = INSTALL.Installation(root / "state")
            state.write(original, "replacement\n")
            state.write(original, "replacement twice\n")
            state.write(added, "new\n")
            state.rollback()
            self.assertEqual(original.read_text(), "original\n")
            self.assertEqual(original.stat().st_mode & 0o777, 0o740)
            self.assertFalse(added.exists())
            self.assertEqual(untouched.read_text(), "preserve user code\n")

    def test_rollback_preserves_original_symlink_without_modifying_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target"
            target.write_text("original target")
            entry = root / "qrun"
            entry.symlink_to("target")
            state = INSTALL.Installation(root / "state")
            state.write(entry, "wrapper")
            self.assertFalse(entry.is_symlink())
            self.assertEqual(target.read_text(), "original target")
            state.rollback()
            self.assertTrue(entry.is_symlink())
            self.assertEqual(entry.readlink(), Path("target"))
            self.assertEqual(target.read_text(), "original target")

    def test_wrapper_quotes_paths_and_calls_original_cli(self):
        script = INSTALL.qrun_script("/python with space/bin/python", Path("/cann with space"))
        result = subprocess.run(["bash", "-n"], input=script, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("from qlib.cli.run import run", script)
        self.assertIn("unset PYTHONPATH PYTHONHOME LD_PRELOAD", script)


if __name__ == "__main__":
    unittest.main()
