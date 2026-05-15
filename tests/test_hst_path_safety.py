import tempfile
import unittest
from pathlib import Path

from utils.hst_path_safety import PathSafetyError, ensure_run_output_dir, ensure_within_project


class PathSafetyTest(unittest.TestCase):
    def test_allows_project_relative_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            self.assertEqual(ensure_within_project("hst_tmp/x.txt", root), root / "hst_tmp/x.txt")

    def test_rejects_escape_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(PathSafetyError):
                ensure_within_project("../outside.txt", Path(tmp).resolve())

    def test_run_output_must_be_under_hst_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            self.assertEqual(ensure_run_output_dir("hst_runs/run_a", root), root / "hst_runs/run_a")
            with self.assertRaises(PathSafetyError):
                ensure_run_output_dir("hst_outputs/run_a", root)


if __name__ == "__main__":
    unittest.main()
