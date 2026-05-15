import unittest

from scripts.hst_recovery_analysis import analyze


class RecoveryAnalysisTest(unittest.TestCase):
    def test_recovery_gap_and_baseline_match(self):
        rows = [
            {"run_name": "tst", "method": "vanilla_tst", "step": 2, "phase": "superposition", "loss_eval_ntp": 5.0},
            {"run_name": "tst", "method": "vanilla_tst", "step": 4, "phase": "superposition", "loss_eval_ntp": 4.0},
            {"run_name": "tst", "method": "vanilla_tst", "step": 6, "phase": "recovery", "loss_eval_ntp": 4.4},
            {"run_name": "tst", "method": "vanilla_tst", "step": 8, "phase": "recovery", "loss_eval_ntp": 3.0},
        ]
        baseline = [
            {"step": 6, "phase": "ntp", "loss_eval_ntp": 3.5},
            {"step": 8, "phase": "ntp", "loss_eval_ntp": 3.2},
        ]
        result = analyze(rows, baseline)
        self.assertEqual(result["recovery_start_step"], 6)
        self.assertEqual(result["last_ntp_eval_before_recovery"], 4.0)
        self.assertEqual(result["first_ntp_eval_after_recovery"], 4.4)
        self.assertAlmostEqual(result["recovery_gap"], 0.4)
        self.assertEqual(result["steps_to_beat_baseline_same_step"], 2)

    def test_missing_baseline_is_null(self):
        rows = [
            {"run_name": "tst", "method": "vanilla_tst", "step": 1, "phase": "superposition", "loss_eval_ntp": 5.0},
            {"run_name": "tst", "method": "vanilla_tst", "step": 2, "phase": "recovery", "loss_eval_ntp": 4.5},
        ]
        result = analyze(rows)
        self.assertIsNone(result["steps_to_beat_baseline_same_step"])


if __name__ == "__main__":
    unittest.main()
