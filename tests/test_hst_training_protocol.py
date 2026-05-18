import unittest

try:
    import torch

    from pathlib import Path

    from trainer.train_hst_pretrain import TrainConfig, batch_for_phase, baseline_seq_len, checkpoint_step, dense_eval_anchor_step, phase_for_step, recovery_start_step, should_run_eval, token_counts, train_raw_seq_len, tst_ratio, validate_config
except Exception:
    torch = None


@unittest.skipIf(torch is None, "torch is not installed")
class TrainingProtocolTest(unittest.TestCase):
    def test_equal_flops_raw_len_expands_only_for_tst(self):
        cfg = TrainConfig(method="vanilla_tst", baseline_seq_len=128, max_seq_len=128, superpose_size=4, paper_equal_flops=1)
        self.assertEqual(baseline_seq_len(cfg), 128)
        self.assertEqual(train_raw_seq_len(cfg), 512)
        ntp = TrainConfig(method="ntp_baseline", baseline_seq_len=128, max_seq_len=128, superpose_size=4, paper_equal_flops=1)
        self.assertEqual(train_raw_seq_len(ntp), 128)

    def test_batch_for_phase_crops_recovery_to_baseline(self):
        cfg = TrainConfig(method="vanilla_tst", baseline_seq_len=8, max_seq_len=8, superpose_size=2, paper_equal_flops=1)
        batch = torch.arange(32).view(2, 16)
        self.assertEqual(batch_for_phase(batch, cfg, "superposition").shape, (2, 16))
        self.assertEqual(batch_for_phase(batch, cfg, "recovery").shape, (2, 8))

    def test_token_counts_distinguish_raw_and_latent(self):
        cfg = TrainConfig(method="vanilla_tst", baseline_seq_len=8, max_seq_len=8, superpose_size=2, paper_equal_flops=1)
        batch = torch.zeros(2, 16, dtype=torch.long)
        raw, latent, effective = token_counts(batch, cfg, "superposition")
        self.assertEqual(raw, 32)
        self.assertEqual(latent, 14)
        self.assertEqual(effective, 32)

    def test_recovery_ratio_maps_to_paper_tst_ratio(self):
        cfg = TrainConfig(method="vanilla_tst", max_steps=20000, recovery_ratio=0.7)
        self.assertAlmostEqual(tst_ratio(cfg), 0.3)
        self.assertEqual(recovery_start_step(cfg), 6000)
        self.assertEqual(phase_for_step(cfg, 5999), "superposition")
        self.assertEqual(phase_for_step(cfg, 6000), "recovery")

    def test_ntp_baseline_has_zero_tst_ratio(self):
        cfg = TrainConfig(method="ntp_baseline", max_steps=20000, recovery_ratio=0.7)
        self.assertEqual(tst_ratio(cfg), 0.0)
        self.assertEqual(recovery_start_step(cfg), 20000)

    def test_checkpoint_step_sorts_numerically(self):
        paths = [Path("step_5000.pt"), Path("step_10000.pt"), Path("step_50.pt")]
        self.assertEqual([p.name for p in sorted(paths, key=checkpoint_step)], ["step_50.pt", "step_5000.pt", "step_10000.pt"])

    def test_phase_override_for_paper_recovery(self):
        cfg = TrainConfig(method="ntp_baseline", max_steps=14000, global_step_offset=6000, phase_override="recovery")
        self.assertEqual(phase_for_step(cfg, 0), "recovery")
        batch = torch.arange(32).view(2, 16)
        self.assertEqual(batch_for_phase(batch, cfg, "recovery").shape, (2, 16))

    def test_recovery_phase_uses_baseline_raw_len(self):
        cfg = TrainConfig(method="vanilla_tst", baseline_seq_len=8, max_seq_len=8, superpose_size=4, paper_equal_flops=1, phase_override="recovery")
        self.assertEqual(train_raw_seq_len(cfg), 8)

    def test_dense_eval_anchor_defaults_to_recovery_start(self):
        cfg = TrainConfig(method="vanilla_tst", max_steps=20000, recovery_ratio=0.7)
        self.assertEqual(dense_eval_anchor_step(cfg), 6001)

    def test_dense_eval_anchor_for_two_phase_recovery(self):
        cfg = TrainConfig(method="ntp_baseline", phase_override="recovery", global_step_offset=6000, max_steps=14000)
        self.assertEqual(dense_eval_anchor_step(cfg), 6001)

    def test_should_run_dense_eval_near_anchor(self):
        cfg = TrainConfig(method="vanilla_tst", max_steps=20000, recovery_ratio=0.7, eval_interval=500, dense_eval_interval=50, dense_eval_window=100)
        self.assertTrue(should_run_eval(cfg, step=5950, global_step=5951))
        self.assertTrue(should_run_eval(cfg, step=6000, global_step=6001))
        self.assertFalse(should_run_eval(cfg, step=5898, global_step=5899))

    def test_sparse_anchor_residual_config_validation(self):
        cfg = TrainConfig(method="sparse_anchor_residual_tst", superpose_size=4, anchor_slot_idx=1, residual_codebook_size=64)
        validate_config(cfg)


if __name__ == "__main__":
    unittest.main()
