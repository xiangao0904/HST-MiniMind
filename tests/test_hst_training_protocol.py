import unittest

try:
    import torch
    import yaml

    from pathlib import Path

    from trainer.train_hst_pretrain import AdaptiveRecoveryState, CalibrationState, TrainConfig, adaptive_recovery_deadline, batch_for_phase, baseline_seq_len, calibration_weight_for_loss, checkpoint_step, dense_eval_anchor_step, method_to_mode, maybe_update_adaptive_recovery, model_seq_len, phase_for_step, recovery_start_step, should_run_eval, token_counts, train_raw_seq_len, tst_ratio, validate_config
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

    def test_paper_residual_structured_uses_residual_composer_and_repeated_ce(self):
        cfg = TrainConfig(
            method="paper_residual_structured_tst",
            baseline_seq_len=128,
            max_seq_len=128,
            superpose_size=4,
            paper_equal_flops=1,
            loss_mode="repeated_ce",
        )
        validate_config(cfg)
        self.assertEqual(method_to_mode(cfg), "residual_structured")
        self.assertEqual(train_raw_seq_len(cfg), 512)

    def test_paper_residual_structured_full_config_matches_vanilla_protocol(self):
        vanilla = yaml.safe_load(Path("configs/hst/paper_vanilla_tst_s4_r03_full_120k.yaml").read_text(encoding="utf-8"))
        residual = yaml.safe_load(Path("configs/hst/paper_residual_structured_s4_r03_full_120k.yaml").read_text(encoding="utf-8"))
        allowed_diffs = {
            "method",
            "run_name",
            "output_dir",
            "slot_gate_type",
            "order_alpha",
            "hier_alpha",
            "block_mode",
            "chunks_per_block",
        }
        diffs = {key for key in set(vanilla) | set(residual) if vanilla.get(key) != residual.get(key)}
        self.assertEqual(diffs, allowed_diffs)
        self.assertEqual(residual["method"], "paper_residual_structured_tst")
        self.assertEqual(residual["loss_mode"], "repeated_ce")
        self.assertEqual(residual["lr_scheduler"], "wsd")

    def test_adaptive_method_uses_adaptive_composer_and_larger_position_budget(self):
        cfg = TrainConfig(
            method="adaptive_residual_tst",
            baseline_seq_len=128,
            max_seq_len=128,
            superpose_size=8,
            adaptive_min_superpose_size=4,
            paper_equal_flops=1,
        )
        validate_config(cfg)
        self.assertEqual(method_to_mode(cfg), "adaptive_residual_structured")
        self.assertEqual(train_raw_seq_len(cfg), 1024)
        self.assertEqual(model_seq_len(cfg), 256)

    def test_calibrated_methods_validate_protocol_fields(self):
        cfg = TrainConfig(
            method="adaptive_calibrated_tst",
            superpose_size=8,
            adaptive_min_superpose_size=4,
            calibration_loss_weight=0.1,
            calibration_seq_len=192,
            calibration_interval=2,
        )
        validate_config(cfg)
        self.assertEqual(method_to_mode(cfg), "adaptive_residual_structured")

    def test_conflict_adaptive_calibrated_uses_residual_composer(self):
        cfg = TrainConfig(
            method="conflict_adaptive_calibrated_tst",
            superpose_size=4,
            calibration_loss_weight=0.3,
            calibration_seq_len=384,
            calibration_loss_weight_min=0.0,
            calibration_loss_weight_max=0.5,
            calibration_ema_beta=0.98,
        )
        validate_config(cfg)
        self.assertEqual(method_to_mode(cfg), "residual_structured")

    def test_adaptive_calibration_weight_is_clamped(self):
        cfg = TrainConfig(
            method="conflict_adaptive_calibrated_tst",
            calibration_loss_weight=0.3,
            calibration_loss_weight_min=0.05,
            calibration_loss_weight_max=0.5,
            calibration_ema_beta=0.5,
        )
        state = CalibrationState()
        weight = calibration_weight_for_loss(cfg, state, torch.tensor(1.0), torch.tensor(10.0))
        self.assertEqual(weight, 0.5)
        self.assertEqual(state.last_weight, 0.5)
        self.assertIsNotNone(state.last_ratio_ema)

    def test_invalid_adaptive_calibration_bounds_are_rejected(self):
        cfg = TrainConfig(superpose_size=4, calibration_loss_weight_min=0.6, calibration_loss_weight_max=0.5)
        with self.assertRaisesRegex(ValueError, "calibration_loss_weight_max"):
            validate_config(cfg)

    def test_adaptive_recovery_switch_triggers_from_ratio_ema(self):
        cfg = TrainConfig(
            method="conflict_adaptive_calibrated_tst",
            max_steps=100,
            superpose_size=4,
            recovery_ratio=0.5,
            calibration_adaptive=1,
            adaptive_recovery_switch=1,
            adaptive_recovery_threshold=0.66,
            adaptive_recovery_patience=2,
            adaptive_recovery_min_superpose_steps=10,
        )
        validate_config(cfg)
        recovery_state = AdaptiveRecoveryState()
        calibration_state = CalibrationState(last_ratio_ema=0.65)
        maybe_update_adaptive_recovery(cfg, recovery_state, calibration_state, 9, "superposition")
        self.assertIsNone(recovery_state.triggered_step)
        maybe_update_adaptive_recovery(cfg, recovery_state, calibration_state, 10, "superposition")
        self.assertEqual(recovery_state.triggered_step, 11)
        self.assertEqual(phase_for_step(cfg, 11, recovery_state.triggered_step), "recovery")

    def test_adaptive_recovery_deadline_caps_planned_recovery_start(self):
        cfg = TrainConfig(
            method="conflict_adaptive_calibrated_tst",
            max_steps=100,
            superpose_size=4,
            recovery_ratio=0.5,
            calibration_adaptive=1,
            adaptive_recovery_switch=1,
            adaptive_recovery_min_superpose_steps=10,
            adaptive_recovery_max_superpose_steps=40,
        )
        validate_config(cfg)
        self.assertEqual(recovery_start_step(cfg), 50)
        self.assertEqual(adaptive_recovery_deadline(cfg), 40)

    def test_adaptive_recovery_95k_config_is_valid(self):
        cfg_data = yaml.safe_load(Path("configs/hst/adaptive_recovery_conflict_s4_r05_w03_seq384_95k.yaml").read_text(encoding="utf-8"))
        cfg = TrainConfig(**cfg_data)
        validate_config(cfg)
        self.assertEqual(cfg.max_steps, 95000)
        self.assertEqual(cfg.lr_schedule_steps, 95000)
        self.assertEqual(cfg.recovery_ratio, 0.5)
        self.assertEqual(cfg.adaptive_recovery_switch, 1)
        self.assertEqual(method_to_mode(cfg), "residual_structured")


if __name__ == "__main__":
    unittest.main()
