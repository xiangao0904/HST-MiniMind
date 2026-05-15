import unittest

try:
    import torch

    from trainer.train_hst_pretrain import TrainConfig, batch_for_phase, baseline_seq_len, token_counts, train_raw_seq_len
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


if __name__ == "__main__":
    unittest.main()
