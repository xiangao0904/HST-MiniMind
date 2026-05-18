import unittest

try:
    import torch
    from torch import nn

    from model.hst_losses import ntp_loss, ordered_slot_loss, repeated_token_ce_loss, sparse_anchor_residual_loss
except Exception:
    torch = None


@unittest.skipIf(torch is None, "torch is not installed")
class HstLossTest(unittest.TestCase):
    def test_ntp_loss_is_finite(self):
        logits = torch.randn(2, 5, 17)
        ids = torch.randint(0, 17, (2, 5))
        self.assertTrue(torch.isfinite(ntp_loss(logits, ids)))

    def test_repeated_ce_loss_is_finite(self):
        logits = torch.randn(2, 4, 17)
        targets = torch.randint(0, 17, (2, 4, 3))
        self.assertTrue(torch.isfinite(repeated_token_ce_loss(logits, targets)))

    def test_ordered_slot_loss_is_finite(self):
        hidden = torch.randn(2, 4, 8)
        targets = torch.randint(0, 17, (2, 4, 3))
        head = nn.Linear(8, 17)
        out_slots = nn.Embedding(3, 8)
        self.assertTrue(torch.isfinite(ordered_slot_loss(hidden, targets, head, out_slots)))

    def test_sparse_anchor_residual_loss_is_finite(self):
        anchor_logits = torch.randn(2, 4, 17)
        hidden = torch.randn(2, 4, 8)
        anchor_targets = torch.randint(0, 17, (2, 4))
        residual_targets = torch.randint(0, 8, (2, 4, 2))
        residual_head = nn.Linear(8, 16)
        gate_mask = torch.tensor([[1, 0, 1, 0], [1, 1, 0, 0]], dtype=torch.float32)
        loss = sparse_anchor_residual_loss(
            anchor_logits,
            hidden,
            anchor_targets,
            residual_targets,
            residual_head,
            residual_vocab_size=8,
            residual_gate_mask=gate_mask,
            residual_loss_weight=0.5,
        )
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
