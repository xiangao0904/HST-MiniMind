import unittest

try:
    import torch
    from torch import nn

    from model.hst_losses import ntp_loss, ordered_slot_loss, repeated_token_ce_loss
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

    def test_repeated_ce_loss_accepts_target_mask(self):
        logits = torch.randn(2, 4, 17)
        targets = torch.randint(0, 17, (2, 4, 3))
        mask = torch.ones(2, 4, 3, dtype=torch.bool)
        mask[:, -1, 1:] = False
        self.assertTrue(torch.isfinite(repeated_token_ce_loss(logits, targets, mask)))

    def test_ordered_slot_loss_is_finite(self):
        hidden = torch.randn(2, 4, 8)
        targets = torch.randint(0, 17, (2, 4, 3))
        head = nn.Linear(8, 17)
        out_slots = nn.Embedding(3, 8)
        self.assertTrue(torch.isfinite(ordered_slot_loss(hidden, targets, head, out_slots)))

if __name__ == "__main__":
    unittest.main()
