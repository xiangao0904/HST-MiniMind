import unittest

try:
    import torch
    from torch import nn

    from model.hst_superposition import SuperpositionComposer, SuperpositionConfig
except Exception:
    torch = None


@unittest.skipIf(torch is None, "torch is not installed")
class SuperpositionShapeTest(unittest.TestCase):
    def _run_mode(self, mode, superpose_size=2):
        embed = nn.Embedding(64, 16)
        cfg = SuperpositionConfig(mode=mode, superpose_size=superpose_size, hidden_size=16, vocab_size=64)
        composer = SuperpositionComposer(embed, 16, 64, cfg)
        batch = torch.randint(0, 64, (3, 17))
        out = composer.compose(batch)
        self.assertEqual(out["inputs_embeds"].shape, (3, (16 // superpose_size) - 1, 16))
        self.assertEqual(out["chunk_targets"].shape, (3, (16 // superpose_size) - 1, superpose_size))

    def test_vanilla_shapes(self):
        self._run_mode("mean")

    def test_order_shapes(self):
        self._run_mode("order_aware")

    def test_boundary_shapes(self):
        self._run_mode("boundary_aware")

    def test_hierarchical_shapes(self):
        self._run_mode("hierarchical", 4)


if __name__ == "__main__":
    unittest.main()
