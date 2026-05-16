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

    def test_residual_structured_shapes(self):
        self._run_mode("residual_structured", 4)

    def test_order_aware_distinguishes_reversed_chunk(self):
        embed = nn.Embedding(16, 4)
        with torch.no_grad():
            embed.weight.zero_()
            embed.weight[1] = torch.tensor([1.0, 0.0, 0.0, 0.0])
            embed.weight[2] = torch.tensor([0.0, 1.0, 0.0, 0.0])
            embed.weight[3] = torch.tensor([0.0, 0.0, 1.0, 0.0])
        cfg = SuperpositionConfig(mode="order_aware", superpose_size=3, hidden_size=4, vocab_size=16)
        composer = SuperpositionComposer(embed, 4, 16, cfg)
        with torch.no_grad():
            composer.slot_embed.weight[0] = torch.tensor([0.0, 0.0, 0.0, 0.0])
            composer.slot_embed.weight[1] = torch.tensor([0.5, 0.5, 0.5, 0.5])
            composer.slot_embed.weight[2] = torch.tensor([1.0, 1.0, 1.0, 1.0])
        forward = composer.compose(torch.tensor([[1, 2, 3, 4, 5, 6]]))["inputs_embeds"]
        reversed_chunk = composer.compose(torch.tensor([[3, 2, 1, 4, 5, 6]]))["inputs_embeds"]
        self.assertFalse(torch.allclose(forward[:, 0], reversed_chunk[:, 0]))

    def test_hierarchical_summary_does_not_use_future_chunks(self):
        embed = nn.Embedding(64, 8)
        cfg = SuperpositionConfig(
            mode="hierarchical",
            superpose_size=2,
            hidden_size=8,
            vocab_size=64,
            chunks_per_block=4,
            hier_alpha=0.5,
        )
        composer = SuperpositionComposer(embed, 8, 64, cfg)
        prefix = [1, 2, 3, 4]
        first = torch.tensor([prefix + [5, 6, 7, 8, 9, 10]])
        second = torch.tensor([prefix + [55, 56, 57, 58, 59, 60]])
        first_z = composer.compose(first)["inputs_embeds"]
        second_z = composer.compose(second)["inputs_embeds"]
        self.assertTrue(torch.allclose(first_z[:, :2], second_z[:, :2]))
        self.assertFalse(torch.allclose(first_z[:, 2], second_z[:, 2]))

    def test_residual_structured_zero_weights_match_mean(self):
        embed = nn.Embedding(64, 8)
        batch = torch.randint(0, 64, (2, 16))
        mean_cfg = SuperpositionConfig(mode="mean", superpose_size=4, hidden_size=8, vocab_size=64)
        residual_cfg = SuperpositionConfig(
            mode="residual_structured",
            superpose_size=4,
            hidden_size=8,
            vocab_size=64,
            order_alpha=0.0,
            hier_alpha=0.0,
        )
        mean_z = SuperpositionComposer(embed, 8, 64, mean_cfg).compose(batch)["inputs_embeds"]
        residual_z = SuperpositionComposer(embed, 8, 64, residual_cfg).compose(batch)["inputs_embeds"]
        self.assertTrue(torch.allclose(mean_z, residual_z))

    def test_residual_structured_order_signal_is_small_residual(self):
        embed = nn.Embedding(16, 4)
        with torch.no_grad():
            embed.weight.zero_()
            embed.weight[1] = torch.tensor([1.0, 0.0, 0.0, 0.0])
            embed.weight[2] = torch.tensor([0.0, 1.0, 0.0, 0.0])
            embed.weight[3] = torch.tensor([0.0, 0.0, 1.0, 0.0])
            embed.weight[4] = torch.tensor([0.0, 0.0, 0.0, 1.0])
        cfg = SuperpositionConfig(
            mode="residual_structured",
            superpose_size=4,
            hidden_size=4,
            vocab_size=16,
            order_alpha=0.05,
            hier_alpha=0.0,
        )
        composer = SuperpositionComposer(embed, 4, 16, cfg)
        with torch.no_grad():
            composer.slot_embed.weight[0] = torch.tensor([-1.0, -1.0, -1.0, -1.0])
            composer.slot_embed.weight[1] = torch.tensor([-0.3, -0.3, -0.3, -0.3])
            composer.slot_embed.weight[2] = torch.tensor([0.3, 0.3, 0.3, 0.3])
            composer.slot_embed.weight[3] = torch.tensor([1.0, 1.0, 1.0, 1.0])
        forward = composer.compose(torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]]))["inputs_embeds"]
        reversed_chunk = composer.compose(torch.tensor([[4, 3, 2, 1, 5, 6, 7, 8]]))["inputs_embeds"]
        self.assertFalse(torch.allclose(forward[:, 0], reversed_chunk[:, 0]))
        mean_cfg = SuperpositionConfig(mode="mean", superpose_size=4, hidden_size=4, vocab_size=16)
        mean_z = SuperpositionComposer(embed, 4, 16, mean_cfg).compose(torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]]))[
            "inputs_embeds"
        ]
        self.assertLess(torch.norm(forward[:, 0] - mean_z[:, 0]), torch.norm(mean_z[:, 0]))

    def test_residual_structured_hierarchy_does_not_use_future_chunks(self):
        embed = nn.Embedding(64, 8)
        cfg = SuperpositionConfig(
            mode="residual_structured",
            superpose_size=2,
            hidden_size=8,
            vocab_size=64,
            chunks_per_block=4,
            order_alpha=0.05,
            hier_alpha=0.05,
        )
        composer = SuperpositionComposer(embed, 8, 64, cfg)
        prefix = [1, 2, 3, 4]
        first = torch.tensor([prefix + [5, 6, 7, 8, 9, 10]])
        second = torch.tensor([prefix + [55, 56, 57, 58, 59, 60]])
        first_z = composer.compose(first)["inputs_embeds"]
        second_z = composer.compose(second)["inputs_embeds"]
        self.assertTrue(torch.allclose(first_z[:, :2], second_z[:, :2]))
        self.assertFalse(torch.allclose(first_z[:, 2], second_z[:, 2]))

    def test_residual_structured_first_chunk_keeps_mean_without_order(self):
        embed = nn.Embedding(64, 8)
        batch = torch.randint(0, 64, (2, 16))
        mean_cfg = SuperpositionConfig(mode="mean", superpose_size=4, hidden_size=8, vocab_size=64)
        residual_cfg = SuperpositionConfig(
            mode="residual_structured",
            superpose_size=4,
            hidden_size=8,
            vocab_size=64,
            order_alpha=0.0,
            hier_alpha=0.5,
        )
        mean_z = SuperpositionComposer(embed, 8, 64, mean_cfg).compose(batch)["inputs_embeds"]
        residual_z = SuperpositionComposer(embed, 8, 64, residual_cfg).compose(batch)["inputs_embeds"]
        self.assertTrue(torch.allclose(mean_z[:, 0], residual_z[:, 0]))


if __name__ == "__main__":
    unittest.main()
