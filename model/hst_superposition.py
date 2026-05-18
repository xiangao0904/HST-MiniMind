from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class SuperpositionConfig:
    mode: str = "mean"
    superpose_size: int = 2
    hidden_size: int = 256
    vocab_size: int = 32000
    slot_gate_type: str = "embedding"
    type_vocab_size: int = 11
    block_mode: str = "fixed"
    chunks_per_block: int = 8
    order_alpha: float = 0.1
    hier_alpha: float = 0.1
    anchor_slot_idx: int = 0
    residual_codebook_size: int = 256
    sar_gate_threshold: float = 0.05


class SuperpositionComposer(nn.Module):
    def __init__(
        self,
        token_embedding: nn.Embedding,
        hidden_size: int,
        vocab_size: int,
        config: SuperpositionConfig,
        token_type_ids: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.token_embedding = token_embedding
        self.config = config
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        s = config.superpose_size
        self.slot_embed = nn.Embedding(s, hidden_size)
        self.slot_gate = nn.Parameter(torch.ones(s, hidden_size))
        self.type_embed = nn.Embedding(config.type_vocab_size, hidden_size)
        self.block_type_embed = nn.Embedding(1, hidden_size)
        residual_slots = max(1, s - 1)
        self.residual_codebook = nn.Embedding(config.residual_codebook_size, hidden_size)
        self.residual_head = nn.Linear(hidden_size, residual_slots * config.residual_codebook_size)
        if token_type_ids is None:
            token_type_ids = torch.zeros(vocab_size, dtype=torch.long)
        self.register_buffer("token_type_ids", token_type_ids.long(), persistent=False)

    def compose(self, input_ids: torch.Tensor) -> dict[str, torch.Tensor | dict[str, int | str]]:
        s = self.config.superpose_size
        if s <= 0:
            raise ValueError("superpose_size must be positive")
        usable_len = (input_ids.size(1) // s) * s
        if usable_len < s * 2:
            raise ValueError("sequence too short for next-chunk superposition loss")
        ids = input_ids[:, :usable_len]
        chunks = ids.view(ids.size(0), -1, s)
        source_chunks = chunks[:, :-1, :]
        target_chunks = chunks[:, 1:, :]

        embeds = self.token_embedding(source_chunks)
        local_z = self._compose_local(source_chunks, embeds)
        if self.config.mode == "hierarchical":
            local_z = self._add_hierarchy(local_z)
        attention_mask = torch.ones(local_z.shape[:2], dtype=torch.long, device=local_z.device)
        result = {
            "inputs_embeds": local_z,
            "chunk_targets": target_chunks,
            "attention_mask": attention_mask,
            "metadata": {
                "usable_len": usable_len,
                "num_chunks": chunks.size(1),
                "mode": self.config.mode,
            },
        }
        if self.config.mode == "sparse_anchor_residual":
            result.update(self._sparse_anchor_targets(source_chunks, target_chunks, embeds, local_z))
        return result

    def _compose_local(self, source_chunks: torch.Tensor, embeds: torch.Tensor) -> torch.Tensor:
        mode = self.config.mode
        if mode in {"mean", "vanilla"}:
            return embeds.mean(dim=2)
        if mode == "order_aware":
            return self._order_aware(embeds)
        if mode == "boundary_aware":
            type_ids = self.token_type_ids[source_chunks.clamp_max(self.token_type_ids.numel() - 1)]
            return (embeds + self.type_embed(type_ids)).mean(dim=2)
        if mode == "hierarchical":
            return self._order_aware(embeds)
        if mode == "residual_structured":
            return self._residual_structured(embeds)
        if mode == "sparse_anchor_residual":
            return self._sparse_anchor_residual(embeds)
        raise ValueError(f"unknown superposition mode: {mode}")

    def _order_aware(self, embeds: torch.Tensor) -> torch.Tensor:
        slots = torch.arange(self.config.superpose_size, device=embeds.device)
        if self.config.slot_gate_type == "diagonal":
            gated = embeds * self.slot_gate.view(1, 1, self.config.superpose_size, self.hidden_size)
            return gated.sum(dim=2) / self.config.superpose_size
        if self.config.slot_gate_type == "embedding":
            slot_scale = 1.0 + torch.tanh(self.slot_embed(slots)).view(1, 1, -1, self.hidden_size)
            return (embeds * slot_scale).sum(dim=2) / self.config.superpose_size
        raise ValueError(f"unknown slot_gate_type: {self.config.slot_gate_type}")

    def _order_residual(self, embeds: torch.Tensor, z_mean: torch.Tensor) -> torch.Tensor:
        slots = torch.arange(self.config.superpose_size, device=embeds.device)
        if self.config.slot_gate_type == "diagonal":
            slot_weight = self.slot_gate.view(1, 1, self.config.superpose_size, self.hidden_size)
        elif self.config.slot_gate_type == "embedding":
            slot_weight = torch.tanh(self.slot_embed(slots)).view(1, 1, -1, self.hidden_size)
        else:
            raise ValueError(f"unknown slot_gate_type: {self.config.slot_gate_type}")
        slot_weight = slot_weight - slot_weight.mean(dim=2, keepdim=True)
        token_delta = embeds - z_mean.unsqueeze(2)
        return (token_delta * slot_weight).sum(dim=2) / self.config.superpose_size

    def _residual_structured(self, embeds: torch.Tensor) -> torch.Tensor:
        z_mean = embeds.mean(dim=2)
        order_residual = self._order_residual(embeds, z_mean)
        local_z = z_mean + self.config.order_alpha * order_residual
        hier_residual = self._causal_block_summary(local_z, add_block_type=False) - local_z
        return local_z + self.config.hier_alpha * hier_residual

    def _sparse_anchor_residual(self, embeds: torch.Tensor) -> torch.Tensor:
        z_mean = embeds.mean(dim=2)
        order_residual = self._order_residual(embeds, z_mean)
        gate_mask = self._sar_gate_mask(order_residual)
        local_z = z_mean + gate_mask * self.config.order_alpha * order_residual
        if self.config.hier_alpha > 0.0:
            hier_residual = self._causal_block_summary(local_z, add_block_type=False) - local_z
            local_z = local_z + gate_mask * self.config.hier_alpha * hier_residual
        return local_z

    def _sar_gate_mask(self, residual: torch.Tensor) -> torch.Tensor:
        gate_score = residual.pow(2).mean(dim=2, keepdim=True).sqrt()
        return (gate_score > self.config.sar_gate_threshold).to(residual.dtype)

    def _sparse_anchor_targets(
        self,
        source_chunks: torch.Tensor,
        target_chunks: torch.Tensor,
        source_embeds: torch.Tensor,
        local_z: torch.Tensor,
    ) -> dict[str, torch.Tensor | dict[str, float]]:
        anchor_idx = self.config.anchor_slot_idx
        target_embeds = self.token_embedding(target_chunks)
        anchor_targets = target_chunks[:, :, anchor_idx]
        if self.config.superpose_size <= 1:
            residual_code_targets = target_chunks.new_zeros(target_chunks.size(0), target_chunks.size(1), 1)
        else:
            residual_slots = [slot for slot in range(self.config.superpose_size) if slot != anchor_idx]
            residual_embeds = target_embeds[:, :, residual_slots, :]
            anchor_embed = target_embeds[:, :, anchor_idx : anchor_idx + 1, :]
            delta = residual_embeds - anchor_embed
            codebook = self.residual_codebook.weight
            dist = (
                delta.pow(2).sum(dim=-1, keepdim=True)
                - 2.0 * torch.matmul(delta, codebook.t())
                + codebook.pow(2).sum(dim=1).view(1, 1, 1, -1)
            )
            residual_code_targets = dist.argmin(dim=-1)
        source_mean = source_embeds.mean(dim=2)
        source_residual = self._order_residual(source_embeds, source_mean)
        residual_gate_mask = self._sar_gate_mask(source_residual).squeeze(-1)
        metadata = {
            "sar_gate_rate": float(residual_gate_mask.float().mean().detach().cpu()),
            "sar_anchor_slot": float(anchor_idx),
        }
        return {
            "anchor_targets": anchor_targets,
            "residual_code_targets": residual_code_targets,
            "residual_gate_mask": residual_gate_mask,
            "metadata": {**metadata, "usable_len": target_chunks.size(1), "mode": self.config.mode},
        }

    def _add_hierarchy(self, local_z: torch.Tensor) -> torch.Tensor:
        block_z = self._causal_block_summary(local_z)
        return local_z + self.config.hier_alpha * block_z

    def _causal_block_summary(self, local_z: torch.Tensor, add_block_type: bool = True) -> torch.Tensor:
        bsz, chunk_len, hidden = local_z.shape
        block = max(1, self.config.chunks_per_block)
        padded_len = ((chunk_len + block - 1) // block) * block
        if padded_len != chunk_len:
            pad = local_z.new_zeros(bsz, padded_len - chunk_len, hidden)
            work = torch.cat([local_z, pad], dim=1)
            valid = torch.cat(
                [
                    local_z.new_ones(bsz, chunk_len, 1),
                    local_z.new_zeros(bsz, padded_len - chunk_len, 1),
                ],
                dim=1,
            )
        else:
            work = local_z
            valid = local_z.new_ones(bsz, chunk_len, 1)
        block_work = work.view(bsz, -1, block, hidden)
        block_valid = valid.view(bsz, -1, block, 1)
        prefix_sum = block_work.cumsum(dim=2)
        prefix_count = block_valid.cumsum(dim=2).clamp_min(1.0)
        block_z = (prefix_sum / prefix_count).view(bsz, padded_len, hidden)[:, :chunk_len, :]
        if add_block_type:
            block_z = block_z + self.block_type_embed.weight.view(1, 1, hidden)
        return block_z
