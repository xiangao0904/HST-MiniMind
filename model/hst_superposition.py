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
    adaptive_min_superpose_size: int = 4
    adaptive_hard_token_types: str = "3,9,10"
    adaptive_hard_threshold: float = 0.0


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
        hard_types = [int(item) for item in config.adaptive_hard_token_types.split(",") if item.strip()]
        self.register_buffer("adaptive_hard_type_ids", torch.tensor(hard_types, dtype=torch.long), persistent=False)
        self.last_metadata: dict[str, int | float | str] = {}

    def compose(self, input_ids: torch.Tensor) -> dict[str, torch.Tensor | dict[str, int | str]]:
        if self.config.mode == "adaptive_residual_structured":
            return self._compose_adaptive(input_ids)
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
                "latent_tokens": local_z.size(0) * local_z.size(1),
                "usable_tokens": ids.numel(),
                "effective_superpose_size": float(usable_len / max(1, chunks.size(1))),
                "mode": self.config.mode,
            },
        }
        if self.config.mode == "sparse_anchor_residual":
            result.update(self._sparse_anchor_targets(source_chunks, target_chunks, embeds, local_z))
        self.last_metadata = dict(result["metadata"])
        return result

    def _compose_adaptive(self, input_ids: torch.Tensor) -> dict[str, torch.Tensor | dict[str, int | float | str]]:
        s = self.config.superpose_size
        min_s = self.config.adaptive_min_superpose_size
        if min_s <= 0 or min_s > s or s % min_s != 0:
            raise ValueError("adaptive_min_superpose_size must divide superpose_size")
        usable_len = (input_ids.size(1) // s) * s
        if usable_len < s * 2:
            raise ValueError("sequence too short for adaptive superposition loss")

        sample_groups: list[list[torch.Tensor]] = []
        hard_windows = 0
        total_windows = 0
        for sample in input_ids[:, :usable_len]:
            groups: list[torch.Tensor] = []
            for window in sample.view(-1, s):
                total_windows += 1
                if self._adaptive_window_is_hard(window):
                    hard_windows += 1
                    for start in range(0, s, min_s):
                        groups.append(window[start : start + min_s])
                else:
                    groups.append(window)
            if len(groups) < 2:
                raise ValueError("adaptive superposition requires at least two packed groups")
            sample_groups.append(groups)

        bsz = input_ids.size(0)
        max_source_len = max(len(groups) - 1 for groups in sample_groups)
        source_ids = input_ids.new_zeros(bsz, max_source_len, s)
        target_ids = input_ids.new_zeros(bsz, max_source_len, s)
        source_mask = torch.zeros(bsz, max_source_len, s, dtype=torch.bool, device=input_ids.device)
        target_mask = torch.zeros_like(source_mask)
        source_lengths = []
        target_slots = 0

        for batch_idx, groups in enumerate(sample_groups):
            source_lengths.append(len(groups) - 1)
            for group_idx, group in enumerate(groups[:-1]):
                source_ids[batch_idx, group_idx, : group.numel()] = group
                source_mask[batch_idx, group_idx, : group.numel()] = True
                target = groups[group_idx + 1]
                target_ids[batch_idx, group_idx, : target.numel()] = target
                target_mask[batch_idx, group_idx, : target.numel()] = True
                target_slots += int(target.numel())

        local_z = self._adaptive_residual_structured(source_ids, source_mask)
        attention_mask = torch.arange(max_source_len, device=input_ids.device).view(1, -1) < torch.tensor(
            source_lengths, device=input_ids.device
        ).view(-1, 1)
        hard_rate = hard_windows / max(1, total_windows)
        effective_s = target_slots / max(1, int(target_mask[:, :, 0].sum().item()))
        metadata = {
            "usable_len": usable_len,
            "num_chunks": max_source_len + 1,
            "latent_tokens": int(attention_mask.sum().item()),
            "usable_tokens": int(input_ids.size(0) * usable_len),
            "adaptive_hard_window_rate": float(hard_rate),
            "adaptive_effective_superpose_size": float(effective_s),
            "mode": self.config.mode,
        }
        result = {
            "inputs_embeds": local_z,
            "chunk_targets": target_ids,
            "chunk_target_mask": target_mask,
            "attention_mask": attention_mask.long(),
            "metadata": metadata,
        }
        self.last_metadata = dict(metadata)
        return result

    def _adaptive_window_is_hard(self, window: torch.Tensor) -> bool:
        if self.adaptive_hard_type_ids.numel() == 0:
            return False
        type_ids = self.token_type_ids[window.clamp_max(self.token_type_ids.numel() - 1)]
        hard = (type_ids.unsqueeze(-1) == self.adaptive_hard_type_ids.view(1, -1)).any(dim=-1)
        return float(hard.float().mean().detach().cpu()) > self.config.adaptive_hard_threshold

    def _adaptive_residual_structured(self, source_ids: torch.Tensor, source_mask: torch.Tensor) -> torch.Tensor:
        embeds = self.token_embedding(source_ids)
        mask = source_mask.to(embeds.dtype).unsqueeze(-1)
        count = mask.sum(dim=2).clamp_min(1.0)
        z_mean = (embeds * mask).sum(dim=2) / count
        order_residual = self._masked_order_residual(embeds, z_mean, source_mask)
        local_z = z_mean + self.config.order_alpha * order_residual
        hier_residual = self._causal_block_summary(local_z, add_block_type=False) - local_z
        return local_z + self.config.hier_alpha * hier_residual

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

    def _masked_order_residual(self, embeds: torch.Tensor, z_mean: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        slots = torch.arange(self.config.superpose_size, device=embeds.device)
        if self.config.slot_gate_type == "diagonal":
            slot_weight = self.slot_gate.view(1, 1, self.config.superpose_size, self.hidden_size)
        elif self.config.slot_gate_type == "embedding":
            slot_weight = torch.tanh(self.slot_embed(slots)).view(1, 1, -1, self.hidden_size)
        else:
            raise ValueError(f"unknown slot_gate_type: {self.config.slot_gate_type}")
        mask_f = mask.to(embeds.dtype).unsqueeze(-1)
        active = mask_f.sum(dim=2, keepdim=True).clamp_min(1.0)
        slot_mean = (slot_weight * mask_f).sum(dim=2, keepdim=True) / active
        centered = (slot_weight - slot_mean) * mask_f
        token_delta = embeds - z_mean.unsqueeze(2)
        return (token_delta * centered).sum(dim=2) / active.squeeze(2)

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
