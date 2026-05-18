from __future__ import annotations

import torch
import torch.nn.functional as F


def ntp_loss(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    logits = logits[:, :-1, :].contiguous()
    targets = input_ids[:, 1:].contiguous()
    return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))


def repeated_token_ce_loss(logits: torch.Tensor, chunk_targets: torch.Tensor) -> torch.Tensor:
    if logits.size(1) != chunk_targets.size(1):
        raise ValueError("logits and chunk_targets must have the same chunk length")
    vocab = logits.size(-1)
    expanded = logits.unsqueeze(2).expand(-1, -1, chunk_targets.size(-1), -1)
    return F.cross_entropy(expanded.reshape(-1, vocab), chunk_targets.reshape(-1))


def ordered_slot_loss(hidden: torch.Tensor, chunk_targets: torch.Tensor, lm_head, out_slot_embed) -> torch.Tensor:
    slot_hidden = hidden.unsqueeze(2) + out_slot_embed.weight.view(1, 1, -1, hidden.size(-1))
    logits = lm_head(slot_hidden)
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), chunk_targets.reshape(-1))


def sparse_anchor_residual_loss(
    anchor_logits: torch.Tensor,
    hidden: torch.Tensor,
    anchor_targets: torch.Tensor,
    residual_code_targets: torch.Tensor,
    residual_head,
    residual_vocab_size: int,
    residual_gate_mask: torch.Tensor,
    residual_loss_weight: float = 1.0,
) -> torch.Tensor:
    anchor_loss = F.cross_entropy(anchor_logits.reshape(-1, anchor_logits.size(-1)), anchor_targets.reshape(-1))
    residual_logits = residual_head(hidden).view(hidden.size(0), hidden.size(1), residual_code_targets.size(-1), residual_vocab_size)
    active = residual_gate_mask.unsqueeze(-1).expand_as(residual_code_targets) > 0
    if active.any():
        residual_loss = F.cross_entropy(residual_logits[active], residual_code_targets[active])
    else:
        residual_loss = anchor_logits.new_zeros(())
    return anchor_loss + residual_loss_weight * residual_loss
