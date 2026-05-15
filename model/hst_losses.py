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
