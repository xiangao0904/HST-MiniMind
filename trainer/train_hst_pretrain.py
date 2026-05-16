#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from utils.hst_path_safety import ensure_run_output_dir, ensure_within_project, safe_mkdir

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except Exception as exc:  # pragma: no cover - exercised on machines without torch.
    raise SystemExit("PyTorch is required for training. Install it in a project/local environment.") from exc

from model.hst_losses import ntp_loss, ordered_slot_loss, repeated_token_ce_loss
from model.hst_superposition import SuperpositionComposer, SuperpositionConfig
from model.hst_token_types import build_token_type_cache


METHODS = {
    "ntp_baseline",
    "vanilla_tst",
    "order_aware_tst",
    "boundary_aware_tst",
    "hierarchical_tst",
    "residual_structured_tst",
}


@dataclass
class TrainConfig:
    method: str = "ntp_baseline"
    experiment_method: str = ""
    data_path: str = "./hst_tmp/tiny_pretrain.jsonl"
    run_name: str = "debug"
    output_dir: str = "./hst_runs/debug"
    max_steps: int = 3
    global_step_offset: int = 0
    eval_interval: int = 1
    dense_eval_interval: int = 0
    dense_eval_window: int = 0
    dense_eval_anchor_step: int = 0
    online_eval_max_batches: int = 2
    save_interval: int = 100
    seed: int = 42
    superpose_size: int = 2
    superpose_mode: str = "mean"
    loss_mode: str = "repeated_ce"
    recovery_ratio: float = 0.0
    learning_rate: float = 3e-4
    batch_size: int = 2
    max_seq_len: int = 128
    baseline_seq_len: int = 0
    paper_equal_flops: int = 0
    device: str = "cpu"
    dry_run: int = 0
    debug: int = 0
    from_resume: int = 0
    tokenizer_backend: str = "char"
    tokenizer_path: str = "./tokenizer/minimind_tokenizer"
    use_tokenized_cache: int = 0
    tokenized_cache_path: str = ""
    block_mode: str = "fixed"
    chunks_per_block: int = 8
    order_alpha: float = 0.1
    hier_alpha: float = 0.1
    slot_gate_type: str = "embedding"
    type_vocab_size: int = 11
    log_jsonl: str = "metrics.jsonl"
    use_wandb: int = 0
    wandb_project: str = "hst-minimind"
    wandb_entity: str = ""
    use_swanlab: int = 0
    hidden_size: int = 128
    num_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.0
    phase_override: str = ""
    init_checkpoint_path: str = ""


class CharTokenizer:
    def __init__(self, texts: list[str]) -> None:
        chars = sorted(set("".join(texts)))
        self.id_to_text = {0: "<pad>", 1: "<bos>", 2: "<eos>", 3: "<unk>"}
        for ch in chars:
            if ch not in self.id_to_text.values():
                self.id_to_text[len(self.id_to_text)] = ch
        self.text_to_id = {v: k for k, v in self.id_to_text.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_text)

    def encode(self, text: str, max_len: int) -> list[int]:
        ids = [self.text_to_id["<bos>"]]
        ids.extend(self.text_to_id.get(ch, self.text_to_id["<unk>"]) for ch in text)
        ids.append(self.text_to_id["<eos>"])
        ids = ids[:max_len]
        if len(ids) < max_len:
            ids.extend([self.text_to_id["<pad>"]] * (max_len - len(ids)))
        return ids


class MiniMindTokenizer:
    def __init__(self, tokenizer_path: Path) -> None:
        try:
            from tokenizers import Tokenizer
        except Exception as exc:
            raise RuntimeError("tokenizer_backend=minimind requires the tokenizers package") from exc
        path = tokenizer_path / "tokenizer.json" if tokenizer_path.is_dir() else tokenizer_path
        self.tokenizer = Tokenizer.from_file(str(path))
        self.pad_token_id = self._first_token_id(["<pad>", "[PAD]", "<unk>", "<eos>"], default=0)
        self.eos_token_id = self._first_token_id(["<eos>", "</s>", "<|endoftext|>"], default=None)
        self.id_to_text = {
            token_id: (self.tokenizer.id_to_token(token_id) or "")
            for token_id in range(self.tokenizer.get_vocab_size())
        }

    def _first_token_id(self, candidates: list[str], default: int | None) -> int | None:
        for token in candidates:
            token_id = self.tokenizer.token_to_id(token)
            if token_id is not None:
                return int(token_id)
        return default

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    def encode(self, text: str, max_len: int) -> list[int]:
        ids = self.tokenizer.encode(text).ids
        if self.eos_token_id is not None:
            ids.append(self.eos_token_id)
        ids = ids[:max_len]
        if len(ids) < max_len:
            ids.extend([int(self.pad_token_id)] * (max_len - len(ids)))
        return ids


class JsonlTextDataset(Dataset):
    def __init__(self, path: Path, tokenizer: CharTokenizer, max_seq_len: int) -> None:
        self.rows = load_texts(path)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.tensor(self.tokenizer.encode(self.rows[idx], self.max_seq_len), dtype=torch.long)


class TokenizedTensorDataset(Dataset):
    def __init__(self, path: Path, max_seq_len: int) -> None:
        data = torch.load(path, map_location="cpu")
        if isinstance(data, dict):
            self.input_ids = data["input_ids"]
            self.vocab_size = int(data["vocab_size"])
            raw_id_to_text = data.get("id_to_text") or []
            self.id_to_text = {i: str(text) for i, text in enumerate(raw_id_to_text)}
        else:
            self.input_ids = data
            self.vocab_size = int(self.input_ids.max().item()) + 1
            self.id_to_text = {}
        if self.input_ids.size(1) < max_seq_len:
            raise ValueError(f"tokenized cache seq_len {self.input_ids.size(1)} is smaller than required {max_seq_len}")
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return self.input_ids.size(0)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.input_ids[idx, : self.max_seq_len].long()


class TinyCausalLM(nn.Module):
    def __init__(self, vocab_size: int, max_seq_len: int, hidden_size: int, num_layers: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_embedding = nn.Embedding(max_seq_len, hidden_size)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.ln_f = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, input_ids: torch.Tensor | None = None, inputs_embeds: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("input_ids or inputs_embeds is required")
            x = self.token_embedding(input_ids)
        else:
            x = inputs_embeds
        seq_len = x.size(1)
        pos = torch.arange(seq_len, device=x.device)
        x = x + self.pos_embedding(pos).view(1, seq_len, -1)
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=1)
        hidden = self.blocks(x, mask=mask)
        hidden = self.ln_f(hidden)
        return {"logits": self.lm_head(hidden), "hidden_states": hidden}


def load_texts(path: Path) -> list[str]:
    texts = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            texts.append(str(row.get("text", "")))
    if not texts:
        raise ValueError(f"empty dataset: {path}")
    return texts


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    for field, value in asdict(TrainConfig()).items():
        arg_type = type(value)
        parser.add_argument(f"--{field}", type=arg_type, default=None)
    ns = parser.parse_args()
    data = {}
    if ns.config:
        config_path = ensure_within_project(ns.config)
        data.update(yaml.safe_load(config_path.read_text(encoding="utf-8")) or {})
    for key, value in vars(ns).items():
        if key != "config" and value is not None:
            data[key] = value
    cfg = TrainConfig(**data)
    validate_config(cfg)
    return cfg


def validate_config(cfg: TrainConfig) -> None:
    if cfg.method not in METHODS:
        raise ValueError(f"unknown method: {cfg.method}")
    if cfg.dry_run and cfg.max_steps > 20:
        raise ValueError("dry_run requires max_steps <= 20")
    if cfg.global_step_offset < 0:
        raise ValueError("global_step_offset must be non-negative")
    if cfg.dense_eval_interval < 0:
        raise ValueError("dense_eval_interval must be non-negative")
    if cfg.dense_eval_window < 0:
        raise ValueError("dense_eval_window must be non-negative")
    if cfg.dense_eval_anchor_step < 0:
        raise ValueError("dense_eval_anchor_step must be non-negative")
    if cfg.dense_eval_window > 0 and cfg.dense_eval_interval <= 0:
        raise ValueError("dense_eval_window requires dense_eval_interval > 0")
    if cfg.online_eval_max_batches <= 0:
        raise ValueError("online_eval_max_batches must be positive")
    if not 0.0 <= cfg.recovery_ratio <= 1.0:
        raise ValueError("recovery_ratio must be in [0, 1]")
    if cfg.tokenizer_backend not in {"char", "minimind"}:
        raise ValueError("tokenizer_backend must be char or minimind")
    if cfg.use_tokenized_cache and not cfg.tokenized_cache_path:
        raise ValueError("use_tokenized_cache requires tokenized_cache_path")
    if cfg.phase_override not in {"", "ntp", "superposition", "recovery"}:
        raise ValueError("phase_override must be empty, ntp, superposition, or recovery")
    if cfg.phase_override == "superposition" and cfg.method == "ntp_baseline":
        raise ValueError("phase_override=superposition requires a TST method")
    if cfg.order_alpha < 0.0:
        raise ValueError("order_alpha must be non-negative")
    if cfg.hier_alpha < 0.0:
        raise ValueError("hier_alpha must be non-negative")
    if cfg.use_swanlab:
        raise ValueError("swanlab integration is not implemented")


def tst_ratio(cfg: TrainConfig) -> float:
    if cfg.method == "ntp_baseline":
        return 0.0
    return 1.0 - cfg.recovery_ratio


def recovery_start_step(cfg: TrainConfig) -> int:
    if cfg.method == "ntp_baseline":
        return cfg.max_steps
    return int(cfg.max_steps * tst_ratio(cfg))


def phase_for_step(cfg: TrainConfig, step: int) -> str:
    if cfg.phase_override:
        return cfg.phase_override
    if cfg.method == "ntp_baseline":
        return "ntp"
    recovery_start = recovery_start_step(cfg)
    return "superposition" if step < recovery_start else "recovery"


def dense_eval_anchor_step(cfg: TrainConfig) -> int | None:
    if cfg.dense_eval_anchor_step > 0:
        return cfg.dense_eval_anchor_step
    if cfg.phase_override == "recovery":
        return cfg.global_step_offset + 1
    if cfg.phase_override == "superposition":
        return cfg.global_step_offset + cfg.max_steps + 1
    if cfg.method == "ntp_baseline":
        return None
    return recovery_start_step(cfg) + 1


def should_run_eval(cfg: TrainConfig, step: int, global_step: int) -> bool:
    if global_step % cfg.eval_interval == 0 or step == 0:
        return True
    anchor = dense_eval_anchor_step(cfg)
    if cfg.dense_eval_interval <= 0 or cfg.dense_eval_window <= 0 or anchor is None:
        return False
    if abs(global_step - anchor) > cfg.dense_eval_window:
        return False
    return abs(global_step - anchor) % cfg.dense_eval_interval == 0


def method_to_mode(cfg: TrainConfig) -> str:
    return {
        "vanilla_tst": "mean",
        "order_aware_tst": "order_aware",
        "boundary_aware_tst": "boundary_aware",
        "hierarchical_tst": "hierarchical",
        "residual_structured_tst": "residual_structured",
    }.get(cfg.method, cfg.superpose_mode)


def baseline_seq_len(cfg: TrainConfig) -> int:
    return cfg.baseline_seq_len or cfg.max_seq_len


def train_raw_seq_len(cfg: TrainConfig) -> int:
    if cfg.phase_override in {"ntp", "recovery"}:
        return baseline_seq_len(cfg)
    if cfg.paper_equal_flops and cfg.method != "ntp_baseline":
        return baseline_seq_len(cfg) * cfg.superpose_size
    return cfg.max_seq_len


def batch_for_phase(batch: torch.Tensor, cfg: TrainConfig, phase: str) -> torch.Tensor:
    if phase in {"ntp", "recovery"}:
        return batch[:, : baseline_seq_len(cfg)]
    return batch[:, : train_raw_seq_len(cfg)]


def token_counts(batch: torch.Tensor, cfg: TrainConfig, phase: str) -> tuple[int, int, int]:
    bsz, seq_len = batch.shape
    raw_tokens = bsz * seq_len
    if phase in {"ntp", "recovery"}:
        return raw_tokens, raw_tokens, raw_tokens
    usable_len = (seq_len // cfg.superpose_size) * cfg.superpose_size
    latent_tokens = bsz * max(0, (usable_len // cfg.superpose_size) - 1)
    return raw_tokens, latent_tokens, bsz * usable_len


def evaluate(model, composer, loader, cfg: TrainConfig, device: torch.device, phase: str, max_batches: int = 2) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            batch = batch_for_phase(batch.to(device), cfg, phase)
            losses.append(compute_loss(model, composer, batch, cfg, phase).detach())
    model.train()
    return float(torch.stack(losses).mean().cpu())


def compute_loss(model, composer, batch: torch.Tensor, cfg: TrainConfig, phase: str) -> torch.Tensor:
    if phase in {"ntp", "recovery"}:
        return ntp_loss(model(input_ids=batch)["logits"], batch)
    if composer is None:
        raise ValueError("superposition phase requires a composer")
    composed = composer.compose(batch)
    out = model(inputs_embeds=composed["inputs_embeds"])
    if cfg.loss_mode == "ordered_slot":
        return ordered_slot_loss(out["hidden_states"], composed["chunk_targets"], model.lm_head, composer.slot_embed)
    return repeated_token_ce_loss(out["logits"], composed["chunk_targets"])


def save_checkpoint(path: Path, model, optimizer, step: int, cfg: TrainConfig, local_step: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "local_step": local_step if local_step is not None else step,
            "global_step_offset": cfg.global_step_offset,
            "config": asdict(cfg),
        },
        path,
    )


def checkpoint_step(path: Path) -> int:
    try:
        return int(path.stem.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return -1


def load_latest_checkpoint(ckpt_dir: Path, model, optimizer) -> int:
    checkpoints = sorted(ckpt_dir.glob("step_*.pt"), key=checkpoint_step)
    if not checkpoints:
        return 0
    data = torch.load(checkpoints[-1], map_location="cpu")
    model.load_state_dict(data["model"])
    optimizer.load_state_dict(data["optimizer"])
    return int(data.get("local_step", data.get("step", 0)))


def load_model_checkpoint(path: Path, model) -> int:
    data = torch.load(path, map_location="cpu")
    model.load_state_dict(data["model"])
    return int(data.get("step", 0))


def init_wandb(cfg: TrainConfig, output_dir: Path):
    if not cfg.use_wandb:
        return None
    wandb_root = safe_mkdir(output_dir / "artifacts" / "wandb")
    os.environ.setdefault("WANDB_DIR", str(wandb_root))
    os.environ.setdefault("WANDB_CACHE_DIR", str(wandb_root / "cache"))
    os.environ.setdefault("WANDB_CONFIG_DIR", str(wandb_root / "config"))
    safe_mkdir(os.environ["WANDB_CACHE_DIR"])
    safe_mkdir(os.environ["WANDB_CONFIG_DIR"])
    try:
        import wandb
    except Exception as exc:
        raise RuntimeError("use_wandb=1 requires the wandb package in the active environment") from exc
    init_kwargs = {
        "project": cfg.wandb_project,
        "name": cfg.run_name,
        "dir": str(wandb_root),
        "config": asdict(cfg),
    }
    if cfg.wandb_entity:
        init_kwargs["entity"] = cfg.wandb_entity
    run = wandb.init(**init_kwargs)
    wandb.define_metric("step")
    for metric in (
        "loss_train",
        "loss_eval_ntp",
        "loss_eval_phase",
        "lr",
        "raw_tokens_seen",
        "effective_data_tokens_seen",
        "gpu_mem_gb",
    ):
        wandb.define_metric(metric, step_metric="step")
    run.summary.update(
        {
            "run_name": cfg.run_name,
            "method": cfg.method,
            "experiment_method": cfg.experiment_method or cfg.method,
            "superpose_size": cfg.superpose_size,
            "recovery_ratio": cfg.recovery_ratio,
            "tst_ratio": tst_ratio(cfg),
            "baseline_seq_len": baseline_seq_len(cfg),
            "paper_equal_flops": cfg.paper_equal_flops,
        }
    )
    return run


def build_wandb_record(record: dict) -> dict:
    keys = (
        "step",
        "loss_train",
        "loss_eval_ntp",
        "loss_eval_phase",
        "lr",
        "raw_tokens_seen",
        "effective_data_tokens_seen",
        "gpu_mem_gb",
    )
    return {key: record[key] for key in keys if record.get(key) is not None}


def main() -> None:
    cfg = parse_args()
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    data_path = ensure_within_project(cfg.data_path)
    output_dir = ensure_run_output_dir(cfg.output_dir)
    init_checkpoint_path = ensure_within_project(cfg.init_checkpoint_path) if cfg.init_checkpoint_path else None
    safe_mkdir(output_dir)
    ckpt_dir = safe_mkdir(output_dir / "checkpoints")
    safe_mkdir(output_dir / "outputs")
    safe_mkdir(output_dir / "artifacts")
    (output_dir / "config.yaml").write_text(yaml.safe_dump(asdict(cfg), sort_keys=True), encoding="utf-8")
    wandb_run = init_wandb(cfg, output_dir)

    dataset_seq_len = train_raw_seq_len(cfg)
    if cfg.use_tokenized_cache:
        dataset = TokenizedTensorDataset(ensure_within_project(cfg.tokenized_cache_path), dataset_seq_len)
        tokenizer = None
        vocab_size = dataset.vocab_size
        id_to_text = dataset.id_to_text
    else:
        texts = load_texts(data_path)
        if cfg.tokenizer_backend == "minimind":
            tokenizer = MiniMindTokenizer(ensure_within_project(cfg.tokenizer_path))
        else:
            tokenizer = CharTokenizer(texts)
        dataset = JsonlTextDataset(data_path, tokenizer, dataset_seq_len)
        vocab_size = tokenizer.vocab_size
        id_to_text = tokenizer.id_to_text
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    eval_loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, drop_last=True)
    device = torch.device(cfg.device)
    model = TinyCausalLM(vocab_size, baseline_seq_len(cfg), cfg.hidden_size, cfg.num_layers, cfg.num_heads, cfg.dropout).to(device)
    if not id_to_text:
        id_to_text = {i: "" for i in range(vocab_size)}
    composer = None
    if cfg.method != "ntp_baseline" and cfg.phase_override not in {"ntp", "recovery"}:
        token_types = torch.tensor(build_token_type_cache(id_to_text, output_dir / "token_type_cache.json"), dtype=torch.long, device=device)
        sp_cfg = SuperpositionConfig(
            mode=method_to_mode(cfg),
            superpose_size=cfg.superpose_size,
            hidden_size=cfg.hidden_size,
            vocab_size=vocab_size,
            slot_gate_type=cfg.slot_gate_type,
            type_vocab_size=cfg.type_vocab_size,
            block_mode=cfg.block_mode,
            chunks_per_block=cfg.chunks_per_block,
            order_alpha=cfg.order_alpha,
            hier_alpha=cfg.hier_alpha,
        )
        composer = SuperpositionComposer(model.token_embedding, cfg.hidden_size, vocab_size, sp_cfg, token_types).to(device)
    model_param_ids = {id(p) for p in model.parameters()}
    composer_params = [p for p in composer.parameters() if id(p) not in model_param_ids] if composer is not None else []
    optimizer = torch.optim.AdamW(list(model.parameters()) + composer_params, lr=cfg.learning_rate)
    if init_checkpoint_path is not None and not cfg.from_resume:
        load_model_checkpoint(init_checkpoint_path, model)
    start_step = load_latest_checkpoint(ckpt_dir, model, optimizer) if cfg.from_resume else 0
    metrics_path = output_dir / cfg.log_jsonl
    start_time = time.time()
    data_iter = iter(loader)
    raw_tokens_seen = start_step * cfg.batch_size * dataset_seq_len
    latent_tokens_seen = start_step * cfg.batch_size * baseline_seq_len(cfg)
    effective_data_tokens_seen = raw_tokens_seen

    print(yaml.safe_dump(asdict(cfg), sort_keys=True))
    for step in range(start_step, cfg.max_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)
        global_step = cfg.global_step_offset + step + 1
        phase = phase_for_step(cfg, step)
        batch = batch_for_phase(batch.to(device), cfg, phase)
        optimizer.zero_grad(set_to_none=True)
        loss = compute_loss(model, composer, batch, cfg, phase)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}: {loss.item()}")
        loss.backward()
        optimizer.step()

        loss_eval_ntp = None
        loss_eval_phase = None
        if should_run_eval(cfg, step, global_step):
            loss_eval_ntp = evaluate(model, composer, eval_loader, cfg, device, "ntp", cfg.online_eval_max_batches)
            loss_eval_phase = loss_eval_ntp if phase in {"ntp", "recovery"} else evaluate(model, composer, eval_loader, cfg, device, phase, cfg.online_eval_max_batches)
        if (step + 1) % cfg.save_interval == 0 or step + 1 == cfg.max_steps:
            save_checkpoint(ckpt_dir / f"step_{global_step}.pt", model, optimizer, global_step, cfg, local_step=step + 1)
        raw_step, latent_step, effective_step = token_counts(batch, cfg, phase)
        raw_tokens_seen += raw_step
        latent_tokens_seen += latent_step
        effective_data_tokens_seen += effective_step
        record = {
            "time": datetime.now(timezone.utc).isoformat(),
            "run_name": cfg.run_name,
            "method": cfg.method,
            "experiment_method": cfg.experiment_method or cfg.method,
            "step": global_step,
            "local_step": step + 1,
            "global_step_offset": cfg.global_step_offset,
            "phase": phase,
            "loss_train": float(loss.detach().cpu()),
            "loss_eval": loss_eval_ntp,
            "loss_eval_ntp": loss_eval_ntp,
            "loss_eval_phase": loss_eval_phase,
            "lr": optimizer.param_groups[0]["lr"],
            "tokens_seen": raw_tokens_seen,
            "raw_tokens_seen": raw_tokens_seen,
            "latent_tokens_seen": latent_tokens_seen,
            "effective_tokens_seen": effective_data_tokens_seen,
            "effective_data_tokens_seen": effective_data_tokens_seen,
            "superpose_size": cfg.superpose_size,
            "recovery_ratio": cfg.recovery_ratio,
            "tst_ratio": tst_ratio(cfg),
            "baseline_seq_len": baseline_seq_len(cfg),
            "raw_seq_len": batch.size(1),
            "paper_equal_flops": cfg.paper_equal_flops,
            "wall_time_sec": time.time() - start_time,
            "gpu_mem_gb": torch.cuda.max_memory_allocated(device) / (1024**3) if device.type == "cuda" else 0.0,
        }
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        if wandb_run is not None:
            wandb_run.log(build_wandb_record(record), step=global_step)
        print(json.dumps(record, ensure_ascii=False))

    shutil.copy2(metrics_path, output_dir / "outputs" / "metrics.jsonl")
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
