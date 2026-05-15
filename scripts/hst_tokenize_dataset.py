#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from utils.hst_path_safety import ensure_within_project, safe_mkdir


def load_tokenizer(tokenizer_path: Path):
    try:
        from tokenizers import Tokenizer
    except Exception as exc:
        raise RuntimeError("hst_tokenize_dataset.py requires the tokenizers package") from exc
    path = tokenizer_path / "tokenizer.json" if tokenizer_path.is_dir() else tokenizer_path
    return Tokenizer.from_file(str(path))


def first_token_id(tokenizer, candidates: list[str], default: int) -> int:
    for token in candidates:
        token_id = tokenizer.token_to_id(token)
        if token_id is not None:
            return int(token_id)
    return default


def count_rows(path: Path, limit: int | None) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
                if limit is not None and count >= limit:
                    break
    return count


def iter_token_ids(path: Path, tokenizer, eos_id: int | None, limit: int | None):
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            text = str(json.loads(line).get("text", ""))
            ids = tokenizer.encode(text).ids
            if eos_id is not None:
                ids.append(int(eos_id))
            yield ids
            count += 1
            if limit is not None and count >= limit:
                break


def count_packed_blocks(path: Path, tokenizer, eos_id: int | None, seq_len: int, limit: int | None) -> int:
    buffered = 0
    blocks = 0
    for ids in iter_token_ids(path, tokenizer, eos_id, limit):
        buffered += len(ids)
        if buffered >= seq_len:
            new_blocks = buffered // seq_len
            blocks += new_blocks
            buffered -= new_blocks * seq_len
    return blocks


def write_packed_cache(input_path: Path, tokenizer, eos_id: int | None, seq_len: int, limit: int | None, dtype: torch.dtype) -> torch.Tensor:
    num_blocks = count_packed_blocks(input_path, tokenizer, eos_id, seq_len, limit)
    if num_blocks == 0:
        raise ValueError("not enough tokens to create a packed cache block")
    input_ids = torch.empty((num_blocks, seq_len), dtype=dtype)
    buffer: list[int] = []
    block_idx = 0
    for ids in iter_token_ids(input_path, tokenizer, eos_id, limit):
        buffer.extend(ids)
        while len(buffer) >= seq_len:
            input_ids[block_idx] = torch.tensor(buffer[:seq_len], dtype=dtype)
            del buffer[:seq_len]
            block_idx += 1
            if block_idx % 10000 == 0:
                print(f"packed {block_idx}/{num_blocks}", flush=True)
    return input_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokenizer_path", default="./tokenizer/minimind_tokenizer")
    parser.add_argument("--seq_len", type=int, required=True)
    parser.add_argument("--num_examples", type=int)
    parser.add_argument("--dtype", choices=["int32", "int64"], default="int32")
    parser.add_argument("--pack_documents", type=int, default=0)
    args = parser.parse_args()

    input_path = ensure_within_project(args.input)
    output_path = ensure_within_project(args.output)
    tokenizer = load_tokenizer(ensure_within_project(args.tokenizer_path))
    pad_id = first_token_id(tokenizer, ["<pad>", "[PAD]", "<unk>", "<eos>"], 0)
    eos_id = tokenizer.token_to_id("<eos>")
    dtype = torch.int32 if args.dtype == "int32" else torch.int64
    if args.pack_documents:
        input_ids = write_packed_cache(input_path, tokenizer, eos_id, args.seq_len, args.num_examples, dtype)
        num_rows = input_ids.size(0)
    else:
        num_rows = count_rows(input_path, args.num_examples)
        input_ids = torch.full((num_rows, args.seq_len), pad_id, dtype=dtype)
        row_idx = 0
        for ids in iter_token_ids(input_path, tokenizer, eos_id, args.num_examples):
            ids = ids[: args.seq_len]
            if ids:
                input_ids[row_idx, : len(ids)] = torch.tensor(ids, dtype=dtype)
            row_idx += 1
            if row_idx % 10000 == 0:
                print(f"encoded {row_idx}/{num_rows}", flush=True)

    id_to_text = [tokenizer.id_to_token(i) or "" for i in range(tokenizer.get_vocab_size())]
    safe_mkdir(output_path.parent)
    torch.save(
        {
            "input_ids": input_ids,
            "vocab_size": tokenizer.get_vocab_size(),
            "pad_token_id": pad_id,
            "seq_len": args.seq_len,
            "id_to_text": id_to_text,
        },
        output_path,
    )
    print(f"wrote {num_rows} rows x {args.seq_len} tokens to {output_path}")


if __name__ == "__main__":
    main()
