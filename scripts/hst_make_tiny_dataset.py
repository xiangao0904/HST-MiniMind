#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.hst_path_safety import safe_mkdir, ensure_within_project


SAMPLES = [
    "北京到上海的高铁今天准点到达。",
    "结构感知的 token superposition 需要保留局部顺序。",
    "The quick brown fox jumps over 13 lazy dogs.",
    "Loss(t) = alpha * x + beta / 3.14.",
    "def add(a, b):\n    return a + b\n",
    "if user.is_admin:\n    grant_access(user)\nelse:\n    deny(user)\n",
    "标点，真的，会影响语义！不是吗？",
    "function call(args) differs from function call args.",
    "A B C D is not the same as D C B A.",
    "x = y + 1; y = x + 1; print(x, y)",
    "line1\nline2\nline3",
    "[INFO] step=42 loss=3.1415 lr=0.0003",
]


def build_example(i: int, rng: random.Random) -> dict[str, str]:
    left = SAMPLES[i % len(SAMPLES)]
    right = SAMPLES[rng.randrange(len(SAMPLES))]
    return {"text": f"{left}\n{right}"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="./hst_tmp/tiny_pretrain.jsonl")
    parser.add_argument("--num_examples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output = ensure_within_project(args.output)
    safe_mkdir(output.parent)
    rng = random.Random(args.seed)

    with output.open("w", encoding="utf-8") as f:
        for i in range(args.num_examples):
            f.write(json.dumps(build_example(i, rng), ensure_ascii=False) + "\n")

    print(f"wrote {args.num_examples} examples to {output}")


if __name__ == "__main__":
    main()
