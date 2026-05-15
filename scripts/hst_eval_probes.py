#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.hst_path_safety import ensure_within_project, safe_mkdir


ORDER_PROBES = [
    ("AB", "BA"),
    ("北京 到 上海", "上海 到 北京"),
    ("x = y + 1", "y = x + 1"),
    ("if a: b()", "b(): if a"),
]
BOUNDARY_PROBES = [
    ("function call(args)", "function call args"),
    ("中文，标点影响语义。", "中文 标点 影响 语义"),
    ("line1\nline2", "line1 line2"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="./hst_outputs/probe_results.jsonl")
    parser.add_argument("--run_name", default="untrained_probe_template")
    args = parser.parse_args()
    output = ensure_within_project(args.output)
    safe_mkdir(output.parent)
    with output.open("w", encoding="utf-8") as f:
        for name, probes in (("order", ORDER_PROBES), ("boundary", BOUNDARY_PROBES)):
            for a, b in probes:
                f.write(json.dumps({"run_name": args.run_name, "probe_type": name, "text_a": a, "text_b": b, "loss_a": None, "loss_b": None}, ensure_ascii=False) + "\n")
    print(f"wrote probe template to {output}")


if __name__ == "__main__":
    main()
