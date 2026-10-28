"""Create a deterministic evenly sampled manifest for chunked evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num_samples", type=int, required=True)
    args = parser.parse_args()
    records = json.loads(args.input.read_text(encoding="utf-8"))
    if not 1 <= args.num_samples <= len(records):
        raise ValueError("num_samples must be in [1, len(input manifest)]")
    if args.num_samples == 1:
        indices = [0]
    else:
        indices = [
            round(index * (len(records) - 1) / (args.num_samples - 1))
            for index in range(args.num_samples)
        ]
    selected = [records[index] for index in indices]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(selected, indent=2), encoding="utf-8")
    print(f"selected {len(selected)} / {len(records)} records")


if __name__ == "__main__":
    main()
