#!/usr/bin/env python3
"""Create-only deterministic composition of validated canonical manifests."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.summary.exists():
        raise FileExistsError(args.output if args.output.exists() else args.summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    source_counts = Counter()
    split_counts = Counter()
    task_counts = Counter()
    rows = 0
    with args.output.open("x", encoding="utf-8", newline="\n") as destination:
        for path in args.input:
            with path.open("r", encoding="utf-8") as source:
                for line_number, raw in enumerate(source, 1):
                    if not raw.strip():
                        continue
                    row = json.loads(raw)
                    identifier = str(row.get("sample_id", ""))
                    if not identifier:
                        raise ValueError(f"missing sample_id: {path}:{line_number}")
                    if identifier in seen:
                        raise ValueError(f"duplicate sample_id across manifests: {identifier}")
                    seen.add(identifier)
                    destination.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
                    source_counts[str(row.get("source_dataset"))] += 1
                    split_counts[str(row.get("split"))] += 1
                    task_counts[str(row.get("task_type"))] += 1
                    rows += 1
    if rows == 0:
        args.output.unlink(missing_ok=True)
        raise ValueError("cannot compose an empty manifest")
    report = {
        "status": "passed",
        "inputs": [str(path) for path in args.input],
        "input_sha256": {str(path): sha256(path) for path in args.input},
        "output": str(args.output),
        "output_sha256": sha256(args.output),
        "rows": rows,
        "source_counts": dict(sorted(source_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "task_counts": dict(sorted(task_counts.items())),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with args.summary.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
