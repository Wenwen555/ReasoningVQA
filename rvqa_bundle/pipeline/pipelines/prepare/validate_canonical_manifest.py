#!/usr/bin/env python3
"""Independent structural validation for canonical ReasoningVQA manifests."""

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipelines.prepare.canonical_schema import ANSWER_MODES, DEFAULT_ANSWER_MODE, validate_row


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-source", action="append", default=[])
    parser.add_argument("--expected-split", action="append", default=[])
    parser.add_argument("--expected-rows", type=int)
    parser.add_argument(
        "--answer-mode",
        choices=ANSWER_MODES,
        default=DEFAULT_ANSWER_MODE,
        help="open (default) expects a plain answer target; json expects the "
        "legacy {answer, reasoning_path} target",
    )
    parser.add_argument("--check-images", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    errors = []
    sample_ids = set()
    source_counts = Counter()
    split_counts = Counter()
    task_counts = Counter()
    rows = 0
    with args.manifest.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            rows += 1
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                errors.append(f"line:{line_number}:json:{exc}")
                continue
            if not isinstance(row, dict):
                errors.append(f"line:{line_number}:non_object")
                continue
            row_errors = validate_row(
                row, check_image=args.check_images, answer_mode=args.answer_mode
            )
            errors.extend(f"line:{line_number}:{value}" for value in row_errors)
            identifier = row.get("sample_id")
            if identifier in sample_ids:
                errors.append(f"line:{line_number}:duplicate_sample_id:{identifier}")
            sample_ids.add(identifier)
            source_counts[str(row.get("source_dataset"))] += 1
            split_counts[str(row.get("split"))] += 1
            task_counts[str(row.get("task_type"))] += 1
    if rows == 0:
        errors.append("empty_manifest")
    if args.expected_rows is not None and rows != args.expected_rows:
        errors.append(f"rows:{rows}!={args.expected_rows}")
    if args.expected_source and set(source_counts) != set(args.expected_source):
        errors.append(f"sources:{sorted(source_counts)}!={sorted(args.expected_source)}")
    if args.expected_split and set(split_counts) != set(args.expected_split):
        errors.append(f"splits:{sorted(split_counts)}!={sorted(args.expected_split)}")
    report = {
        "status": "passed" if not errors else "failed",
        "manifest": str(args.manifest),
        "manifest_sha256": sha256(args.manifest),
        "answer_mode": args.answer_mode,
        "rows": rows,
        "source_counts": dict(sorted(source_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "task_counts": dict(sorted(task_counts.items())),
        "check_images": args.check_images,
        "errors": errors[:500],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    raise SystemExit(0 if not errors else 2)


if __name__ == "__main__":
    main()
