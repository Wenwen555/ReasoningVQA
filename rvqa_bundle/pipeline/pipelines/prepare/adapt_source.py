#!/usr/bin/env python3
"""Convert one source-specific JSONL into a create-only canonical manifest."""

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipelines.prepare.canonical_schema import (
    ANSWER_MODES,
    DEFAULT_ANSWER_MODE,
    canonical_json,
)
from pipelines.prepare.source_adapters import get_adapter


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--split")
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--allow-mixed-source", action="store_true")
    parser.add_argument(
        "--answer-mode",
        choices=ANSWER_MODES,
        default=DEFAULT_ANSWER_MODE,
        help="open (default) emits a plain answer target; json keeps the legacy "
        "{answer, reasoning_path} target and multiple-choice prompt",
    )
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.output.exists() or args.summary.exists():
        raise FileExistsError(args.output if args.output.exists() else args.summary)
    if args.limit is not None and args.limit < 1:
        raise ValueError("limit must be positive")
    adapter = get_adapter(args.adapter)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    selected = skipped = source_rows = 0
    samples = set()
    split_counts = Counter()
    with args.input.open("r", encoding="utf-8") as source, args.output.open(
        "x", encoding="utf-8", newline="\n"
    ) as destination:
        for line_number, raw in enumerate(source, 1):
            if not raw.strip():
                continue
            source_rows += 1
            row = json.loads(raw)
            if not isinstance(row, dict):
                raise ValueError(f"non-object source row at line {line_number}")
            if not adapter.matches(row):
                if args.allow_mixed_source:
                    skipped += 1
                    continue
                raise ValueError(
                    f"source row {line_number} does not match adapter {args.adapter!r}; "
                    "use --allow-mixed-source only for an intentionally mixed input"
                )
            if args.image_root is not None:
                relative = row.get("image_name")
                image = row.get("image") if isinstance(row.get("image"), dict) else {}
                relative = relative or image.get("file_name")
                if not str(relative or "").strip():
                    relative = Path(str(row.get("image_path") or image.get("image_path") or "")).name
                if not str(relative or "").strip():
                    raise ValueError(f"cannot relocate image path at source line {line_number}")
                row = dict(row)
                row["image_path"] = str(args.image_root.joinpath(*Path(str(relative)).parts))
            value = adapter.adapt(
                row, split=args.split, answer_mode=args.answer_mode
            )
            if value["sample_id"] in samples:
                raise ValueError(f"duplicate canonical sample_id: {value['sample_id']}")
            samples.add(value["sample_id"])
            split_counts[value["split"]] += 1
            destination.write(canonical_json(value) + "\n")
            selected += 1
            if args.limit is not None and selected >= args.limit:
                break
    if selected == 0:
        args.output.unlink(missing_ok=True)
        raise ValueError(f"adapter {args.adapter!r} selected no rows")
    report = {
        "status": "passed",
        "adapter": args.adapter,
        "answer_mode": args.answer_mode,
        "input": str(args.input),
        "input_sha256": sha256(args.input),
        "output": str(args.output),
        "output_sha256": sha256(args.output),
        "source_rows_read": source_rows,
        "selected_rows": selected,
        "skipped_rows": skipped,
        "split_counts": dict(sorted(split_counts.items())),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with args.summary.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
