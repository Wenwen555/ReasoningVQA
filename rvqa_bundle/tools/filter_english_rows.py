#!/usr/bin/env python3
"""Filter non-English and degenerate rows out of ReasoningVQA jsonl subsets.

A row is DROPPED when either:
  1. DEgenerate data: ``answers`` is empty, ``correct`` is empty, or ``correct``
     does not contain exactly one ``1`` (and only 0/1 values otherwise); or
  2. NON-ENGLISH: the gold answer (``answers[correct.index(1)]``) normalizes to
     the empty string under the canonical scorer normalizer.

The normalizer is copied VERBATIM from ``scoring/score_predictions.py``
(``normalize_text``, itself ported verbatim from the canonical open-ended /
MC scorers).

Usage:
    python3 filter_english_rows.py \
        --input  /path/to/train.jsonl \
        --output /path/to/train.english.jsonl \
        --report /path/to/train.english.report.json \
        [--examples 8]

The script never writes to the input path. Pass ``--protect-write-prefix`` (or
set ``RVQA_PROTECTED_WRITE_PREFIX``) to refuse writes under a read-only tree.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any


# --- canonical normalizer (VERBATIM from scoring/score_predictions.py) ------ #
def normalize_text(value: Any) -> str:
    # Verbatim from evaluate_reasonvqa_openended_qwen.py / _mc_qwen.py.
    value = str(value or "").casefold().strip()
    value = value.replace("osnabr\u00fcck", "osnabruck")
    value = re.sub(r"[\(\)\[\]\{\}\"'`]", " ", value)
    value = re.sub(r"[^a-z0-9]+", " ", value)
    value = re.sub(r"\b(the|a|an)\b", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def classify_row(row: dict[str, Any]) -> tuple[str, str, str]:
    """Return (decision, reason, gold_raw).

    decision is "keep" | "drop".
    reason is "" for keep, else "degenerate_*" or "empty_normalized_gold".
    gold_raw is the un-normalized gold answer ("" when unavailable/degenerate).
    """
    answers = row.get("answers")
    correct = row.get("correct")

    # --- degenerate checks ------------------------------------------------- #
    if not isinstance(answers, list) or len(answers) == 0:
        return "drop", "degenerate_answers_empty", ""

    if not isinstance(correct, list) or len(correct) == 0:
        return "drop", "degenerate_correct_empty", ""

    if any((not isinstance(x, int)) or x not in (0, 1) for x in correct):
        return "drop", "degenerate_correct_not_one_hot", ""

    if sum(correct) != 1:
        # Covers both zero 1s and more than one 1.
        return "drop", "degenerate_correct_not_exactly_one_1", ""

    if len(answers) != len(correct):
        return "drop", "degenerate_answers_correct_length_mismatch", ""

    # --- gold + non-English classification --------------------------------- #
    gold_raw = answers[correct.index(1)]
    if normalize_text(gold_raw) == "":
        return "drop", "empty_normalized_gold", str(gold_raw)

    return "keep", "", str(gold_raw)


def run(input_path: Path, output_path: Path, report_path: Path, n_examples: int) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    rows_before = 0
    kept = 0
    dropped_degenerate = 0
    dropped_non_english = 0
    reason_counts: dict[str, int] = {}
    dropped: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []

    with input_path.open("r", encoding="utf-8") as fin, output_path.open(
        "w", encoding="utf-8"
    ) as fout:
        for lineno, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            rows_before += 1
            row = json.loads(line)
            decision, reason, gold_raw = classify_row(row)

            if decision == "keep":
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                kept += 1
                continue

            reason_counts[reason] = reason_counts.get(reason, 0) + 1
            if reason == "empty_normalized_gold":
                dropped_non_english += 1
            else:
                dropped_degenerate += 1

            qid = row.get("question_id")
            dropped.append({"question_id": qid, "reason": reason, "line": lineno})

            if reason == "empty_normalized_gold" and len(examples) < n_examples:
                examples.append(
                    {
                        "question_id": qid,
                        "raw_gold": gold_raw,
                        "normalized_gold": normalize_text(gold_raw),
                    }
                )

    report = {
        "input": str(input_path),
        "output": str(output_path),
        "rows_before": rows_before,
        "rows_after": kept,
        "rows_dropped": rows_before - kept,
        "reason_breakdown": {
            "empty_normalized_gold": dropped_non_english,
            "degenerate_total": dropped_degenerate,
            "by_reason": reason_counts,
        },
        "dropped_non_english": dropped_non_english,
        "dropped_degenerate": dropped_degenerate,
        "dropped_ids": dropped,
        "examples": examples,
    }

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, type=Path, help="source jsonl (read-only)")
    ap.add_argument("--output", required=True, type=Path, help="filtered jsonl output")
    ap.add_argument("--report", required=True, type=Path, help="JSON report path")
    ap.add_argument("--examples", type=int, default=8, help="examples to embed (default 8)")
    ap.add_argument(
        "--protect-write-prefix",
        default=os.environ.get("RVQA_PROTECTED_WRITE_PREFIX", ""),
        help="refuse to write under this path prefix (optional safety guard)",
    )
    args = ap.parse_args()

    # Optional safety guard against writing into a read-only tree.
    if args.protect_write_prefix:
        protected = Path(args.protect_write_prefix).resolve()
        for p in (args.output, args.report):
            if protected in p.resolve().parents or p.resolve() == protected:
                raise SystemExit(f"refusing to write under protected prefix: {p}")

    report = run(args.input, args.output, args.report, args.examples)

    print(f"input : {report['input']}")
    print(f"output: {report['output']}")
    print(
        f"before={report['rows_before']} after={report['rows_after']} "
        f"dropped={report['rows_dropped']} "
        f"(empty_normalized_gold={report['dropped_non_english']}, "
        f"degenerate={report['dropped_degenerate']})"
    )
    for ex in report["examples"]:
        print(
            f"  dropped qid={ex['question_id']!r} raw={ex['raw_gold']!r} "
            f"normalized={ex['normalized_gold']!r}"
        )


if __name__ == "__main__":
    main()
