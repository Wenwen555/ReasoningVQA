#!/usr/bin/env python3
"""Independent re-score of canonical ReasoningVQA predictions."""

import argparse
import hashlib
import json
import re
from pathlib import Path


ANSWER_MODES = ("open", "json")
DEFAULT_ANSWER_MODE = "open"


def normalized(value):
    return " ".join(re.sub(r"[^\w\s]", " ", str(value or "").casefold()).split())


def parse(value):
    text = str(value or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument(
        "--answer-mode",
        choices=ANSWER_MODES,
        default=DEFAULT_ANSWER_MODE,
        help="open (default) re-scores the raw output as the answer; json re-scores "
        "the legacy {answer, reasoning_path} object",
    )
    parser.add_argument("--expected-rows", type=int)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = [
        json.loads(line)
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    predictions = [
        json.loads(line)
        for line in args.predictions.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected = len(manifest) if args.expected_rows is None else args.expected_rows
    errors = []
    if len(manifest) != expected or len(predictions) != expected:
        errors.append(f"rows:{len(manifest)}/{len(predictions)}/{expected}")
    if len({row.get("sample_id") for row in manifest}) != len(manifest):
        errors.append("manifest_duplicate_ids")
    by_id = {row["sample_id"]: row for row in manifest}
    seen = set()
    if args.answer_mode == "open":
        totals = {"answer_correct": 0}
    else:
        totals = {"schema_valid": 0, "answer_correct": 0, "path_exact": 0, "full_chain": 0}
    for prediction in predictions:
        identifier = prediction.get("sample_id")
        if identifier in seen:
            errors.append(f"duplicate_prediction:{identifier}")
        seen.add(identifier)
        row = by_id.get(identifier)
        if row is None:
            errors.append(f"unknown_prediction:{identifier}")
            continue
        if prediction.get("source_dataset") != row["source_dataset"]:
            errors.append(f"source:{identifier}")
        if prediction.get("mode") != args.mode:
            errors.append(f"mode:{identifier}")
        if args.mode == "correct-image" and prediction.get("used_image_id") != row["image_id"]:
            errors.append(f"image:{identifier}")
        if args.mode == "no-image" and prediction.get("used_image_id") is not None:
            errors.append(f"no_image:{identifier}")
        accepted = {normalized(value) for value in row["answers"]}
        if args.answer_mode == "open":
            # The raw generated text IS the answer; no JSON parsing is required.
            answer = prediction.get("raw_output", "")
            calculated = {
                "answer_correct": int(
                    bool(normalized(answer)) and normalized(answer) in accepted
                )
            }
        else:
            parsed = parse(prediction.get("raw_output"))
            schema = int(
                isinstance(parsed, dict)
                and set(parsed) == {"answer", "reasoning_path"}
                and isinstance(parsed.get("answer"), str)
                and isinstance(parsed.get("reasoning_path"), list)
            )
            answer = parsed.get("answer", "") if isinstance(parsed, dict) else prediction.get("raw_output", "")
            answer_correct = int(bool(normalized(answer)) and normalized(answer) in accepted)
            path = parsed.get("reasoning_path", []) if isinstance(parsed, dict) else []
            path_exact = int(path == row["reasoning_path"])
            calculated = {
                "schema_valid": schema,
                "answer_correct": answer_correct,
                "path_exact": path_exact,
                "full_chain": int(schema and answer_correct and path_exact),
            }
        for key, value in calculated.items():
            totals[key] += value
            if prediction.get(key) != value:
                errors.append(f"score:{identifier}:{key}")
    if seen != set(by_id):
        errors.append("prediction_coverage")
    report = {
        "status": "passed" if not errors else "failed",
        "rows": len(predictions),
        "mode": args.mode,
        "answer_mode": args.answer_mode,
        **totals,
        "predictions_sha256": sha256(args.predictions),
        "errors": errors[:500],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    raise SystemExit(0 if not errors else 2)


if __name__ == "__main__":
    main()
