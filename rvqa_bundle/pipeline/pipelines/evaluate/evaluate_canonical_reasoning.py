#!/usr/bin/env python3
"""Source-independent generation and deterministic scoring for canonical rows."""

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path

from PIL import Image

from evaluate_structured_reasoning import Generator, extract_json, sha256, text_generate


ANSWER_MODES = ("open", "json")
DEFAULT_ANSWER_MODE = "open"


def normalized(value):
    return " ".join(re.sub(r"[^\w\s]", " ", str(value or "").casefold()).split())


def parsed_answer(parsed, raw):
    if isinstance(parsed, dict):
        return str(parsed.get("answer", "")).strip()
    return str(raw or "").strip()


def parsed_path(parsed):
    value = parsed.get("reasoning_path") if isinstance(parsed, dict) else []
    return value if isinstance(value, list) else []


def score(row, raw, answer_mode=DEFAULT_ANSWER_MODE):
    """Score one generation.

    ``open`` treats the raw generated text as the answer and only reports
    ``answer_correct`` (= open_exact). ``json`` keeps the legacy
    ``{answer, reasoning_path}`` parsing and the schema/path/full-chain rubric.
    """
    accepted = {normalized(value) for value in row["answers"]}
    if answer_mode == "open":
        answer = str(raw or "").strip()
        answer_correct = int(bool(normalized(answer)) and normalized(answer) in accepted)
        return {
            "parsed_output": None,
            "predicted_answer": answer,
            "answer_correct": answer_correct,
        }
    parsed = extract_json(raw)
    answer = parsed_answer(parsed, raw)
    answer_correct = int(bool(normalized(answer)) and normalized(answer) in accepted)
    path = parsed_path(parsed)
    path_exact = int(path == row["reasoning_path"])
    schema_valid = int(
        isinstance(parsed, dict)
        and set(parsed) == {"answer", "reasoning_path"}
        and isinstance(parsed.get("answer"), str)
        and isinstance(parsed.get("reasoning_path"), list)
    )
    return {
        "parsed_output": parsed,
        "predicted_answer": answer,
        "predicted_reasoning_path": path,
        "schema_valid": schema_valid,
        "answer_correct": answer_correct,
        "path_exact": path_exact,
        "full_chain": int(schema_valid and answer_correct and path_exact),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("correct-image", "no-image", "cyclic-shuffled-image"),
        default="correct-image",
    )
    parser.add_argument(
        "--answer-mode",
        choices=ANSWER_MODES,
        default=DEFAULT_ANSWER_MODE,
        help="open (default) scores the raw generated text as the answer; json "
        "parses the legacy {answer, reasoning_path} object",
    )
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = [
        json.loads(line)
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.max_rows is not None:
        rows = rows[: args.max_rows]
    if not rows or len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("empty or duplicate canonical manifest")
    args.output.mkdir(parents=True)
    contract = {
        "manifest": str(args.manifest),
        "manifest_sha256": sha256(args.manifest),
        "model": str(args.model),
        "adapter": str(args.adapter) if args.adapter else None,
        "mode": args.mode,
        "answer_mode": args.answer_mode,
        "rows": len(rows),
        "max_new_tokens": args.max_new_tokens,
        "schema_version": "reasoningvqa.canonical.v1",
        "scoring_protocol": (
            "canonical-open-v1" if args.answer_mode == "open" else "canonical-answer-path-v1"
        ),
    }
    (args.output / "contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    generator = Generator(str(args.model), str(args.adapter) if args.adapter else "")
    results = []
    started = time.time()
    with (args.output / "predictions.jsonl").open("x", encoding="utf-8", newline="\n") as handle:
        for index, row in enumerate(rows):
            used_image_id = None
            if args.mode == "no-image":
                raw = text_generate(generator, row["sft_prompt"], args.max_new_tokens)
            else:
                source = row if args.mode == "correct-image" else rows[(index + 1) % len(rows)]
                used_image_id = source["image_id"]
                with Image.open(source["image_path"]) as image:
                    raw = generator.generate(image.convert("RGB"), row["sft_prompt"], args.max_new_tokens)
            result = {
                "sample_id": row["sample_id"],
                "source_dataset": row["source_dataset"],
                "task_type": row["task_type"],
                "mode": args.mode,
                "answer_mode": args.answer_mode,
                "used_image_id": used_image_id,
                "raw_output": raw,
                "gold_answer": row["answer"],
                "gold_reasoning_path": row.get("reasoning_path", []),
                **score(row, raw, answer_mode=args.answer_mode),
            }
            results.append(result)
            handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            progress = {
                "row": index + 1,
                "total": len(rows),
                "answer": result["answer_correct"],
            }
            if args.answer_mode == "json":
                progress.update({
                    "path": result["path_exact"],
                    "full": result["full_chain"],
                })
            print(json.dumps(progress, sort_keys=True), flush=True)
    per_source = Counter()
    correct_source = Counter()
    for result in results:
        per_source[result["source_dataset"]] += 1
        correct_source[result["source_dataset"]] += result["answer_correct"]
    summary = {
        "status": "passed",
        "rows": len(results),
        "mode": args.mode,
        "answer_mode": args.answer_mode,
        "answer_correct": sum(item["answer_correct"] for item in results),
        "source_rows": dict(sorted(per_source.items())),
        "source_answer_correct": dict(sorted(correct_source.items())),
        "elapsed_seconds": time.time() - started,
    }
    if args.answer_mode == "json":
        summary.update({
            "schema_valid": sum(item["schema_valid"] for item in results),
            "path_exact": sum(item["path_exact"] for item in results),
            "full_chain": sum(item["full_chain"] for item in results),
        })
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
