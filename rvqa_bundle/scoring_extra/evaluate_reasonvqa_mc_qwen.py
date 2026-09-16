from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable


DEFAULT_MODEL_PATH = Path("/home/ma-user/work/vlm/Qwen2.5-VL-7B-Instruct")
DEFAULT_VAL_JSONL = Path(
    "/home/ma-user/work/reasoningVQA/managed/40_llamafactory/data/val/"
    "val_2k_seed20260622.jsonl"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/home/ma-user/work/reasoningVQA/managed/50_training_results/eval_mc"
)


@dataclass(frozen=True)
class Score:
    exact: bool
    relaxed: bool
    parsed_choice: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen2.5-VL multiple-choice ReasonVQA answers on raw JSONL rows."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--vqa-jsonl", type=Path, default=DEFAULT_VAL_JSONL)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--min-visual-tokens", type=int, default=256)
    parser.add_argument("--max-visual-tokens", type=int, default=768)
    parser.add_argument("--device", choices=("auto", "npu", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--torch-dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
    )
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument(
        "--model-class",
        choices=(
            "auto",
            "qwen2_5_vl",
            "auto_image_text_to_text",
            "auto_vision2seq",
            "auto_causal_lm",
        ),
        default="auto",
    )
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--shuffle-choices", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--progress-every", type=int, default=25)
    return parser.parse_args()


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def normalize_text(value: Any) -> str:
    value = str(value or "").casefold().strip()
    value = value.replace("osnabrück", "osnabruck")
    value = re.sub(r"[\(\)\[\]\{\}\"'`]", " ", value)
    value = re.sub(r"[^a-z0-9]+", " ", value)
    value = re.sub(r"\b(the|a|an)\b", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def relaxed_match(prediction: str, answer: str) -> bool:
    pred_norm = normalize_text(prediction)
    answer_norm = normalize_text(answer)
    if not pred_norm or not answer_norm:
        return False
    if pred_norm == answer_norm:
        return True
    padded_pred = f" {pred_norm} "
    padded_answer = f" {answer_norm} "
    if padded_answer in padded_pred or padded_pred in padded_answer:
        return True
    return SequenceMatcher(None, pred_norm, answer_norm).ratio() >= 0.86


def correct_index(row: dict[str, Any]) -> int:
    correct = row.get("correct") or []
    if len(correct) != len(row.get("answers") or []):
        raise ValueError(f"Row has mismatched answers/correct: qid={row.get('question_id')}")
    matches = [index for index, value in enumerate(correct) if int(value) == 1]
    if len(matches) != 1:
        raise ValueError(f"Row has no unique correct answer: qid={row.get('question_id')}")
    return matches[0]


def correct_answer(row: dict[str, Any]) -> str:
    return str(row["answers"][correct_index(row)])


def choice_view(row: dict[str, Any], shuffle_choices: bool) -> dict[str, Any]:
    if not shuffle_choices:
        return row
    gold = correct_answer(row)
    negatives = [
        str(answer)
        for index, answer in enumerate(row["answers"])
        if index != correct_index(row)
    ]
    rng = random.Random(int(row["question_id"]) + 104729)
    rng.shuffle(negatives)
    gold_index = rng.randrange(len(row["answers"]))
    answers = negatives[:gold_index] + [gold] + negatives[gold_index:]
    correct = [1 if index == gold_index else 0 for index in range(len(answers))]
    view = dict(row)
    view["answers"] = answers
    view["correct"] = correct
    return view


def parse_mc_prediction(prediction: str, choices: list[str]) -> str | None:
    text = prediction.strip()
    upper = text.upper()
    letter_match = re.search(
        r"(?:^|\b)(?:ANSWER\s*(?:IS|:)?\s*)?([A-D])(?:\b|[\.\):]|$)",
        upper,
    )
    if letter_match:
        return letter_match.group(1)

    normalized_prediction = normalize_text(text)
    matched = [
        index
        for index, choice in enumerate(choices)
        if normalize_text(choice) and normalize_text(choice) in normalized_prediction
    ]
    if len(matched) == 1:
        return chr(65 + matched[0])
    return None


def extract_final_answer_text(prediction: str) -> str:
    text = str(prediction or "").strip()
    if not text:
        return ""
    answer_matches = re.findall(
        r"<answer>\s*(.*?)\s*</answer>",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if answer_matches:
        return answer_matches[-1].strip()
    final_matches = re.findall(
        r"(?:final\s+answer|answer)\s*[:：]\s*(.*)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if final_matches:
        return final_matches[-1].strip()
    text = re.sub(r"<think>.*?</think>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<reasoning>.*?</reasoning>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    return text.strip()


def score_mc_prediction(prediction: str, prompt_row: dict[str, Any]) -> Score:
    choices = [str(answer) for answer in prompt_row["answers"]]
    parsed = parse_mc_prediction(prediction, choices)
    gold_index = correct_index(prompt_row)
    gold_letter = chr(65 + gold_index)
    answer = correct_answer(prompt_row)
    exact = parsed == gold_letter
    return Score(
        exact=exact,
        relaxed=exact or relaxed_match(prediction, answer),
        parsed_choice=parsed,
    )


def build_prompt(row: dict[str, Any]) -> str:
    choices = "\n".join(
        f"{chr(65 + index)}. {answer}" for index, answer in enumerate(row["answers"])
    )
    return (
        "Answer the visual multiple-choice question. Use the image and any "
        "required factual knowledge. Choose one option.\n"
        f"Question: {row['question']}\n"
        f"Choices:\n{choices}\n"
        "Respond with only the option letter (A, B, C, or D)."
    )


def load_rows(path: Path, limit: int | None) -> list[dict[str, Any]]:
    rows = []
    for row in read_jsonl(path):
        if not row.get("image_path") or not row.get("answers") or not row.get("correct"):
            raise ValueError(f"Invalid MC row: qid={row.get('question_id')}")
        correct_index(row)
        rows.append(row)
    if limit is not None:
        rows = rows[:limit]
    return rows


def load_done(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    done: dict[int, dict[str, Any]] = {}
    for row in read_jsonl(path):
        done[int(row["question_id"])] = row
    return done


def iter_batches(items: list[dict[str, Any]], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def choose_device(kind: str) -> str:
    if kind == "cpu":
        return "cpu"
    if kind in {"auto", "npu"}:
        try:
            import torch  # noqa: F401
            import torch_npu  # noqa: F401

            if torch.npu.is_available():
                return "npu:0"
        except Exception:
            if kind == "npu":
                raise
    if kind in {"auto", "cuda"}:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
    if kind in {"npu", "cuda"}:
        raise RuntimeError(f"Requested device {kind!r} is unavailable")
    return "cpu"


def resolve_torch_dtype(dtype_name: str, device: str) -> Any:
    import torch

    if dtype_name == "float16" or (dtype_name == "auto" and device.startswith("npu")):
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float32":
        return torch.float32
    return "auto"


def move_inputs_to_device(inputs: Any, device: str) -> Any:
    if hasattr(inputs, "to"):
        return inputs.to(device)
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }


def generate_batch(
    model: Any,
    processor: Any,
    rows: list[dict[str, Any]],
    device: str,
    max_new_tokens: int,
    shuffle_choices: bool,
) -> list[dict[str, Any]]:
    import torch
    from PIL import Image

    texts = []
    images = []
    prompt_rows = []
    for row in rows:
        prompt_row = choice_view(row, shuffle_choices)
        prompt_rows.append(prompt_row)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": build_prompt(prompt_row)},
                ],
            }
        ]
        texts.append(
            processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )
        images.append(Image.open(row["image_path"]).convert("RGB"))

    try:
        inputs = processor(text=texts, images=images, padding=True, return_tensors="pt")
    finally:
        for image in images:
            image.close()

    inputs = move_inputs_to_device(inputs, device)
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id,
        )

    trimmed_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    predictions = processor.batch_decode(
        trimmed_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    results = []
    for row, prompt_row, prediction in zip(rows, prompt_rows, predictions):
        scored_prediction = extract_final_answer_text(prediction)
        answer = correct_answer(prompt_row)
        score = score_mc_prediction(scored_prediction, prompt_row)
        results.append(
            {
                "question_id": int(row["question_id"]),
                "image_id": row.get("image_id"),
                "image_path": row.get("image_path"),
                "question": row.get("question"),
                "answers": list(prompt_row["answers"]),
                "original_answers": list(row["answers"]),
                "choices_shuffled": list(prompt_row["answers"]) != list(row["answers"]),
                "gold_answer": answer,
                "gold_letter": chr(65 + correct_index(prompt_row)),
                "raw_prediction": prediction.strip(),
                "scored_prediction": scored_prediction,
                "parsed_choice": score.parsed_choice,
                "exact": int(score.exact),
                "relaxed": int(score.relaxed),
            }
        )
    return results


def accuracy(rows: list[dict[str, Any]], key: str) -> float:
    return float(sum(int(row[key]) for row in rows) / len(rows)) if rows else math.nan


def summarize(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(predictions),
        "mc_acc": accuracy(predictions, "exact"),
        "relaxed_acc": accuracy(predictions, "relaxed"),
        "parse_rate": float(
            sum(1 for row in predictions if row.get("parsed_choice")) / len(predictions)
        )
        if predictions
        else math.nan,
    }


def write_report(output_dir: Path, metrics: dict[str, Any], run_config: dict[str, Any]) -> None:
    lines = [
        "# ReasonVQA MC Evaluation",
        "",
        f"- Model: `{run_config['model']}`",
        f"- Adapter: `{run_config.get('adapter') or '-'}`",
        f"- VQA JSONL: `{run_config['vqa_jsonl']}`",
        f"- Samples: {metrics['n']}",
        f"- Shuffle choices: {run_config['shuffle_choices']}",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| MC accuracy | {metrics['mc_acc']:.4f} |",
        f"| Relaxed accuracy | {metrics['relaxed_acc']:.4f} |",
        f"| Parse rate | {metrics['parse_rate']:.4f} |",
    ]
    (output_dir / "metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_generation_class(args: argparse.Namespace) -> Any:
    import transformers

    if args.model_class == "qwen2_5_vl":
        return transformers.Qwen2_5_VLForConditionalGeneration

    if args.model_class == "auto":
        try:
            config = transformers.AutoConfig.from_pretrained(
                args.model,
                trust_remote_code=True,
                local_files_only=args.local_files_only,
            )
            if getattr(config, "model_type", "") == "qwen2_5_vl":
                return transformers.Qwen2_5_VLForConditionalGeneration
        except Exception:
            pass
        candidates = (
            "Qwen3VLForConditionalGeneration",
            "Qwen3_VLForConditionalGeneration",
            "AutoModelForImageTextToText",
            "AutoModelForVision2Seq",
            "AutoModelForCausalLM",
        )
    elif args.model_class == "auto_image_text_to_text":
        candidates = ("AutoModelForImageTextToText",)
    elif args.model_class == "auto_vision2seq":
        candidates = ("AutoModelForVision2Seq",)
    elif args.model_class == "auto_causal_lm":
        candidates = ("AutoModelForCausalLM",)
    else:
        raise ValueError(f"Unsupported model_class: {args.model_class}")

    for name in candidates:
        cls = getattr(transformers, name, None)
        if cls is not None:
            return cls
    raise RuntimeError(f"No usable model class found for model_class={args.model_class}")


def load_model_and_processor(args: argparse.Namespace, device: str) -> tuple[Any, Any]:
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
        min_pixels=args.min_visual_tokens * 28 * 28,
        max_pixels=args.max_visual_tokens * 28 * 28,
    )
    processor.tokenizer.padding_side = "left"

    model_kwargs: dict[str, Any] = {
        "torch_dtype": resolve_torch_dtype(args.torch_dtype, device),
        "trust_remote_code": True,
        "local_files_only": args.local_files_only,
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    model_cls = load_generation_class(args)
    model = model_cls.from_pretrained(args.model, **model_kwargs)
    if args.adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
    model.to(device)
    model.eval()
    return model, processor


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_tag = "adapter" if args.adapter else "baseline"
    output_dir = args.output_dir or (DEFAULT_OUTPUT_ROOT / f"{model_tag}_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "predictions.jsonl"
    metrics_path = output_dir / "metrics.json"

    rows = load_rows(args.vqa_jsonl, args.limit)
    done = load_done(prediction_path) if args.resume else {}
    work_items = [row for row in rows if int(row["question_id"]) not in done]

    device = choose_device(args.device)
    model, processor = load_model_and_processor(args, device)

    predictions = list(done.values())
    total = len(predictions) + len(work_items)
    started = time.time()
    for batch_index, batch in enumerate(iter_batches(work_items, args.batch_size), start=1):
        batch_results = generate_batch(
            model=model,
            processor=processor,
            rows=batch,
            device=device,
            max_new_tokens=args.max_new_tokens,
            shuffle_choices=args.shuffle_choices,
        )
        predictions.extend(batch_results)
        append_jsonl(prediction_path, batch_results)
        if len(predictions) % args.progress_every == 0 or len(predictions) >= total:
            print(
                json.dumps(
                    {
                        "completed": len(predictions),
                        "total": total,
                        "batch": batch_index,
                        "elapsed_sec": round(time.time() - started, 1),
                        "last_question_id": batch_results[-1]["question_id"],
                        "last_prediction": batch_results[-1]["raw_prediction"],
                        "last_parsed_choice": batch_results[-1]["parsed_choice"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    predictions = sorted(predictions, key=lambda row: int(row["question_id"]))
    write_jsonl(prediction_path, predictions)
    metrics = summarize(predictions)
    run_config = {
        "model": str(args.model),
        "adapter": str(args.adapter) if args.adapter else None,
        "vqa_jsonl": str(args.vqa_jsonl),
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "min_visual_tokens": args.min_visual_tokens,
        "max_visual_tokens": args.max_visual_tokens,
        "device": device,
        "torch_dtype": args.torch_dtype,
        "model_class": args.model_class,
        "limit": args.limit,
        "shuffle_choices": args.shuffle_choices,
    }
    metrics_path.write_text(
        json.dumps({"metrics": metrics, "run_config": run_config}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_report(output_dir, metrics, run_config)
    print(json.dumps({"output_dir": str(output_dir), "metrics": metrics}, ensure_ascii=False))


if __name__ == "__main__":
    main()
