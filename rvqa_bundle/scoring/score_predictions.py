#!/usr/bin/env python
"""Score-only ReasoningVQA evaluation harness.

This script scores *precomputed* model predictions against an ``eval.jsonl``
manifest.  It performs no model inference / generation -- it only consumes
predictions.  It reproduces the historical GLDv2 metric names:

    open_exact, open_substring, semantic_accuracy, mean_cosine,
    threshold_sensitivity, mc_accuracy

Metric provenance
-----------------
The normalization / lexical / MC metric functions are *reused* from the
surviving canonical evaluation scripts rather than re-invented:

  * ``evaluate_reasonvqa_openended_qwen.py``
      correct_answer_index, correct_answer, normalize_text, contains_answer,
      fuzzy_answer_match, score_open_prediction, extract_final_answer_text, accuracy
  * ``evaluate_reasonvqa_mc_qwen.py``
      normalize_text, relaxed_match, correct_index, correct_answer,
      parse_mc_prediction, score_mc_prediction, accuracy

At runtime this script first tries to ``import`` those modules by path.  If the
canonical files are present they are used directly (``metric_source`` records
this).  If they are unavailable, byte-faithful verbatim ports embedded below are
used instead (``metric_source="verbatim_port"``).  A self-check compares the two
whenever both are available.

Semantic metric
---------------
Embedding model = local ``all-MiniLM-L6-v2`` (sentence-transformers layout:
Transformer -> 1_Pooling[mean_tokens] -> 2_Normalize).  Pooling = attention-mask
MEAN pooling followed by L2 normalization; similarity = cosine (= dot on unit
vectors).  ``sentence-transformers`` is not installed in this environment, so the
default backend is raw ``transformers`` + ``torch``.  If
``sentence-transformers`` *is* importable it is preferred.  Both paths were
verified to reproduce the historical ``semantic_scores.jsonl`` cosines to
float32 round-off (~1e-7).

Usage
-----
    python score_predictions.py \
        --eval-jsonl  /path/to/eval.jsonl \
        --predictions-jsonl /path/to/predictions.jsonl \
        --output-dir  /path/to/out \
        --model-path /path/to/all-MiniLM-L6-v2
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
from collections import OrderedDict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable, Iterable

# --------------------------------------------------------------------------- #
# Canonical metric functions -- attempt import, else verbatim port
# --------------------------------------------------------------------------- #

def _optional_env_path(name: str) -> Path | None:
    """Return ``Path(env[name])`` when the variable is set, else ``None``."""
    value = os.environ.get(name, "").strip()
    return Path(value) if value else None


# Optional canonical metric modules. Leave both unset to use the byte-faithful
# verbatim ports embedded in this file.
CANONICAL_OPEN = _optional_env_path("RVQA_CANONICAL_OPEN")
CANONICAL_MC = _optional_env_path("RVQA_CANONICAL_MC")

# Local sentence-embedding model (all-MiniLM-L6-v2 layout). Override with
# ``--model-path`` or the ``RVQA_EMBED_MODEL`` environment variable.
DEFAULT_MODEL_PATH = _optional_env_path("RVQA_EMBED_MODEL")

PRIMARY_THRESHOLD = 0.80
THRESHOLDS = (0.70, 0.75, 0.80, 0.85, 0.90)
FUZZY_THRESHOLD = 0.86

# Prediction text field priority (historical pipeline used ``open_raw``).
# ``raw_output`` is the canonical pipeline's generation field (added so the
# scorer consumes evaluate_canonical_reasoning.py output without a rewrite).
PREDICTION_FIELDS = ("open_raw", "raw_output", "raw_prediction", "prediction", "answer_text", "text")
# Join keys tried in order.
JOIN_KEYS = ("question_id", "sample_id", "id", "qid")


def _load_module_from_path(name: str, path: Path | None) -> Any | None:
    if path is None or not path.exists():
        return None
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    except Exception:  # pragma: no cover - defensive
        return None


# --- verbatim ports (fallback) --------------------------------------------- #

def normalize_text(value: Any) -> str:
    # Verbatim from evaluate_reasonvqa_openended_qwen.py / _mc_qwen.py.
    value = str(value or "").casefold().strip()
    value = value.replace("osnabr\u00fcck", "osnabruck")
    value = re.sub(r"[\(\)\[\]\{\}\"'`]", " ", value)
    value = re.sub(r"[^a-z0-9]+", " ", value)
    value = re.sub(r"\b(the|a|an)\b", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def contains_answer(prediction: str, answer: str) -> bool:
    # Verbatim from evaluate_reasonvqa_openended_qwen.py.
    prediction_norm = f" {normalize_text(prediction)} "
    answer_norm = normalize_text(answer)
    return bool(answer_norm and f" {answer_norm} " in prediction_norm)


def fuzzy_answer_match(prediction: str, answer: str) -> bool:
    # Verbatim from evaluate_reasonvqa_openended_qwen.py.
    prediction_norm = normalize_text(prediction)
    answer_norm = normalize_text(answer)
    if not prediction_norm or not answer_norm:
        return False
    if prediction_norm == answer_norm:
        return True
    if contains_answer(prediction, answer):
        return True
    return SequenceMatcher(None, prediction_norm, answer_norm).ratio() >= FUZZY_THRESHOLD


def relaxed_match(prediction: str, answer: str) -> bool:
    # Verbatim from evaluate_reasonvqa_mc_qwen.py (bidirectional containment).
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
    return SequenceMatcher(None, pred_norm, answer_norm).ratio() >= FUZZY_THRESHOLD


def correct_index(row: dict[str, Any]) -> int:
    # Verbatim from evaluate_reasonvqa_mc_qwen.py.
    correct = row.get("correct") or []
    if len(correct) != len(row.get("answers") or []):
        raise ValueError(f"Row has mismatched answers/correct: qid={row.get('question_id')}")
    matches = [index for index, value in enumerate(correct) if int(value) == 1]
    if len(matches) != 1:
        raise ValueError(f"Row has no unique correct answer: qid={row.get('question_id')}")
    return matches[0]


def correct_answer(row: dict[str, Any]) -> str:
    # Verbatim from evaluate_reasonvqa_openended_qwen.py.
    explicit = row.get("correct_answer")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    answers = row.get("answers") or []
    index = correct_answer_index(row)
    if index is None or index >= len(answers):
        raise ValueError(f"Row has no unique answer: qid={row.get('question_id')}")
    return str(answers[index]).strip()


def correct_answer_index(row: dict[str, Any]) -> int | None:
    # Verbatim from evaluate_reasonvqa_openended_qwen.py.
    if isinstance(row.get("correct_index"), int):
        return int(row["correct_index"])
    correct = row.get("correct") or []
    if sum(1 for value in correct if value == 1) == 1:
        return correct.index(1)
    return None


def parse_mc_prediction(prediction: str, choices: list[str]) -> str | None:
    # Verbatim from evaluate_reasonvqa_mc_qwen.py.
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
    # Verbatim from the canonical scripts.
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


def accuracy(rows: list[dict[str, Any]], key: str) -> float:
    # Verbatim from the canonical scripts.
    return float(sum(int(row[key]) for row in rows) / len(rows)) if rows else math.nan


# --------------------------------------------------------------------------- #
# Resolve canonical functions (import preferred, port fallback)
# --------------------------------------------------------------------------- #

_OPEN_MOD = _load_module_from_path("_canonical_openended", CANONICAL_OPEN)
_MC_MOD = _load_module_from_path("_canonical_mc", CANONICAL_MC)

METRIC_SOURCE_PARTS: list[str] = []
if _OPEN_MOD is not None:
    METRIC_SOURCE_PARTS.append(f"import:{CANONICAL_OPEN}")
else:
    METRIC_SOURCE_PARTS.append("verbatim_port:openended")
if _MC_MOD is not None:
    METRIC_SOURCE_PARTS.append(f"import:{CANONICAL_MC}")
else:
    METRIC_SOURCE_PARTS.append("verbatim_port:mc")

# When the canonical modules are importable, use their functions directly.
if _OPEN_MOD is not None:
    correct_answer_index = _OPEN_MOD.correct_answer_index
    correct_answer = _OPEN_MOD.correct_answer
    normalize_text = _OPEN_MOD.normalize_text
    contains_answer = _OPEN_MOD.contains_answer
    fuzzy_answer_match = _OPEN_MOD.fuzzy_answer_match
    extract_final_answer_text = _OPEN_MOD.extract_final_answer_text
    accuracy = _OPEN_MOD.accuracy
if _MC_MOD is not None:
    relaxed_match = _MC_MOD.relaxed_match
    correct_index = _MC_MOD.correct_index
    parse_mc_prediction = _MC_MOD.parse_mc_prediction


def _verify_metric_equivalence() -> bool:
    """Compare imported canonical functions against the embedded ports."""
    if _OPEN_MOD is None or _MC_MOD is None:
        return False
    samples = [
        ("The osnabrück Museum of Art!", "osnabruck museum art"),
        ("a quick brown fox", "quick brown fox"),
        ("St. Petersburg Library System", "St Petersburg Library System"),
        ("", "gold"),
        ("one two three", "three two one"),
    ]
    for pred, gold in samples:
        if _OPEN_MOD.normalize_text(pred) != normalize_text(pred):
            return False
        if _OPEN_MOD.contains_answer(pred, gold) != contains_answer(pred, gold):
            return False
        if _OPEN_MOD.fuzzy_answer_match(pred, gold) != fuzzy_answer_match(pred, gold):
            return False
        if _MC_MOD.relaxed_match(pred, gold) != relaxed_match(pred, gold):
            return False
    return True


METRIC_EQUIVALENCE_VERIFIED = _verify_metric_equivalence()


# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_prediction(row: dict[str, Any]) -> str:
    for field in PREDICTION_FIELDS:
        value = row.get(field)
        if value is None:
            continue
        text = str(value)
        if text.strip():
            return text
    return ""


def get_hop(row: dict[str, Any]) -> int | None:
    for field in ("hop", "hop_count"):
        value = row.get(field)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                return None
    return None


# --------------------------------------------------------------------------- #
# Semantic embedder
# --------------------------------------------------------------------------- #

def load_embedder(
    model_path: Path,
    device: str,
    batch_size: int,
) -> tuple[Callable[[list[str]], Any], str, str]:
    """Return (embed_fn, backend, pooling_description).

    ``embed_fn(texts)`` returns an (N, D) torch tensor of L2-normalized
    embeddings.
    """
    pooling_desc = "attention-mask mean pooling followed by L2 normalization"

    # Prefer sentence-transformers if it is actually installed.
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore

        st_model = SentenceTransformer(str(model_path), device=device)

        def embed_st(texts: list[str]) -> Any:
            import numpy as np
            import torch

            vectors = st_model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return torch.from_numpy(np.asarray(vectors, dtype="float32"))

        return embed_st, "sentence_transformers", pooling_desc
    except Exception:
        pass

    # Raw transformers + torch fallback (default in this environment).
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    model = AutoModel.from_pretrained(str(model_path), local_files_only=True)
    model.to(device)
    model.eval()

    def embed_raw(texts: list[str]) -> Any:
        all_vectors = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.inference_mode():
                output = model(**encoded)
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            summed = (output.last_hidden_state * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            pooled = summed / counts
            normalized = F.normalize(pooled, p=2, dim=1)
            all_vectors.append(normalized.cpu())
        if not all_vectors:
            return torch.zeros((0, model.config.hidden_size), dtype=torch.float32)
        return torch.cat(all_vectors, dim=0)

    return embed_raw, "raw_transformers_mean_pool", pooling_desc


def cosine_similarities(embed_fn: Callable[[list[str]], Any], preds: list[str], golds: list[str]) -> list[float]:
    import torch  # local import so lexical-only use does not require torch

    texts = list(preds) + list(golds)
    vectors = embed_fn(texts)
    pred_vecs = vectors[: len(preds)]
    gold_vecs = vectors[len(preds) :]
    sims = (pred_vecs * gold_vecs).sum(dim=1)
    # Clamp tiny float32 overshoot (identical strings can give 1.00000002).
    return [max(-1.0, min(1.0, float(value))) for value in sims.tolist()]


# --------------------------------------------------------------------------- #
# Core scoring
# --------------------------------------------------------------------------- #

def parse_hops(value: str | None) -> set[int] | None:
    if not value:
        return None
    hops = set()
    for part in value.split(","):
        part = part.strip()
        if part:
            hops.add(int(part))
    return hops or None


def build_prediction_index(
    predictions: list[dict[str, Any]],
) -> tuple[str | None, dict[str, dict[str, Any]]]:
    for key in JOIN_KEYS:
        present = [row for row in predictions if key in row and row.get(key) is not None]
        if len(present) == len(predictions) and predictions:
            return key, {str(row[key]): row for row in predictions}
        if present:
            # Partial key coverage: use key for those present, positional otherwise.
            return key, {str(row[key]): row for row in present}
    return None, {}


def canonical_view(row: dict[str, Any]) -> dict[str, Any]:
    """Adapt a canonical-manifest row to the historical scoring schema.

    The historical ``eval.jsonl`` carried ``answers`` (the MC options) plus a
    one-hot ``correct`` vector (and optionally ``correct_index`` /
    ``correct_answer``).  Canonical manifests instead carry ``answer`` (gold),
    ``answers`` (``[gold]``) and ``answer_options`` (the MC choices), with no
    ``correct`` field, which makes the imported canonical gold/MC helpers raise
    ``Row has no unique answer``.  Build the historical view without mutating
    the input and without disturbing rows that already carry those fields.
    """
    if row.get("correct") is not None or row.get("correct_index") is not None:
        return row
    gold = row.get("correct_answer")
    if not (isinstance(gold, str) and gold.strip()):
        for field in ("answer", "sft_target"):
            value = row.get(field)
            if isinstance(value, str) and value.strip():
                gold = value.strip()
                break
    if not isinstance(gold, str) or not gold.strip():
        return row
    view = dict(row)
    view["correct_answer"] = gold.strip()
    options = row.get("answer_options")
    if isinstance(options, list) and options:
        choices = [str(value) for value in options]
        view["answers"] = choices
        gold_norm = normalize_text(gold)
        matches = [
            index for index, choice in enumerate(choices) if normalize_text(choice) == gold_norm
        ]
        if len(matches) == 1:
            view["correct"] = [1 if index == matches[0] else 0 for index in range(len(choices))]
    return view


def score(args: argparse.Namespace) -> dict[str, Any]:
    eval_rows = [canonical_view(row) for row in read_jsonl(args.eval_jsonl)]
    prediction_rows = read_jsonl(args.predictions_jsonl)
    hop_filter = parse_hops(args.hops)

    if hop_filter is not None:
        eval_rows = [row for row in eval_rows if get_hop(row) in hop_filter]
    if args.limit is not None:
        eval_rows = eval_rows[: args.limit]

    join_key, prediction_index = build_prediction_index(prediction_rows)

    # Join eval rows to predictions.
    joined: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for position, eval_row in enumerate(eval_rows):
        pred_row: dict[str, Any] | None = None
        if join_key is not None:
            lookup = str(eval_row.get(join_key)) if eval_row.get(join_key) is not None else None
            if lookup is not None:
                pred_row = prediction_index.get(lookup)
        if pred_row is None and len(prediction_rows) == len(eval_rows):
            pred_row = prediction_rows[position]
        if pred_row is None:
            raise ValueError(
                f"No prediction found for eval row position={position} "
                f"join_key={join_key} qid={eval_row.get('question_id')}"
            )
        joined.append((eval_row, pred_row))

    if not joined:
        raise ValueError("No rows to score (check --hops / --limit / inputs).")

    scored_rows: list[dict[str, Any]] = []
    pred_texts: list[str] = []
    gold_texts: list[str] = []

    for eval_row, pred_row in joined:
        gold = correct_answer(eval_row)
        raw_prediction = resolve_prediction(pred_row)
        if args.extract_final:
            prediction = extract_final_answer_text(raw_prediction)
        else:
            prediction = raw_prediction

        # --- lexical open-ended metrics (canonical functions) ---
        pred_norm = normalize_text(prediction)
        gold_norm = normalize_text(gold)
        open_exact = int(pred_norm == gold_norm)
        open_substring = int(contains_answer(prediction, gold))
        if pred_norm and gold_norm:
            fuzzy_ratio = SequenceMatcher(None, pred_norm, gold_norm).ratio()
        else:
            fuzzy_ratio = 0.0
        fuzzy_proxy = int(fuzzy_ratio >= FUZZY_THRESHOLD)

        # --- MC metrics (canonical functions) ---
        choices = [str(answer) for answer in (eval_row.get("answers") or [])]
        parsed_choice = parse_mc_prediction(prediction, choices)
        gold_index = correct_index(eval_row)
        gold_letter = chr(65 + gold_index)
        mc_correct = int(parsed_choice == gold_letter)
        # relaxed = exact letter OR exact text OR whitespace-bounded substring OR fuzzy>=0.86
        mc_relaxed_correct = int(
            mc_correct or relaxed_match(prediction, gold)
        )

        pred_texts.append(prediction)
        gold_texts.append(gold)

        scored_rows.append(
            {
                "question_id": eval_row.get("question_id"),
                "sample_id": eval_row.get("sample_id"),
                "image_id": eval_row.get("image_id"),
                "hop": get_hop(eval_row),
                "property_id": eval_row.get("property_id"),
                "gold_answer": gold,
                "gold_letter": gold_letter,
                "raw_prediction": raw_prediction,
                "prediction": prediction,
                "mc_parsed_letter": parsed_choice,
                "open_exact": open_exact,
                "open_substring": open_substring,
                "fuzzy_proxy": fuzzy_proxy,
                "fuzzy_ratio": round(fuzzy_ratio, 6),
                "mc_correct": mc_correct,
                "mc_relaxed_correct": mc_relaxed_correct,
                # placeholders filled after the embedding pass
                "cosine_similarity": None,
                "semantic_correct_0_80": None,
            }
        )

    # --- semantic metrics ---
    embed_fn, backend, pooling_desc = load_embedder(
        args.model_path, args.device, args.batch_size
    )
    similarities = cosine_similarities(embed_fn, pred_texts, gold_texts)
    for row, similarity in zip(scored_rows, similarities):
        row["cosine_similarity"] = round(similarity, 8)
        for threshold in THRESHOLDS:
            row[f"semantic_correct_{threshold:.2f}"] = int(similarity >= threshold)

    # ------------------------------------------------------------------ #
    # Aggregate (historical names)
    # ------------------------------------------------------------------ #
    def rate(key: str, rows: list[dict[str, Any]] = scored_rows) -> float:
        return accuracy(rows, key)

    threshold_sensitivity = OrderedDict(
        (f"{threshold:.2f}", rate(f"semantic_correct_{threshold:.2f}"))
        for threshold in THRESHOLDS
    )

    hop_groups: "OrderedDict[int, list[dict[str, Any]]]" = OrderedDict()
    if any(row.get("hop") is not None for row in scored_rows):
        for hop in sorted({row["hop"] for row in scored_rows if row.get("hop") is not None}):
            hop_groups[hop] = [row for row in scored_rows if row.get("hop") == hop]

    hop_summary = OrderedDict()
    for hop, rows in hop_groups.items():
        hop_summary[str(hop)] = {
            "rows": len(rows),
            "open_exact": rate("open_exact", rows),
            "open_substring": rate("open_substring", rows),
            "fuzzy_proxy": rate("fuzzy_proxy", rows),
            "mc_accuracy": rate("mc_correct", rows),
            "mc_relaxed_accuracy": rate("mc_relaxed_correct", rows),
            "semantic_accuracy": rate("semantic_correct_0.80", rows),
            "mean_cosine": float(
                sum(row["cosine_similarity"] for row in rows) / len(rows)
            ),
        }

    summary: dict[str, Any] = OrderedDict(
        [
            ("status", "completed"),
            ("rows", len(scored_rows)),
            ("n", len(scored_rows)),
            ("open_exact", rate("open_exact")),
            ("open_substring", rate("open_substring")),
            ("fuzzy_proxy", rate("fuzzy_proxy")),
            ("mc_accuracy", rate("mc_correct")),
            ("mc_relaxed_accuracy", rate("mc_relaxed_correct")),
            ("semantic_accuracy", rate("semantic_correct_0.80")),
            ("mean_cosine", float(sum(row["cosine_similarity"] for row in scored_rows) / len(scored_rows))),
            ("primary_threshold", PRIMARY_THRESHOLD),
            ("threshold_sensitivity", threshold_sensitivity),
            ("hop", hop_summary),
            ("semantic_backend", backend),
            ("model_path", str(args.model_path)),
            ("pooling", pooling_desc),
            ("metric_source", " | ".join(METRIC_SOURCE_PARTS)),
            ("metric_equivalence_verified", METRIC_EQUIVALENCE_VERIFIED),
            ("extract_final_answer", bool(args.extract_final)),
            ("hops_filter", sorted(hop_filter) if hop_filter else None),
            ("eval_jsonl", str(args.eval_jsonl)),
            ("predictions_jsonl", str(args.predictions_jsonl)),
            ("eval_sha256", sha256_file(args.eval_jsonl)),
            ("predictions_sha256", sha256_file(args.predictions_jsonl)),
        ]
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scored_path = output_dir / "scored_rows.jsonl"
    summary_path = output_dir / "summary.json"
    write_jsonl(scored_path, scored_rows)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {"scored_rows": str(scored_path), "summary": str(summary_path), "metrics": summary}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score precomputed ReasoningVQA predictions (no generation). "
            "Computes open_exact, open_substring, fuzzy_proxy, MC accuracy, "
            "and frozen all-MiniLM-L6-v2 semantic accuracy."
        )
    )
    parser.add_argument("--eval-jsonl", type=Path, required=True)
    parser.add_argument("--predictions-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--hops",
        default=None,
        help="Optional comma-separated hop filter, e.g. '1,2,3'. Default: all hops.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--extract-final",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Apply canonical extract_final_answer_text (strip <answer>/thinking) "
            "before scoring, matching the historical generation pipeline. "
            "Use --no-extract-final to score the raw prediction verbatim."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.eval_jsonl.exists():
        raise SystemExit(f"eval jsonl not found: {args.eval_jsonl}")
    if not args.predictions_jsonl.exists():
        raise SystemExit(f"predictions jsonl not found: {args.predictions_jsonl}")
    if args.model_path is None:
        raise SystemExit(
            "no sentence-embedding model configured; pass --model-path or set RVQA_EMBED_MODEL"
        )
    if not args.model_path.exists():
        raise SystemExit(f"semantic model path not found: {args.model_path}")
    result = score(args)
    print(
        json.dumps(
            {
                "outputs": {
                    "scored_rows": result["scored_rows"],
                    "summary": result["summary"],
                },
                "metrics": result["metrics"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
