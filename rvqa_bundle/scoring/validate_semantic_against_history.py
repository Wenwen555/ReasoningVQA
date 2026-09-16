#!/usr/bin/env python
"""Validate the reimplemented semantic metric against the surviving historical
``semantic_scores.jsonl`` produced by the (now deleted) original scorer.

It recomputes cosine(pred_embedding, gold_embedding) for every row of the
historical ``base-formal-qwen25vl7b-v0.1`` run and compares it against the
stored ``cosine_similarity`` values.

Run with a python that has torch + transformers, e.g.:

    /data/wenjt/conda_envs/videoespresso-common/bin/python \
        validate_semantic_against_history.py
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import score_predictions as sp

RUN = Path(
    "/data/wenjt/projects/ReasoningVQA/experiments/"
    "RVQA-REASONVQA-GLDV2-PAPER-REIMPL-20260802/runs/base-formal-qwen25vl7b-v0.1"
)
MANIFEST = Path(
    "/data/wenjt/projects/ReasoningVQA/experiments/"
    "RVQA-REASONVQA-GLDV2-PAPER-REIMPL-20260802/data/formal-v0.1/test.jsonl"
)
OUT = Path(__file__).with_name("semantic_history_validation.json")


def main() -> None:
    manifest = {json.loads(line)["sample_id"]: json.loads(line) for line in MANIFEST.open()}
    predictions = {json.loads(line)["sample_id"]: json.loads(line) for line in (RUN / "predictions.jsonl").open()}
    historical = {json.loads(line)["sample_id"]: json.loads(line) for line in (RUN / "semantic_scores.jsonl").open()}

    sids = [sid for sid in historical if sid in manifest and sid in predictions]
    preds = [str(predictions[sid].get("open_raw") or "") for sid in sids]
    golds = [str(manifest[sid].get("answer_text") or "") for sid in sids]

    embed_fn, backend, pooling = sp.load_embedder(sp.DEFAULT_MODEL_PATH, "cpu", 64)
    sims = sp.cosine_similarities(embed_fn, preds, golds)

    diffs = []
    for sid, similarity in zip(sids, sims):
        diffs.append(abs(float(historical[sid]["cosine_similarity"]) - similarity))

    result = {
        "status": "completed",
        "rows_compared": len(sids),
        "semantic_backend": backend,
        "pooling": pooling,
        "model_path": str(sp.DEFAULT_MODEL_PATH),
        "mean_abs_diff": statistics.fmean(diffs) if diffs else None,
        "max_abs_diff": max(diffs) if diffs else None,
        "historical_scores": str(RUN / "semantic_scores.jsonl"),
        "manifest": str(MANIFEST),
        # 8-decimal rounding in the historical JSONL bounds achievable agreement.
        "note": "historical cosine_similarity is stored rounded to 8 decimals",
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
