#!/usr/bin/env python
"""Validate the reimplemented semantic metric against a historical
``semantic_scores.jsonl`` produced by the original scorer.

It recomputes cosine(pred_embedding, gold_embedding) for every row of a
historical run and compares it against the stored ``cosine_similarity`` values.

Run with a python that has torch + transformers::

    RVQA_HISTORY_RUN=/path/to/run \
    RVQA_HISTORY_MANIFEST=/path/to/test.jsonl \
    RVQA_EMBED_MODEL=/path/to/all-MiniLM-L6-v2 \
    python validate_semantic_against_history.py

``RVQA_HISTORY_RUN`` must contain ``predictions.jsonl`` and
``semantic_scores.jsonl``. ``OUT`` defaults to the JSON file next to this script.
"""

from __future__ import annotations

import json
import os
import statistics
from pathlib import Path

import score_predictions as sp

OUT = Path(__file__).with_name("semantic_history_validation.json")


def main() -> None:
    run_value = os.environ.get("RVQA_HISTORY_RUN", "").strip()
    manifest_value = os.environ.get("RVQA_HISTORY_MANIFEST", "").strip()
    if not run_value or not manifest_value:
        raise SystemExit("set RVQA_HISTORY_RUN and RVQA_HISTORY_MANIFEST")
    run = Path(run_value)
    manifest_path = Path(manifest_value)

    manifest = {
        json.loads(line)["sample_id"]: json.loads(line)
        for line in manifest_path.open()
    }
    predictions = {
        json.loads(line)["sample_id"]: json.loads(line)
        for line in (run / "predictions.jsonl").open()
    }
    historical = {
        json.loads(line)["sample_id"]: json.loads(line)
        for line in (run / "semantic_scores.jsonl").open()
    }

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
        "historical_scores": str(run / "semantic_scores.jsonl"),
        "manifest": str(manifest_path),
        # 8-decimal rounding in the historical JSONL bounds achievable agreement.
        "note": "historical cosine_similarity is stored rounded to 8 decimals",
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
