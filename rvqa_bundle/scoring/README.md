# ReasoningVQA score-only harness

`score_predictions.py` scores **precomputed** predictions against an `eval.jsonl`
manifest. It performs no generation/inference — it only reads predictions.

## CLI

```bash
python score_predictions.py \
  --eval-jsonl  <eval.jsonl> \
  --predictions-jsonl <predictions.jsonl> \
  --output-dir  <out-dir> \
  [--hops 1,2,3] [--limit N] \
  [--model-path <all-MiniLM-L6-v2>] [--device cpu] [--batch-size 64] \
  [--extract-final | --no-extract-final]
```

* `--eval-jsonl` / `--predictions-jsonl` / `--output-dir` — required.
* `--hops` — optional comma-separated hop filter (`hop` or `hop_count` field).
* `--extract-final` (default on) — apply the canonical
  `extract_final_answer_text` (strip `<think>`, `<answer>` tags) before scoring,
  matching the historical generation pipeline. `--no-extract-final` scores the
  raw prediction verbatim.
* Predictions may be keyed by `question_id` / `sample_id` / `id` / `qid`, or be
  positional (same length). Predicted text is read from the first non-empty of
  `open_raw`, `raw_prediction`, `prediction`, `answer_text`, `text`.
* Gold answer is `answers[correct.index(1)]` (canonical `correct_answer`).

## Outputs

* `<out-dir>/scored_rows.jsonl` — one row per eval row.
* `<out-dir>/summary.json` — aggregate metrics (historical names).

### Aggregate metric keys

| Key | Meaning |
| --- | --- |
| `open_exact` | normalized exact match, `normalize(pred) == normalize(gold)` |
| `open_substring` | whitespace-token-bounded containment: `" gold " in " pred "` |
| `fuzzy_proxy` | lexical proxy only: `SequenceMatcher(normalize(pred), normalize(gold)).ratio() >= 0.86`. **Not** the embedding metric. |
| `mc_accuracy` | parsed option letter equals gold letter |
| `mc_relaxed_accuracy` | `mc_accuracy` OR exact-text OR substring OR fuzzy (canonical `relaxed_match`) |
| `semantic_accuracy` | fraction with cosine >= `primary_threshold` (0.80) |
| `mean_cosine` | mean of per-row cosine similarities |
| `primary_threshold` | 0.80 |
| `threshold_sensitivity` | `{"0.70",...,"0.90"}` fractions at each threshold |
| `hop` | per-hop `{rows, open_exact, open_substring, fuzzy_proxy, mc_accuracy, mc_relaxed_accuracy, semantic_accuracy, mean_cosine}` |
| `semantic_backend` | `raw_transformers_mean_pool` or `sentence_transformers` |
| `pooling` | `attention-mask mean pooling followed by L2 normalization` |
| `metric_source` | whether canonical scripts were imported or verbatim ports used |
| `metric_equivalence_verified` | imported canonical vs embedded port self-check |

### Per-row keys

`question_id, sample_id, image_id, hop, property_id, gold_answer, gold_letter,
raw_prediction, prediction, mc_parsed_letter, open_exact, open_substring,
fuzzy_proxy, fuzzy_ratio, mc_correct, mc_relaxed_correct, cosine_similarity,
semantic_correct_0.70 … semantic_correct_0.90`.

## Metric provenance

Normalization / lexical / MC functions are **imported at runtime** from the
surviving canonical scripts (recorded in `summary.json:metric_source`):

* `.../evaluate_reasonvqa_openended_qwen.py`
* `.../evaluate_reasonvqa_mc_qwen.py`

If those files are unavailable, byte-faithful verbatim ports embedded in
`score_predictions.py` are used. `metric_equivalence_verified` is `true` when
both are present and agree.

## Semantic backend

`sentence-transformers` is **not installed** in any conda env here, so the
default backend is raw `transformers` + `torch`: `AutoModel` + attention-mask
mean pooling + `F.normalize(p=2)`, cosine = dot product. If
`sentence-transformers` becomes importable it is used automatically.

Validated against the surviving historical `semantic_scores.jsonl`
(`base-formal-qwen25vl7b-v0.1`, 894 rows):
`validate_semantic_against_history.py` reports mean abs diff `3.7e-09`,
max abs diff `1.5e-07` (float32 round-off; historical values are stored to 8 dp).
See `semantic_history_validation.json`.

## Exact commands for the new eval sets

```bash
PY=/data/wenjt/conda_envs/videoespresso-common/bin/python

# iNaturalist
$PY score_predictions.py \
  --eval-jsonl /data/wenjt/reasoningvqa_subsets/inaturalist/eval.jsonl \
  --predictions-jsonl /path/to/inat_predictions.jsonl \
  --output-dir /data/wenjt/reasoningvqa_bench/scoring/out_inaturalist

# Visual Genome
$PY score_predictions.py \
  --eval-jsonl /data/wenjt/reasoningvqa_subsets/visual_genome/eval.jsonl \
  --predictions-jsonl /path/to/vg_predictions.jsonl \
  --output-dir /data/wenjt/reasoningvqa_bench/scoring/out_visual_genome
```

## Self-test artifacts in this directory

* `synthetic_predictions.jsonl` — 5 rows exercising exact / substring / fuzzy /
  semantic-only / wrong paths. Run with `--limit 5`.
* `oracle_predictions.*.jsonl` — gold-answer predictions (full eval sets),
  self-test only.
* `synthetic_run/`, `oracle_run_*` — outputs of the above.
* `validate_semantic_against_history.py` + `semantic_history_validation.json`.

### Known caveat discovered via the oracle self-test

The canonical normalization is ASCII-only, so non-ASCII gold answers
(e.g. Syloti Nagri, Japanese) normalize to the empty string. On the oracle run
this caps `open_substring`/`mc_accuracy` below 1.0:

* iNat: 32/1014 gold answers normalize empty
* VG: 26/1000 gold answers normalize empty

`open_exact` and `semantic_accuracy` are unaffected. This is expected canonical
behaviour, not a bug in the harness, but it will lower `open_substring` and
`mc_accuracy` on these new sets relative to the Latin-only GLDv2 set.
