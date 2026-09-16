# ReasoningVQA

A configuration-driven training and evaluation pipeline for **multi-hop knowledge
reasoning VQA** with vision-language models. Source datasets (iNaturalist, GLDv2,
Visual Genome) are normalized into a single canonical manifest, then fed to a
shared LoRA SFT trainer and a deterministic evaluation/scoring stack.

## Highlights

- **One schema for all sources** — source adapters emit `reasoningvqa.canonical.v1`, so
  training and evaluation never branch on dataset names.
- **Config-driven CLI** — one entrypoint renders and runs `prepare`, `train`,
  `evaluate` (or `all`) from a YAML file.
- **Qwen-VL LoRA SFT** — Qwen2.5-VL / Qwen3-VL (and MiniCPM-V) BF16 MM-LoRA,
  response-only loss, language-only or full-family scope, optional multi-GPU
  `device_map="balanced"`.
- **Deterministic evaluation** — open-ended or structured generation, plus an
  independent re-scoring step that fails on any mismatch.
- **Score-only harness** — score precomputed predictions without loading a model.

## Repository Structure

```text
rvqa_bundle/
├── pipeline/                 # core pipeline code
│   ├── run_pipeline.py       # CLI entrypoint
│   ├── config_schema.py      # config validation
│   ├── configs/              # pipeline templates
│   ├── model_backends/       # model-family operation registry
│   ├── pipelines/            # prepare / train / evaluate stages
│   │   ├── prepare/          # adapters, canonical schema, composition
│   │   ├── train/            # LoRA SFT trainer
│   │   └── evaluate/         # generation, scoring, re-scoring
│   ├── tools/                # LoRA inventory build/verify
│   └── tests/                # contract tests
├── configs/                  # runbook configs per model
├── scoring/                  # score-only harness
├── scoring_extra/            # canonical MC metric source
├── tools/                    # data filtering utilities
├── requirements/             # environment pins (CUDA / Ascend)
└── gldv2/                    # GLDv2 subset notes and stats
```

## Installation

```bash
conda create -y -n rvqa python=3.10 && conda activate rvqa
pip install -r rvqa_bundle/requirements/requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu121
```

The LoRA target inventories are keyed to the exact `transformers` version, so
keep the pins. Ascend environments use `requirements/requirements-ascend*.txt`.

Build the LoRA inventory for your checkpoint (no weights are loaded):

```bash
python rvqa_bundle/pipeline/tools/build_model_inventory.py \
  --model  /path/to/Qwen2.5-VL-7B-Instruct \
  --output /path/to/inventory.json
```

## Data Preparation

Every source row is normalized to the canonical schema, validated, then merged
into train/dev manifests:

```bash
cd rvqa_bundle/pipeline
python run_pipeline.py --config /path/to/config.yaml prepare --dry-run
python run_pipeline.py --config /path/to/config.yaml prepare
```

| Adapter | Source |
| --- | --- |
| `inaturalist` | iNaturalist taxonomy + Wikidata |
| `gldv2` | GLDv2 landmarks |
| `visual_genome` | Visual Genome scene graphs |

The canonical manifest is the only contract read downstream
(`sample_id`, `image_path`, `sft_prompt`, `sft_target`, answer fields,
reasoning path, provenance).

## Training

```bash
python run_pipeline.py --config /path/to/config.yaml train
```

Key flags on `pipelines/train/train_qwen_vl_sft.py`:

- `--lora-scope {language,all}` — language-only or all LoRA families
- `--gpus 0,1` — shard a large checkpoint with `device_map="balanced"`
- `--init-adapter`, `--epochs`, `--max-rows` — resume / smoke runs

The trainer writes `adapter/`, `run_config.json`, `step_metrics.jsonl` and
`result.json`, and verifies parameter integrity before saving.

## Evaluation

```bash
python run_pipeline.py --config /path/to/config.yaml evaluate
```

- **Canonical** (`evaluate_canonical_reasoning.py`): `open` or `json` answer mode,
  with `correct-image`, `no-image` and `cyclic-shuffled-image` modes.
- **Structured** (`evaluate_structured_reasoning.py`): root / evidence / hop /
  answer rubric with rotation consistency.
- Both are followed by an **independent validator** that recomputes every score.

## Score-Only Harness

Score precomputed predictions (no inference):

```bash
python rvqa_bundle/scoring/score_predictions.py \
  --eval-jsonl /path/to/eval.jsonl \
  --predictions-jsonl /path/to/predictions.jsonl \
  --output-dir /path/to/out \
  --model-path /path/to/all-MiniLM-L6-v2
```

Reports `open_exact`, `open_substring`, `fuzzy_proxy`, `mc_accuracy`,
`semantic_accuracy`, `mean_cosine` and per-hop aggregates. If the canonical
metric modules are unavailable it falls back to embedded ports
(`RVQA_CANONICAL_OPEN` / `RVQA_CANONICAL_MC` can point at them).

## Configuration

The shipped configs under `pipeline/configs/` and `configs/` intentionally leave
machine-specific paths as empty strings. Fill in:

| Field | Meaning |
| --- | --- |
| `source.pipeline_root` | directory containing `prepare/`, `train/`, `evaluate/` |
| `output.experiment_root` | run output directory |
| `runtime.python` | training-environment interpreter |
| `model.checkpoint` | local model directory |
| `model.inventory` | LoRA inventory JSON |
| dataset `input` / `image_root` | source JSONL and image directory |

`--dry-run` prints the rendered commands; `--print-command` only renders.

## Tests

```bash
cd rvqa_bundle/pipeline
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v
```

## Citation

This repository builds on the [ReasonVQA](https://duong-tr.github.io/ReasonVQA/)
benchmark (ICCV 2025). Please cite the original work when using the data.

## License

Released under the license of the upstream ReasonVQA dataset (AGPL-3.0); see the
official dataset page for details.
