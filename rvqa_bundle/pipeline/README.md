# ReasoningVQA Public Pipeline

Self-contained, configuration-driven implementation of the active ReasoningVQA workflow.

```bash
# Fill in the empty paths in the config first.
python run_pipeline.py --config configs/example_pipeline.yaml prepare --dry-run
python run_pipeline.py --config configs/example_pipeline.yaml train --dry-run
python run_pipeline.py --config configs/example_pipeline.yaml evaluate --dry-run
python run_pipeline.py --config configs/example_pipeline.yaml all --dry-run
```

Responsibilities:

- `run_pipeline.py`: config loading, action selection, command rendering, config freeze.
- `pipelines/prepare/source_adapters/`: source-specific iNaturalist, GLDv2, and
  Visual Genome normalization.
- `pipelines/prepare/canonical_schema.py`: the single manifest contract read by
  shared training and evaluation.
- `pipelines/prepare/`: canonical adaptation, validation, deterministic
  composition, and the retained GLDv2 curriculum transform.
- `pipelines/train/`: supervised fine-tuning implementation.
- `pipelines/evaluate/`: inference and prediction validation.
- `model_backends/`: validated model-family operation mappings, defaults, and registry.
- `configs/`: paths, model, runtime, ordered steps, and outputs.
- `tests/`: reusable interface and command-expansion contracts.

The current pipeline rejects GRPO entrypoints. Active commands resolve only to scripts
inside this public Pipeline; no historical experiment implementation path is required.

Model-specific YAML steps select a backend operation such as `structured_sft` or
`structured_generation`; the selected backend owns the concrete implementation path.
Generic prepare and independent-validation steps may still name scripts directly.

`configs/example_pipeline.yaml` preserves the current frozen GLDv2 structured run.
`configs/pipeline.template.yaml` is the portable three-source starting point. Its
`datasets.sources` section selects an adapter per input; changing a source path or
dataset family does not change training or evaluation code.
Each source may set `image_root`; the adapter then relocates `image_name` under that
root explicitly. Without it, the recorded source path is preserved. Image existence
is enforced only when `check_images: true` is selected for that source or manifest.

The public data boundary is:

```text
iNaturalist source -> inaturalist adapter --\
GLDv2 source ------> gldv2 adapter --------+-> reasoningvqa.canonical.v1
Visual Genome -----> visual_genome adapter-/             |
                                                          v
                                   validate -> compose -> SFT -> evaluate
```

Adapters preserve source-specific taxonomy, landmark attribution, and scene-graph
provenance, while the public downstream reads only `sample_id`, `image_path`,
`sft_prompt`, `sft_target`, answer/path fields, and the canonical contract. The
portable template creates and validates each source manifest before composing train
and dev outputs. Artifact row counts are derived from manifests, never filenames.

Reusable tests:

```bash
cd rvqa_bundle/pipeline
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v
```

The public source tree is versioned in Git; experiment outputs remain outside this code
repository.
