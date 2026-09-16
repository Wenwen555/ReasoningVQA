# Run configs

Per-model runbook configs for `pipeline/run_pipeline.py`.

All configs ship with **empty machine-specific paths** (pipeline root, experiment
root, python, checkpoint, inventory, dataset inputs). Fill them in before
running; `--dry-run` prints the rendered commands.

Run every command from the pipeline code dir:

```bash
cd rvqa_bundle/pipeline
python run_pipeline.py --config /path/to/configs/<config>.yaml prepare
python run_pipeline.py --config /path/to/configs/<config>.yaml train
python run_pipeline.py --config /path/to/configs/<config>.yaml evaluate
```

All runs use `--lora-scope language` (language-only LoRA; visual/merger are not
trained). Use `--gpus <id,...>` on the trainer for model-parallel runs.

| Config | Checkpoint | Notes |
| --- | --- | --- |
| `vg-qwen25vl7b.yaml` | Qwen2.5-VL-7B-Instruct | single GPU |
| `inat-qwen25vl7b.yaml` | Qwen2.5-VL-7B-Instruct | single GPU |
| `vg-qwen25vl32b.yaml` | Qwen2.5-VL-32B-Instruct | needs model parallelism (`--gpus`) |
| `vg-qwen3vl32b-awq.yaml` | Qwen3-VL-32B-Instruct-AWQ | needs model parallelism + recent transformers |
| `vg-minicpmv26.yaml` | MiniCPM-V-2_6 | string-content chat path (auto-detected) |

The pipeline ships portable templates in `pipeline/configs/`:
`pipeline.template.yaml` (three-source canonical flow) and
`example_pipeline.yaml` (structured GLDv2 flow).
