# Run configs / runbook

Run every command from the pipeline code dir:

```bash
cd /data/wenjt/reasoningvqa_bench/pipeline
```

Configs live in `/data/wenjt/reasoningvqa_bench/configs/`. All runs use
`--lora-scope language` (language-only LoRA; visual/merger are never trained) and
must set `CUDA_VISIBLE_DEVICES` to a GPU we own (currently only GPU 8 is free).

## 1. Visual Genome + Qwen2.5-VL-7B (launched)

```bash
cd /data/wenjt/reasoningvqa_bench/pipeline
PY=/data/wenjt/conda_envs/videoespresso-common/bin/python
$PY run_pipeline.py --config /data/wenjt/reasoningvqa_bench/configs/vg-qwen25vl7b.yaml prepare
# smoke test first (tiny run, separate output dir):
CUDA_VISIBLE_DEVICES=8 TRANSFORMERS_OFFLINE=1 $PY pipeline/pipelines/train/train_qwen_vl_sft.py \
  --model /data/llm_models/Qwen2.5-VL-7B-Instruct \
  --manifest /data/wenjt/reasoningvqa_bench/runs/vg-qwen25vl7b/data/manifests/train.jsonl \
  --inventory /data/wenjt/reasoningvqa_bench/inventories/qwen25vl7b-instruct.json \
  --output /data/wenjt/reasoningvqa_bench/runs/vg-qwen25vl7b/smoke-train \
  --lora-scope language --trust-remote-code --max-rows 8
# full run:
$PY run_pipeline.py --config /data/wenjt/reasoningvqa_bench/configs/vg-qwen25vl7b.yaml train
```

## 2. iNaturalist + Qwen2.5-VL-7B

```bash
cd /data/wenjt/reasoningvqa_bench/pipeline
PY=/data/wenjt/conda_envs/videoespresso-common/bin/python
$PY run_pipeline.py --config /data/wenjt/reasoningvqa_bench/configs/inat-qwen25vl7b.yaml prepare
CUDA_VISIBLE_DEVICES=<free_gpu> TRANSFORMERS_OFFLINE=1 $PY run_pipeline.py \
  --config /data/wenjt/reasoningvqa_bench/configs/inat-qwen25vl7b.yaml train
```

## 3. Other model checkpoints (do NOT launch yet)

All three configs are placeholders and render correct commands, but they are
blocked as follows:

| Config | Checkpoint | Env | Row counts (already prepared*) | Blocker |
|---|---|---|---|---|
| `vg-qwen25vl32b.yaml` | Qwen2.5-VL-32B-Instruct | videoespresso-common | train 4891 / dev 974 | bf16 32B needs >=3 GPUs; trainer uses `device_map={"":0}` = one GPU per process |
| `vg-qwen3vl32b-awq.yaml` | Qwen3-VL-32B-Instruct-AWQ | vvqa4-qwen-awq | train 4891 / dev 974 | AWQ 32B needs >=2 GPUs; same single-GPU trainer limitation |
| `vg-minicpmv26.yaml` | MiniCPM-V-2_6 | vvqa4-minicpm-v26 | train 4891 / dev 974 | known chat-template blocker (transformers 4.40.0; falls back to `AutoModel` + trust_remote_code) |

Once enough GPUs are free and the trainer supports model parallelism for the 32B
checkpoints, each is a single command:

```bash
cd /data/wenjt/reasoningvqa_bench/pipeline
PY=/data/wenjt/conda_envs/videoespresso-common/bin/python   # qwen2.5-32b; use vvqa4-qwen-awq for AWQ
CFG=/data/wenjt/reasoningvqa_bench/configs/vg-qwen25vl32b.yaml
$PY run_pipeline.py --config "$CFG" prepare
CUDA_VISIBLE_DEVICES=8 TRANSFORMERS_OFFLINE=1 $PY run_pipeline.py --config "$CFG" train
```

*The VG prepare artifacts are shared per-config; each placeholder has its own
`output.experiment_root`, so run `prepare` for the config before `train`.
