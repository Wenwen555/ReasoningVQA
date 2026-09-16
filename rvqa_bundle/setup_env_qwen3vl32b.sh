#!/usr/bin/env bash
# =============================================================================
# setup_env_qwen3vl32b.sh — build the Qwen3-VL-32B-Instruct (BF16) Ascend env
# =============================================================================
# Env prefix: /home/ma-user/work/rvqa/conda_envs/vvqa4-qwen3vl32b
#   (historical name — the configs/vg-qwen3vl32b-instruct.yaml runtime.python path)
#
#   conda clone of /home/ma-user/anaconda3/envs/PyTorch-2.7.1
#   + requirements/requirements-ascend-qwen3vl32b.txt  (transformers 5.8.0, ...)
#
# torch / torch_npu / torchvision are inherited from the clone, never pip-installed.
# pip / setuptools / wheel are deliberately NOT upgraded: the base 25.3 / 80.9.0
# work fine, and upgrading setuptools breaks the CANN mindstudio-probe pin.
#
# Usage:  bash setup_env_qwen3vl32b.sh
# =============================================================================
set -euo pipefail

CONDA_BASE="${CONDA_BASE:-/home/ma-user/anaconda3}"
BASE_ENV="${BASE_ENV:-$CONDA_BASE/envs/PyTorch-2.7.1}"
ENV_ROOT="${ENV_ROOT:-/home/ma-user/work/rvqa/conda_envs}"
NAME="${NAME:-vvqa4-qwen3vl32b}"
MODEL_DIR="${MODEL_DIR:-/home/ma-user/work/rvqa/llm_models/Qwen3-VL-32B-Instruct}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQ="$HERE/requirements/requirements-ascend-qwen3vl32b.txt"
PIP_INDEX="https://repo.huaweicloud.com/repository/pypi/simple"

if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
  # shellcheck disable=SC1091
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"

# --- locate conda -----------------------------------------------------------
CONDA=""
if command -v conda >/dev/null 2>&1; then CONDA="$(command -v conda)"; fi
if [[ -z "$CONDA" || ! -x "$CONDA" ]]; then
  for candidate in "$CONDA_BASE/bin/conda" "$CONDA_BASE/condabin/conda"; do
    [[ -x "$candidate" ]] && CONDA="$candidate" && break
  done
fi
[[ -x "$CONDA" ]] || { echo "FATAL: no conda found" >&2; exit 1; }
[[ -d "$BASE_ENV" ]] || { echo "FATAL: base env missing: $BASE_ENV" >&2; exit 1; }
[[ -f "$REQ" ]] || { echo "FATAL: requirements missing: $REQ" >&2; exit 1; }

echo "==> conda    = $CONDA"
echo "==> base env = $BASE_ENV"
echo "==> env      = $ENV_ROOT/$NAME"
echo "==> reqs     = $REQ"

mkdir -p "$ENV_ROOT"

if [[ -x "$ENV_ROOT/$NAME/bin/python" ]]; then
  echo "==> $NAME already exists — skipping clone"
else
  echo "==> cloning base env (this takes a few minutes)"
  "$CONDA" create -y -p "$ENV_ROOT/$NAME" --clone "$BASE_ENV"
fi

P="$ENV_ROOT/$NAME/bin/python"

echo
echo "==> upgrading pip install machinery ONLY if needed (skipped on purpose)"
echo "    (base pip/setuptools/wheel are kept so mindstudio-probe stays satisfied)"

echo
echo "==> installing Qwen3-VL overrides"
"$P" -m pip install -r "$REQ" -i "$PIP_INDEX"

echo
echo "--- $NAME installed versions ---"
"$P" - <<'EOF'
import importlib.metadata as m
for p in ["torch", "torch_npu", "torchvision", "transformers", "tokenizers",
          "accelerate", "peft", "huggingface-hub", "numpy", "pillow",
          "safetensors", "qwen-vl-utils", "setuptools"]:
    try:
        print(f"    {p:18s} {m.version(p)}")
    except Exception:
        print(f"    {p:18s} MISSING")
EOF

echo
echo "=============================================================="
echo "==> smoke test: NPU + qwen3_vl support + processor load"
echo "=============================================================="
MODEL_DIR="$MODEL_DIR" "$P" - <<'EOF'
import os, sys
import importlib.metadata as md

ok = True

import torch
import torch_npu  # noqa: F401
if not torch.npu.is_available():
    print("FATAL: torch.npu.is_available() is False"); ok = False
else:
    print(f"    NPU available, device_count={torch.npu.device_count()}")

import transformers
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
print(f"    transformers {transformers.__version__}")
has_qwen3 = "qwen3_vl" in CONFIG_MAPPING
print(f"    CONFIG_MAPPING has 'qwen3_vl': {has_qwen3}")
if not has_qwen3:
    print("FATAL: this transformers does not know qwen3_vl"); ok = False

model_dir = os.environ["MODEL_DIR"]
if os.path.isdir(model_dir) and has_qwen3:
    try:
        from transformers import AutoConfig, AutoProcessor
        cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
        print(f"    AutoConfig  OK  model_type={cfg.model_type} arch={cfg.architectures}")
        proc = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
        print(f"    AutoProcessor OK  {type(proc).__name__}")
    except Exception as exc:
        print(f"    FATAL: config/processor load failed: {exc!r}"); ok = False
else:
    print(f"    (skip config/processor load: model dir {model_dir} not present yet)")

# device_map={'':0} must resolve to npu:0 (same check as the other envs)
try:
    import torch.nn as nn
    from accelerate import dispatch_model
    probe = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 8))
    probe = dispatch_model(probe, {"": 0})
    dev = next(probe.parameters()).device
    print(f"    device_map={{'': 0}} -> {dev}")
    if dev.type != "npu":
        print("FATAL: device_map did not land on npu"); ok = False
except Exception as exc:
    print(f"    FATAL: device_map smoke failed: {exc!r}"); ok = False

print()
print("RESULT:", "PASSED" if ok else "FAILED")
sys.exit(0 if ok else 1)
EOF

echo
echo "==> DONE. Env: $ENV_ROOT/$NAME"
echo "    $P -c \"import transformers;print(transformers.__version__)\""
