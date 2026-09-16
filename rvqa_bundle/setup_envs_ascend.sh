#!/usr/bin/env bash
# =============================================================================
# ReasoningVQA bench (workflow A) — rebuild the python envs on Ascend NPU
# =============================================================================
# Target box: Ascend snt9b (Atlas 800T A2 / 910B), aarch64, EulerOS 2.10.11,
#             Python 3.11.  NOT x86_64, NOT CUDA.
#
# IMAGE (frozen, from the official compatibility table):
#   PRIMARY  : 2.7.1-cann8.3.rc1-py3.11-euler2.10.11-aarch64-snt9b
#              torch_npu 2.7.1 <-> torch 2.7.1 <-> CANN 8.3.RC1 <-> py3.11
#   FALLBACK : 2.6.0-cann8.2.rc1-py3.11-euler2.10.11-aarch64-snt9b
#              torch_npu 2.6.0 <-> torch 2.6.0 <-> CANN 8.2.RC1
#
# Usage:  bash setup_envs_ascend.sh [ENV_ROOT]
#
# WHY ENV_ROOT IS /home/ma-user/work/rvqa/conda_envs (NOT /data/wenjt/...):
#   On the target box `/data` is a tmpfs mounted READ-ONLY, so ma-user cannot
#   create anything under it (measured 2026-09-16).  The persistent disk is
#   /home/ma-user/work (ext4, ~195 GB free), and 03_remote_unpack.sh rewrites
#   every hardcoded /data/wenjt/ path in configs/*.yaml to
#   /home/ma-user/work/rvqa/.  The env prefix below therefore matches the
#   rewritten `runtime.python:` rows.
#
# WHY CONDA CLONE (NOT `python3.11 -m venv --system-site-packages`):
#   The box already ships a working, mutually-matched Ascend stack in the conda
#   env /home/ma-user/anaconda3/envs/PyTorch-2.7.1 (torch 2.7.1+cpu /
#   torch_npu 2.7.1.post2 / CANN 8.3.RC1).  A venv would only *link* to those
#   system site-packages; a conda clone copies the whole env so the
#   torch / torch_npu / torchvision / CANN link relationships are inherited
#   intact and the experiment env is self-contained.  We then pip-override only
#   the packages listed in requirements-ascend*.txt; torch / torch_npu are
#   NEVER pip-installed.
#
# Run this AFTER the CANN environment is available in the shell:
#   source /usr/local/Ascend/ascend-toolkit/set_env.sh
#
# This script only builds 2 envs:
#   * videoespresso-common   (transformers 4.49.0 — vg-qwen25vl7b / inat-* / vg-32b)
#   * vvqa4-minicpm-v26      (transformers 4.40.0 — vg-minicpmv26)
# The AWQ env (vvqa4-qwen-awq) is intentionally SKIPPED (see the note at the end).
# =============================================================================
set -euo pipefail

# All roots default to empty; export them for your machine.
CONDA_BASE="${CONDA_BASE:-}"
BASE_ENV="${BASE_ENV:-}"
ENV_ROOT="${1:-${ENV_ROOT:-}}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQ="$HERE/requirements"
PY_VER=3.11
PIP_INDEX="https://repo.huaweicloud.com/repository/pypi/simple"

echo "==> ENV_ROOT = $ENV_ROOT"

if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
  # shellcheck disable=SC1091
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
echo "==> ASCEND_RT_VISIBLE_DEVICES=$ASCEND_RT_VISIBLE_DEVICES"
echo "==> Ascend devices on this box:"
npu-smi info || echo "    (npu-smi unavailable — check /usr/local/Ascend/ascend-toolkit/set_env.sh)"

# --- locate conda (the clone source) ----------------------------------------
# CONDA_BASE defaults to the measured remote location /home/ma-user/anaconda3;
# override it if the target box keeps conda elsewhere.
CONDA=""
if command -v conda >/dev/null 2>&1; then
  CONDA="$(command -v conda)"
fi
if [[ -z "$CONDA" || ! -x "$CONDA" ]]; then
  for candidate in "$CONDA_BASE/bin/conda" "$CONDA_BASE/condabin/conda"; do
    if [[ -x "$candidate" ]]; then
      CONDA="$candidate"
      break
    fi
  done
fi
if [[ -z "$CONDA" || ! -x "$CONDA" ]]; then
  echo "FATAL: no conda found (looked in PATH and $CONDA_BASE/{bin,condabin})" >&2
  exit 1
fi
if [[ ! -d "$BASE_ENV" ]]; then
  echo "FATAL: base env to clone not found: $BASE_ENV" >&2
  echo "       set CONDA_BASE=... or BASE_ENV=... to override" >&2
  exit 1
fi
echo "==> conda    = $CONDA"
echo "==> base env = $BASE_ENV"

mkdir -p "$ENV_ROOT"

build_env () {
  local name="$1" reqfile="$2"
  echo
  echo "=============================================================="
  echo "==> building $name  from  $reqfile"
  echo "=============================================================="
  if [[ -x "$ENV_ROOT/$name/bin/python" ]]; then
    echo "    $ENV_ROOT/$name already exists — skipping clone"
  else
    "$CONDA" create -y -p "$ENV_ROOT/$name" --clone "$BASE_ENV"
  fi
  local P="$ENV_ROOT/$name/bin/python"
  "$P" -m pip install --upgrade pip setuptools wheel -i "$PIP_INDEX"
  "$P" -m pip install -r "$reqfile" -i "$PIP_INDEX"
  echo "--- $name installed versions ---"
  "$P" - <<'EOF'
import importlib.metadata as m
for p in ["torch", "torch_npu", "torchvision", "transformers", "tokenizers", "accelerate", "peft", "numpy", "pillow"]:
    try:    print(f"    {p} == {m.version(p)}")
    except Exception: print(f"    {p} MISSING")
EOF
}

build_env videoespresso-common  "$REQ/requirements-ascend.txt"
build_env vvqa4-minicpm-v26     "$REQ/requirements-ascend-minicpmv26.txt"

echo
echo "=============================================================="
echo "==> AWQ env (vvqa4-qwen-awq) — SKIPPED THIS ROUND"
echo "=============================================================="
cat <<'EOF'
Qwen3-VL-32B-Instruct-AWQ is NOT viable on Ascend through plain transformers:
  * transformers 5.8.0's AwqQuantizer hard-depends on gptqmodel, which PyPI
    only ships as an sdist (needs source compilation), and
  * every default gptqmodel/AWQ backend is a CUDA kernel.

Therefore configs/vg-qwen3vl32b-awq.yaml stays blocked/PLACEHOLDER on the
target box.  The replacement path is:
  msModelSlim (W8A8 / W4A16 quantisation)  +  vLLM-Ascend inference.
Do NOT run `pip install -r requirements/requirements-qwen3vl32b-awq.txt` here.
EOF

echo
echo "=============================================================="
echo "==> smoke test (main env): NPU + versions + device_map"
echo "=============================================================="
"$ENV_ROOT/videoespresso-common/bin/python" - <<'EOF'
import sys
import importlib.metadata as md

def ver(package):
    try:
        return md.version(package)
    except Exception:
        return "MISSING"

ok = True
try:
    import torch
except Exception as exc:  # pragma: no cover - environment probe
    print("FATAL: import torch failed:", exc)
    sys.exit(1)
try:
    import torch_npu  # noqa: F401
except Exception as exc:  # pragma: no cover - environment probe
    print("FATAL: import torch_npu failed:", exc)
    sys.exit(1)

if not torch.npu.is_available():
    print("FATAL: torch.npu.is_available() is False")
    ok = False
else:
    print("    NPU available, device count = {}".format(torch.npu.device_count()))

print("--- version table (main env) ---")
for package in ["torch", "torch_npu", "torchvision", "transformers", "peft", "accelerate", "numpy", "pillow"]:
    print("    {:<14s} {}".format(package, ver(package)))

# device_map={"": 0} must resolve to npu:0 (accelerate maps an int to npu:<n>
# when torch_npu is installed).  This exercises the exact trainer code path.
try:
    import torch.nn as nn
    from accelerate import dispatch_model

    probe = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 8))
    probe = dispatch_model(probe, {"": 0})
    device = next(probe.parameters()).device
    print("    device_map={'': 0} ->", device)
    if device.type != "npu":
        print("FATAL: device_map={'': 0} did not land on npu (got {})".format(device))
        ok = False
except Exception as exc:  # pragma: no cover - environment probe
    print("FATAL: device_map smoke failed:", repr(exc))
    ok = False

if not ok:
    sys.exit(1)
print("==> smoke test PASSED")
EOF

echo
echo "==> DONE. Verify with:"
echo "    source /usr/local/Ascend/ascend-toolkit/set_env.sh"
echo "    $ENV_ROOT/videoespresso-common/bin/python -c \"import torch,torch_npu,transformers;print(torch.__version__, torch_npu.__version__, transformers.__version__)\""
echo "    bash smoke_test_ascend.sh   # full remote smoke (imports + tests + 1-row eval)"
