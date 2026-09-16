#!/usr/bin/env bash
# =============================================================================
# ReasoningVQA bench (workflow A) — remote smoke test on Ascend NPU
# =============================================================================
# Run this ON THE TARGET BOX after setup_envs_ascend.sh succeeds.
#
# Usage:
#   source /usr/local/Ascend/ascend-toolkit/set_env.sh
#   bash smoke_test_ascend.sh
#
# Optional overrides:
#   REPO_ROOT=/home/ma-user/work/rvqa
#   ENV_ROOT=$REPO_ROOT/conda_envs
#   ASCEND_RT_VISIBLE_DEVICES=0
#   ASCEND_SMOKE_MODEL=/home/ma-user/work/rvqa/llm_models/Qwen2.5-VL-7B-Instruct
#   ASCEND_SMOKE_MANIFEST=/abs/path/to/dev.jsonl   # canonical manifest for (c)
#
# Four steps:
#   (a) basic imports + torch.npu.is_available()
#   (b) run the pipeline CPU-only unit tests (python -m unittest discover)
#   (c) run_pipeline.py --dry-run + 1-row Qwen2.5-VL-7B evaluation
#   (d) npu-smi info + the current env version table
#
# The script does NOT stop at the first failure: it runs all four steps and
# prints a summary naming every failed step at the end.
# =============================================================================
set -uo pipefail

REPO_ROOT="${REPO_ROOT:-/home/ma-user/work/rvqa}"
ENV_ROOT="${ENV_ROOT:-$REPO_ROOT/conda_envs}"
BENCH="$REPO_ROOT/reasoningvqa_bench"
PIPELINE="$BENCH/pipeline"
PY="$ENV_ROOT/videoespresso-common/bin/python"
CFG="$BENCH/configs/vg-qwen25vl7b.yaml"
MODEL="${ASCEND_SMOKE_MODEL:-/home/ma-user/work/rvqa/llm_models/Qwen2.5-VL-7B-Instruct}"

if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
  # shellcheck disable=SC1091
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi
# ASCEND_RT_VISIBLE_DEVICES uses the in-process logical index (0..3); on this
# box logical 0 == physical card 2 (platform maps 2,3,4,5 -> 0..3), so the
# 1-row smoke runs on a single logical NPU without touching physical 0/1.
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"

echo "==> ENV_ROOT              = $ENV_ROOT"
echo "==> REPO_ROOT             = $REPO_ROOT"
echo "==> python                = $PY"
echo "==> config                = $CFG"
echo "==> model                 = $MODEL"
echo "==> ASCEND_RT_VISIBLE_DEVICES = $ASCEND_RT_VISIBLE_DEVICES"

if [[ ! -x "$PY" ]]; then
  echo "FATAL: $PY not found/executable — run setup_envs_ascend.sh first" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# step helpers
# ---------------------------------------------------------------------------
FAILED_STEPS=()
check () {
  local label="$1"; shift
  echo
  echo "=============================================================="
  echo "==> $label"
  echo "=============================================================="
  if "$@"; then
    echo "[OK]   $label"
  else
    local rc=$?
    echo "[FAIL] $label (exit $rc)"
    FAILED_STEPS+=("$label")
  fi
}

# ---------------------------------------------------------------------------
# (a) basic imports + NPU availability
# ---------------------------------------------------------------------------
step_a_imports () {
  "$PY" - <<'PY'
import sys
import importlib.metadata as md

for package in ["torch", "torch_npu", "torchvision", "transformers", "peft", "accelerate"]:
    try:
        module = __import__(package)
        print("    import {:<14s} OK  {}".format(package, getattr(module, "__version__", "?")))
    except Exception as exc:
        print("    import {:<14s} FAILED: {}".format(package, exc))
        sys.exit(1)

import torch
import torch_npu  # noqa: F401

if not torch.npu.is_available():
    print("FATAL: torch.npu.is_available() is False")
    sys.exit(1)
print("    torch.npu.is_available() = True, device count = {}".format(torch.npu.device_count()))
print("    torch_npu = {}".format(md.version("torch_npu")))
PY
}

# ---------------------------------------------------------------------------
# (b) pipeline CPU-only unit tests
# ---------------------------------------------------------------------------
step_b_tests () {
  ( cd "$PIPELINE" && "$PY" -m unittest discover -s tests -p 'test_*.py' -v )
}

# ---------------------------------------------------------------------------
# (c) evaluate smoke: dry-run + 1 real sample
# ---------------------------------------------------------------------------
step_c_eval () {
  echo "-- (c1) run_pipeline.py --config vg-qwen25vl7b.yaml evaluate --dry-run"
  ( cd "$PIPELINE" && "$PY" run_pipeline.py --config "$CFG" evaluate --dry-run ) || return 1

  echo
  echo "-- (c2) 1-row Qwen2.5-VL-7B evaluation"
  local manifest="${ASCEND_SMOKE_MANIFEST:-$BENCH/runs/vg-qwen25vl7b/data/manifests-open/dev.jsonl}"
  if [[ ! -f "$manifest" ]]; then
    echo "    [WARN] canonical manifest not found: $manifest"
    echo "    [WARN] SKIPPING the 1-row generation (dry-run above still validated the CLI)."
    echo "    [WARN] set ASCEND_SMOKE_MANIFEST=/abs/path/dev.jsonl to force it."
    return 0
  fi
  local out rc=0
  out="$(mktemp -d "${TMPDIR:-/tmp}/rvqa-smoke.XXXXXX")"
  "$PY" "$PIPELINE/pipelines/evaluate/evaluate_canonical_reasoning.py" \
      --manifest "$manifest" \
      --model "$MODEL" \
      --output "$out/eval" \
      --answer-mode open \
      --mode correct-image \
      --max-rows 1 \
      --max-new-tokens 32 || rc=$?
  if [[ $rc -eq 0 ]]; then
    echo "    generated rows:"
    sed -n '1,2p' "$out/eval/predictions.jsonl" 2>/dev/null || true
  fi
  rm -rf "$out"
  return $rc
}

# ---------------------------------------------------------------------------
# (d) npu-smi + version table
# ---------------------------------------------------------------------------
step_d_report () {
  echo "-- npu-smi info"
  npu-smi info || return 1
  echo
  echo "-- current env version table"
  "$PY" - <<'PY'
import importlib.metadata as md

def ver(package):
    try:
        return md.version(package)
    except Exception:
        return "MISSING"

for package in ["torch", "torch_npu", "torchvision", "transformers", "tokenizers",
                "accelerate", "peft", "numpy", "pillow", "safetensors"]:
    print("    {:<14s} {}".format(package, ver(package)))
PY
}

# ---------------------------------------------------------------------------
# run all four, then summarise
# ---------------------------------------------------------------------------
check "step (a): imports + NPU availability"       step_a_imports
check "step (b): pipeline CPU-only unittest"       step_b_tests
check "step (c): evaluate dry-run + 1-row sample"  step_c_eval
check "step (d): npu-smi info + version table"     step_d_report

echo
echo "=============================================================="
echo "==> SMOKE SUMMARY"
echo "=============================================================="
if [[ ${#FAILED_STEPS[@]} -eq 0 ]]; then
  echo "    ALL 4 STEPS PASSED"
  exit 0
fi
echo "    FAILED STEPS (${#FAILED_STEPS[@]}):"
for step in "${FAILED_STEPS[@]}"; do
  echo "      - $step"
done
exit 1
