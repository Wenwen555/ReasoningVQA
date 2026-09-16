#!/usr/bin/env bash
# =============================================================================
# STEP 00 (run ON THE REMOTE Ascend box) — KNOWN-BASELINE verification
# =============================================================================
# Usage:
#   bash 00_ascend_preflight.sh
#
# The target box was MEASURED on 2026-09-16, so this script no longer probes
# open questions. It INSTALLS NOTHING and CHANGES NOTHING. It:
#   - prints the measured identity of the box (image, FLVOR, device visibility)
#   - re-verifies each baseline value and flags [DRIFT] if it changed
#   - prints the scheduling conclusion for 32 GB HBM cards
#
# MEASURED BASELINE (2026-09-16):
#   image         : python_ascend:pytorch_2.7.1-cann_8.3.rc1-py_3.11-euler2.10.11-aarch64-snt9b-...-b4bd7bb
#                   (the recommended image is ALREADY RUNNING — do NOT pre-provision)
#   FLVOR         : modelarts.bm.npu.arm.4snt9b1
#   chips         : 4x Ascend910B4, 32 GB (32768 MB) HBM each, ~29 GB usable/card
#   device ids    : ASCEND_VISIBLE_DEVICES=2,3,4,5 (process view renumbers 0..3)
#   CANN          : /usr/local/Ascend/ascend-toolkit/8.3.RC1
#   python        : /home/ma-user/anaconda3/envs/PyTorch-2.7.1/bin/python (3.11)
#   torch stack   : torch 2.7.1+cpu / torch_npu 2.7.1.post2, device_count=4
#   work volume   : /home/ma-user/work (ext4) ~195 GB free, persistent
#   /data         : tmpfs mounted READ-ONLY -> origin /data/wenjt layout impossible
# =============================================================================
set -uo pipefail

# --- measured baseline ------------------------------------------------------
BASE_IMAGE="swr.cn-south-1.myhuaweicloud.com/atelier/pytorch_ascend:pytorch_2.7.1-cann_8.3.rc1-py_3.11-euler2.10.11-aarch64-snt9b-20260402124057-b4bd7bb"
BASE_FLVOR="modelarts.bm.npu.arm.4snt9b1"
BASE_CHIP="Ascend910B4"
BASE_CHIP_COUNT=4
BASE_HBM_MB=32768
BASE_HBM_USABLE_MB=29696        # ~29 GB after ~2.9 GB platform reservation
BASE_CANN_DIR="/usr/local/Ascend/ascend-toolkit/8.3.RC1"
BASE_PYTHON="/home/ma-user/anaconda3/envs/PyTorch-2.7.1/bin/python"
BASE_WORK_DIR="/home/ma-user/work"
BASE_WORK_MIN_GB=150

# --- helpers ----------------------------------------------------------------
section () { echo; echo "=============================================================="; echo "==> $*"; echo "=============================================================="; }
warn  () { echo "    [WARN]  $*"; }
ok    () { echo "    [OK]    $*"; }
drift () { echo "    [DRIFT] $*"; }

verify_eq () {
  local label="$1" expected="$2" actual="$3"
  if [[ "$actual" == "$expected" ]]; then
    ok "$label = $actual"
  else
    drift "$label: expected '$expected', got '$actual'"
  fi
}

verify_ge () {
  local label="$1" expected="$2" actual="$3"
  if [[ "$actual" =~ ^[0-9]+$ && "$actual" -ge "$expected" ]]; then
    ok "$label = $actual (>= $expected)"
  else
    drift "$label: expected >= $expected, got '$actual'"
  fi
}

echo "=============================================================="
echo "==> ReasoningVQA Ascend preflight (read-only, known-baseline check)"
echo "==> host: $(hostname 2>/dev/null)   user: $(id -un 2>/dev/null)   date: $(date -Iseconds 2>/dev/null)"
echo "=============================================================="

# -----------------------------------------------------------------------------
# 1. measured identity
# -----------------------------------------------------------------------------
section "[1] measured identity (CURRENT_IMAGE_NAME / FLVOR / device visibility)"
echo "    CURRENT_IMAGE_NAME        = ${CURRENT_IMAGE_NAME:-<unset>}"
echo "    FLVOR                     = ${FLVOR:-<unset>}"
echo "    ASCEND_VISIBLE_DEVICES    = ${ASCEND_VISIBLE_DEVICES:-<unset>}"
echo "    NPU_VISIBLE_DEVICES       = ${NPU_VISIBLE_DEVICES:-<unset>}"
echo "    ASCEND_RT_VISIBLE_DEVICES = ${ASCEND_RT_VISIBLE_DEVICES:-<unset>}"
echo "    hostname                  = $(hostname 2>/dev/null)"
echo "    user                      = $(id -un 2>/dev/null)"
echo "    arch                      = $(uname -m 2>/dev/null)"
echo "    uptime                    = $(uptime -p 2>/dev/null || uptime 2>/dev/null)"
echo
echo "  -- baseline re-check --"
verify_eq "arch" "aarch64" "$(uname -m 2>/dev/null)"
if [[ -n "${CURRENT_IMAGE_NAME:-}" ]]; then
  verify_eq "CURRENT_IMAGE_NAME" "$BASE_IMAGE" "$CURRENT_IMAGE_NAME"
else
  drift "CURRENT_IMAGE_NAME is unset (expected the atelier pytorch_ascend tag)"
fi
if [[ -n "${FLVOR:-}" ]]; then
  verify_eq "FLVOR" "$BASE_FLVOR" "$FLVOR"
else
  drift "FLVOR is unset (expected $BASE_FLVOR)"
fi

# -----------------------------------------------------------------------------
# 2. NPU hardware (verify the 4x 910B4 / 32 GB baseline)
# -----------------------------------------------------------------------------
section "[2] NPU hardware"
NPU_COUNT=0
MAX_HBM_MB=0
MIN_HBM_MB=0
if command -v npu-smi >/dev/null 2>&1; then
  echo "-- npu-smi info"
  npu-smi info 2>&1 || warn "npu-smi info failed"

  echo
  echo "-- chip family check"
  if npu-smi info 2>/dev/null | grep -q "$BASE_CHIP"; then
    ok "$BASE_CHIP detected in npu-smi output"
  else
    drift "expected chip $BASE_CHIP not found in npu-smi output"
  fi

  echo
  echo "-- per-chip HBM total (MB), parsed from the HBM-Usage column"
  HBM_LIST="$(npu-smi info 2>/dev/null | awk '{ c=gsub(/\//,"/"); if(c>=2){ n=split($0,a,"/"); t=a[n]; gsub(/[^0-9]/,"",t); if(t!="") print t } }')"
  if [[ -n "$HBM_LIST" ]]; then
    idx=0
    while IFS= read -r hbm; do
      echo "    chip ${idx}: HBM total = ${hbm} MB ($(awk -v m="$hbm" 'BEGIN{printf "%.1f", m/1024}') GiB)"
      idx=$((idx+1))
    done <<< "$HBM_LIST"
    NPU_COUNT="$(wc -l <<< "$HBM_LIST" | tr -d ' ')"
    MAX_HBM_MB="$(sort -n <<< "$HBM_LIST" | tail -1)"
    MIN_HBM_MB="$(sort -n <<< "$HBM_LIST" | head -1)"
    echo
    verify_eq "per-card HBM total (MB)" "$BASE_HBM_MB" "$MAX_HBM_MB"
    echo "    HBM total min/max = ${MIN_HBM_MB} / ${MAX_HBM_MB} MB"
    echo "    (baseline: 32 GB = ${BASE_HBM_MB} MB, ~$((BASE_HBM_USABLE_MB / 1024)) GB usable after platform overhead)"
  else
    warn "could not parse HBM totals from npu-smi output"
  fi
else
  warn "npu-smi not found in PATH (not an Ascend box, or driver not installed)"
fi

# -----------------------------------------------------------------------------
# 3. CANN toolkit (verify /usr/local/Ascend/ascend-toolkit/8.3.RC1)
# -----------------------------------------------------------------------------
section "[3] CANN toolkit"
if [[ -d "$BASE_CANN_DIR" ]]; then
  ok "CANN dir present: $BASE_CANN_DIR"
else
  drift "expected CANN dir missing: $BASE_CANN_DIR"
fi
echo "-- /usr/local/Ascend/ascend-toolkit"
ls -la /usr/local/Ascend/ascend-toolkit 2>/dev/null || warn "ascend-toolkit dir not found"

SET_ENV=/usr/local/Ascend/ascend-toolkit/set_env.sh
if [[ -r "$SET_ENV" ]]; then
  echo
  echo "-- sourcing $SET_ENV"
  set +u
  # shellcheck disable=SC1090
  source "$SET_ENV"
  set -u
  echo "    ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-<unset>}"
  echo "    ASCEND_TOOLKIT_HOME=${ASCEND_TOOLKIT_HOME:-<unset>}"
else
  warn "set_env.sh not found at $SET_ENV — skipping CANN env"
fi

# -----------------------------------------------------------------------------
# 4. base python + torch / torch_npu (verify the smoke-tested stack)
# -----------------------------------------------------------------------------
section "[4] base python + torch / torch_npu"
if [[ -x "$BASE_PYTHON" ]]; then
  ok "base python present: $BASE_PYTHON"
  "$BASE_PYTHON" - <<PY 2>&1 || warn "torch/torch_npu probe failed"
import importlib.metadata as md
import torch
import torch_npu  # noqa: F401

print("    torch            :", torch.__version__)
print("    torch_npu        :", md.version("torch_npu"))
avail = torch.npu.is_available()
count = torch.npu.device_count()
print("    npu.is_available :", avail)
print("    npu.device_count :", count)
try:
    print("    npu.get_device_name(0):", torch.npu.get_device_name(0))
except Exception as exc:
    print("    npu.get_device_name(0): <unavailable>", exc)

if avail and count == ${BASE_CHIP_COUNT}:
    print("    [OK]    NPU available with device_count == ${BASE_CHIP_COUNT}")
else:
    print("    [DRIFT] expected is_available=True and device_count==${BASE_CHIP_COUNT}")
    if count < ${BASE_CHIP_COUNT}:
        print("            -> fewer NPUs visible than measured; re-check ASCEND_RT_VISIBLE_DEVICES")
PY
else
  drift "base python missing: $BASE_PYTHON"
  warn "looked for: $BASE_PYTHON (conda env PyTorch-2.7.1)"
fi

# -----------------------------------------------------------------------------
# 5. disk (verify /home/ma-user/work; /data is read-only tmpfs)
# -----------------------------------------------------------------------------
section "[5] disk space"
echo "-- df -h $BASE_WORK_DIR (required >= ${BASE_WORK_MIN_GB} GB free)"
if df -h "$BASE_WORK_DIR" 2>/dev/null; then
  AVAIL_GB="$(df -BG "$BASE_WORK_DIR" 2>/dev/null | awk 'NR==2{gsub(/G/,"",$4); print $4}')"
  verify_ge "free GB on $BASE_WORK_DIR" "$BASE_WORK_MIN_GB" "$AVAIL_GB"
else
  drift "$BASE_WORK_DIR not mounted/listed"
fi
echo
echo "-- /data (must be READ-ONLY tmpfs; do NOT plan to write there)"
if findmnt "/data" >/dev/null 2>&1; then
  findmnt -no TARGET,FSTYPE,OPTIONS "/data" || true
else
  mount 2>/dev/null | grep ' /data ' || warn "/data not mounted"
fi

# -----------------------------------------------------------------------------
# 6. scheduling conclusion (32 GB HBM class)
# -----------------------------------------------------------------------------
section "[6] scheduling conclusion for 32 GB (32 GiB) HBM cards"
echo "    measured per-card HBM total : ${MAX_HBM_MB:-?} MB"
echo "    usable per card (approx.)   : $((BASE_HBM_USABLE_MB / 1024)) GB (after ~2.9 GB platform reservation)"
echo
if [[ "${MAX_HBM_MB:-0}" -ge 73728 ]]; then
  echo "    Qwen2.5-VL-32B (bf16 ~64 GB): single card would fit on this HBM class."
  echo "        --gpus 0   /   ASCEND_RT_VISIBLE_DEVICES=0"
else
  echo "    Qwen2.5-VL-32B-Instruct (bf16 ~64 GB): DOES NOT FIT ON ONE CARD."
  echo "        MUST shard across >=2 cards with device_map=\"balanced\":"
  echo "        export ASCEND_RT_VISIBLE_DEVICES=0,1     # use 0,1,2 if a 2-card split is tight"
  echo "        --gpus 0,1 --max-memory-per-gpu 26000"
fi
echo
echo "    Qwen2.5-VL-7B-Instruct (bf16 ~16 GB): FITS ON ONE CARD."
echo "        --gpus 0   /   ASCEND_RT_VISIBLE_DEVICES=0"
echo
echo "    MiniCPM-V-2_6: single card (verify the actual on-disk format/size first)."
echo
echo "    Measured 4-card layout (internal index -> physical card 2,3,4,5):"
echo "        card idx 0 (phys 2): vg-qwen25vl7b / inat-qwen25vl7b"
echo "        card idx 1 (phys 3): vg-minicpmv26 (if unblocked) or another 7B"
echo "        card idx 2-3 (phys 4-5): vg-qwen25vl32b --gpus 0,1  (or 0,1,2)"
echo "    See MIGRATION_PLAN.md §10 for the full schedule."

echo
echo "==> preflight done (read-only; nothing was installed or modified)"
