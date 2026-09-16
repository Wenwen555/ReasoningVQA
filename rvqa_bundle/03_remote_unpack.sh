#!/usr/bin/env bash
# =============================================================================
# STEP 03 (run ON THE REMOTE box) — unpack, then rewrite hardcoded paths
# =============================================================================
# Usage:  bash 03_remote_unpack.sh
#
# Assumes the bundle landed in $REPO_ROOT/rvqa_bundle (see 02_transfer_models.sh).
# REPO_ROOT is the MEASURED remote prefix /home/ma-user/work/rvqa (2026-09-16).
#
# WHY WE REWRITE PATHS: on the target box `/data` is a tmpfs mounted READ-ONLY,
# so the origin /data/wenjt layout CANNOT be recreated.  Every hardcoded origin
# path in the extracted tree is rewritten instead:
#
#   /data/wenjt/         -> /home/ma-user/work/rvqa/
#   /data/llm_models/    -> /home/ma-user/work/rvqa/llm_models/
#   CUDA_VISIBLE_DEVICES -> ASCEND_RT_VISIBLE_DEVICES
#
# Rewritten file set:
#   reasoningvqa_bench/configs/*.yaml
#   reasoningvqa_bench/runs/*/pipeline_config.yaml      (if runs/ was unpacked)
#   reasoningvqa_bench/scoring/score_predictions.py
#   reasoningvqa_bench/**/*.py  that contain a hardcoded absolute path
#                               or CUDA_VISIBLE_DEVICES
#   reasoningvqa_bench/runs/**/data/**/*.jsonl          (streaming, image_path;
#       this also covers the historical inat manifests-open jsonl)
#   reasoningvqa_bench/data/*.jsonl                     (if present)
#
# Safe to re-run: unpacking is idempotent; every file the rewriter touches is
# backed up to *.pre-migrate.bak first (cp -n, so an existing backup is kept).
# JSONL is rewritten line-by-line in Python (never read whole-file) and only
# files that actually contain an origin prefix are touched.
# =============================================================================
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/ma-user/work/rvqa}"
BUNDLE="$REPO_ROOT/rvqa_bundle"
BENCH="$REPO_ROOT/reasoningvqa_bench"

echo "==> repo root: $REPO_ROOT"
[[ -d "$BUNDLE" ]] || { echo "MISSING bundle dir: $BUNDLE"; exit 1; }

# -----------------------------------------------------------------------------
# 1. unpack
# -----------------------------------------------------------------------------
echo "==> [1/6] unpacking"
cd "$BUNDLE"

tar -C "$REPO_ROOT" -xzf reasoningvqa_bench_code.tgz
echo "   code   -> $REPO_ROOT/"

tar -C "$REPO_ROOT" -xzf reasoningvqa_subsets_data.tgz
echo "   subsets-> $REPO_ROOT/reasoningvqa_subsets/"
# NOTE: this archive holds the three dataset dirs. It does NOT contain _work/ or *.bak.

# bench data/*.english.jsonl are extracted into the code tree
tar -C "$BENCH" -xzf reasoningvqa_bench_data.tgz
echo "   data   -> $BENCH/data/"

if [[ -f reasoningvqa_official.tgz ]]; then
  tar -C "$REPO_ROOT" -xzf reasoningvqa_official.tgz
  echo "   official -> $REPO_ROOT/reasoningvqa_official/"
fi

# scoring extras
mkdir -p "$REPO_ROOT/projects/ReasoningVQA/experiments/RVQA-REASONVQA-GLDV2-PAPER-REIMPL-20260802/models"
cp -r scoring_extra/all-MiniLM-L6-v2 \
  "$REPO_ROOT/projects/ReasoningVQA/experiments/RVQA-REASONVQA-GLDV2-PAPER-REIMPL-20260802/models/"
echo "   MiniLM -> .../models/all-MiniLM-L6-v2"

mkdir -p "$REPO_ROOT/reasoningvqa_bench/vendor"
cp scoring_extra/evaluate_reasonvqa_mc_qwen.py "$REPO_ROOT/reasoningvqa_bench/vendor/"
echo "   MC evaluator -> $BENCH/vendor/evaluate_reasonvqa_mc_qwen.py"

# inat manifests-open
mkdir -p "$BENCH/runs/inat-qwen25vl7b/data/manifests-open"
cp -r inat_manifests_open/. "$BENCH/runs/inat-qwen25vl7b/data/manifests-open/"
echo "   inat manifests-open restored"

# -----------------------------------------------------------------------------
# 2. rewrite the two hardcoded Python constants in scoring/score_predictions.py
# -----------------------------------------------------------------------------
echo "==> [2/6] rewriting scoring hardcoded paths"
SP="$BENCH/scoring/score_predictions.py"
if [[ -f "$SP" ]]; then
  cp -n "$SP" "$SP.pre-migrate.bak"
  python3 - "$SP" "$BENCH" "$REPO_ROOT" <<'PY'
import re, sys, pathlib
sp, bench, repo = sys.argv[1], sys.argv[2], sys.argv[3]
p = pathlib.Path(sp)
src = p.read_text()

# CANONICAL_MC -> <bench>/vendor/evaluate_reasonvqa_mc_qwen.py
#   (origin pointed at /home/wenjt/project/.worktrees/... which is NOT under /data,
#    so the generic /data/ rewrite in step 3 would not catch it)
src = re.sub(
    r'CANONICAL_MC = Path\([^)]*\)',
    f'CANONICAL_MC = Path("{bench}/vendor/evaluate_reasonvqa_mc_qwen.py")',
    src, flags=re.S)

# DEFAULT_MODEL_PATH -> <repo>/projects/.../all-MiniLM-L6-v2
src = re.sub(
    r'DEFAULT_MODEL_PATH = Path\([^)]*\)',
    f'DEFAULT_MODEL_PATH = Path(\n'
    f'    "{repo}/projects/ReasoningVQA/experiments/"\n'
    f'    "RVQA-REASONVQA-GLDV2-PAPER-REIMPL-20260802/models/all-MiniLM-L6-v2"\n'
    f')',
    src, flags=re.S)

p.write_text(src)
print("   patched", sp)
PY
  grep -n -A3 "CANONICAL_MC = Path\|DEFAULT_MODEL_PATH = Path" "$SP"
else
  echo "   MISSING $SP — skipped"
fi

# -----------------------------------------------------------------------------
# 3. generic path rewrite: /data/... -> $REPO_ROOT, CUDA -> ASCEND
# -----------------------------------------------------------------------------
echo "==> [3/6] rewriting hardcoded origin paths -> $REPO_ROOT"

OLD_PREFIX_A="/data/wenjt/"
NEW_PREFIX_A="$REPO_ROOT/"
OLD_PREFIX_B="/data/llm_models/"
NEW_PREFIX_B="$REPO_ROOT/llm_models/"

REWRITE_LIST="$(mktemp)"
{
  # experiment configs
  find "$BENCH/configs" -type f -name '*.yaml' 2>/dev/null || true
  # per-run pipeline configs (only present if runs/ was ever unpacked)
  find "$BENCH/runs" -type f -name 'pipeline_config.yaml' 2>/dev/null || true
  # scoring module (two hardcoded constants)
  [[ -f "$BENCH/scoring/score_predictions.py" ]] && echo "$BENCH/scoring/score_predictions.py"
  # every other .py that still contains a hardcoded origin path OR CUDA_VISIBLE_DEVICES
  grep -rlE '/data/(wenjt|llm_models)|CUDA_VISIBLE_DEVICES' "$BENCH" --include='*.py' 2>/dev/null || true
} | sort -u > "$REWRITE_LIST"

CHANGED=0
while IFS= read -r f; do
  [[ -n "$f" && -f "$f" ]] || continue
  # skip files that need no change (avoids touching unrelated .yaml/.py)
  grep -qE '/data/(wenjt|llm_models)|CUDA_VISIBLE_DEVICES' "$f" || continue
  cp -n "$f" "$f.pre-migrate.bak"
  sed -i \
    -e "s#${OLD_PREFIX_A}#${NEW_PREFIX_A}#g" \
    -e "s#${OLD_PREFIX_B}#${NEW_PREFIX_B}#g" \
    -e 's#CUDA_VISIBLE_DEVICES#ASCEND_RT_VISIBLE_DEVICES#g' \
    "$f"
  echo "   rewritten: $f"
  CHANGED=$((CHANGED + 1))
done < "$REWRITE_LIST"
rm -f "$REWRITE_LIST"
echo "   files rewritten: $CHANGED (each backed up as <file>.pre-migrate.bak)"

# -----------------------------------------------------------------------------
# 4. streaming rewrite of *.jsonl (image_path etc. live here, not in step 3)
# -----------------------------------------------------------------------------
# `reasoningvqa_bench/runs/inat-qwen25vl7b/data/manifests-open/*.jsonl` (a
# historical artifact, shipped separately by 01_pack_code_data.sh as
# scoring_extra/inat_manifests_open) stores absolute
# `/data/wenjt/reasoningvqa_subsets/...` image_path values.  Since /data is
# read-only on the target box we CANNOT symlink around it, so we rewrite the
# jsonl too.  Files can be large: parse one line at a time, never load whole.
echo "==> [4/6] streaming-rewriting *.jsonl (runs/**/data, inat manifests-open, data/)"
JSONL_LIST="$(mktemp)"
{
  # all run data dirs: covers runs/inat-qwen25vl7b/data/manifests-open/*.jsonl
  find "$BENCH/runs" -type f -path '*/data/*' -name '*.jsonl' 2>/dev/null || true
  # explicit inat manifests-open (also matched above; kept for clarity)
  find "$BENCH/runs/inat-qwen25vl7b/data/manifests-open" -type f -name '*.jsonl' 2>/dev/null || true
  # bench data/*.jsonl (non-recursive) if present
  find "$BENCH/data" -maxdepth 1 -type f -name '*.jsonl' 2>/dev/null || true
} | sort -u > "$JSONL_LIST"

JSONL_FILES=0
JSONL_ROWS=0
while IFS= read -r f; do
  [[ -n "$f" && -f "$f" ]] || continue
  # only touch files that really carry an origin prefix
  grep -qE '/data/(wenjt|llm_models)/' "$f" || continue
  cp -n "$f" "$f.pre-migrate.bak"
  n_lines="$(python3 - "$f" "$OLD_PREFIX_A" "$NEW_PREFIX_A" "$OLD_PREFIX_B" "$NEW_PREFIX_B" <<'PY'
import os, sys
path, old_a, new_a, old_b, new_b = sys.argv[1:6]
n = 0
# Streaming rewrite: one line in memory at a time; the source is never fully read.
with open(path, "r", encoding="utf-8") as src, open(path + ".tmp", "w", encoding="utf-8") as dst:
    for line in src:
        new = line.replace(old_a, new_a).replace(old_b, new_b)
        if new != line:
            n += 1
        dst.write(new)
os.replace(path + ".tmp", path)
print(n)
PY
)"
  echo "   rewritten: $f  (改写了 $n_lines 行)"
  JSONL_FILES=$((JSONL_FILES + 1))
  JSONL_ROWS=$((JSONL_ROWS + n_lines))
done < "$JSONL_LIST"
rm -f "$JSONL_LIST"
echo "   jsonl files rewritten: $JSONL_FILES, total lines changed: $JSONL_ROWS (each backed up as <file>.pre-migrate.bak)"

# -----------------------------------------------------------------------------
# 5. residual /data reference report (manual review)
# -----------------------------------------------------------------------------
echo "==> [5/6] residual /data references after rewrite (manual review)"
RESIDUAL="$(
  grep -rn '/data/' "$BENCH" \
    --include='*.yaml' --include='*.yml' --include='*.py' \
    --include='*.json' --include='*.jsonl' \
    2>/dev/null | grep -v '\.pre-migrate\.bak' || true
)"
if [[ -n "$RESIDUAL" ]]; then
  echo "$RESIDUAL" | sed 's/^/      /'
  echo "   ^ review each line above: rewrite it or provide a symlink before running"
  echo "     (jsonl image_path IS covered by step 4; remaining hits are non-jsonl/other roots)"
else
  echo "   (none)"
fi

# -----------------------------------------------------------------------------
# 6. verification
# -----------------------------------------------------------------------------
echo "==> [6/6] verifying"
cat > /tmp/rvqa_verify.py <<'PY'
import json, os, pathlib, sys
repo = pathlib.Path(os.environ.get("REPO_ROOT", "/home/ma-user/work/rvqa"))
bench = repo / "reasoningvqa_bench"
subsets = repo / "reasoningvqa_subsets"
ok = True

print("--- bench english jsonl + image availability (train + eval) ---")
for ds in ("inaturalist", "visual_genome", "gldv2"):
    img_root = subsets / ds / "images"
    n_img = len(list(img_root.glob("*.jpg"))) if img_root.exists() else 0
    rows_total = 0
    missing = 0
    for split in ("train", "eval"):
        p = bench / "data" / ds / f"{split}.english.jsonl"
        if not p.exists():
            print(f"  {ds:14s} [{split:5s}] MISSING {p}"); ok = False; continue
        rows = [json.loads(l) for l in p.open() if l.strip()]
        rows_total += len(rows)
        for r in rows:
            name = r.get("image_name") or r.get("image") or ""
            ip = r.get("image_path") or ""
            cand = pathlib.Path(ip) if ip and os.path.isabs(ip) else (img_root / name if name else None)
            if cand and not cand.exists():
                missing += 1
    print(f"  {ds:14s} rows={rows_total:5d} images={n_img:5d} missing_refs={missing}")
    # Only a completely empty image dir is fatal. Missing individual refs are
    # reported for information and never fail the run.
    if n_img == 0:
        ok = False
    if ds == "gldv2":
        print(f"  gldv2 missing_refs={missing} (ACCEPTED \u2014 user decision, do not fail)")

print("--- code ---")
for rel in ("pipeline/run_pipeline.py", "scoring/score_predictions.py",
            "inventories/qwen25vl7b-instruct.json", "vendor/evaluate_reasonvqa_mc_qwen.py"):
    q = bench / rel
    print(f"  {'OK ' if q.exists() else 'MISS'} {q}")
    ok = ok and q.exists()

print("--- models (MEASURED remote layout: /home/ma-user/work/rvqa) ---")
models = (
    "/home/ma-user/work/rvqa/llm_models/Qwen2.5-VL-7B-Instruct",
    "/home/ma-user/work/rvqa/models/llm_models/MiniCPM-V-2_6",
    "/home/ma-user/work/rvqa/models/llm_models/Qwen2.5-VL-32B-Instruct",
)
for m in models:
    exists = pathlib.Path(m).exists()
    print(f"  {'OK ' if exists else 'MISS'} {m}")
    if not exists:
        ok = False

# Deliberately not migrated; the user downloads it on the remote. Expected to be absent.
awq = "/home/ma-user/work/rvqa/llm_models/Qwen3-VL-32B-Instruct-AWQ"
print(f"  INFO {awq}")
print("       \u9884\u671f\u7f3a\u5931\uff1a\u7531\u7528\u6237\u8fdc\u7aef\u81ea\u884c\u4e0b\u8f7d (expected absent: user downloads it on the remote)")

print()
print("RESULT:", "READY" if ok else "NEEDS ATTENTION")
PY
REPO_ROOT="$REPO_ROOT" "$REPO_ROOT/conda_envs/videoespresso-common/bin/python" /tmp/rvqa_verify.py 2>/dev/null \
  || REPO_ROOT="$REPO_ROOT" python3 /tmp/rvqa_verify.py

echo
echo "==> done. Next:"
echo "    1) bash $BUNDLE/setup_envs_ascend.sh   # clone PyTorch-2.7.1 + apply overrides"
echo "    2) verify ASCEND_RT_VISIBLE_DEVICES in configs/*.yaml for this box"
echo "    3) \$PY pipeline/run_pipeline.py --config configs/<exp>.yaml prepare --dry-run"
