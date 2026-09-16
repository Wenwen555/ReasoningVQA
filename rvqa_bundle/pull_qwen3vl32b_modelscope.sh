#!/usr/bin/env bash
# =============================================================================
# pull_qwen3vl32b_modelscope.sh — download Qwen3-VL-32B-Instruct from ModelScope
# =============================================================================
# RUN ON THE REMOTE Ascend box (ModelArts).
#
# WHY MODELSCOPE + curl (measured on this box, 2026-09-16):
#   The box has NO direct internet route; everything goes through
#   HTTPS_PROXY=http://proxy.modelarts.com:80.  Measured through that proxy:
#       modelscope.cn   ~13.6 MB/s   (Range supported -> curl -C - resumes)
#       hf-mirror.com   ~ 1.8 MB/s
#   3-way concurrency gave 12.7 MB/s aggregate — the proxy caps at ~13 MB/s, so
#   this script downloads strictly sequentially.  Python HF/ModelScope clients
#   are NOT used: python requests through that proxy dies with
#   "Tunnel connection failed: 503" (see pull_models_remote.sh).
#
#   Repo:  Qwen/Qwen3-VL-32B-Instruct   (~62.14 GiB, 14 shards)
#   Dest:  $REPO_ROOT/llm_models/Qwen3-VL-32B-Instruct
#
# ROBUSTNESS (learned the hard way — an earlier killed run left an orphaned
# curl that wrote to the same file as the restarted run, corrupting
# model-00001-of-00014: 7.71 GB instead of 4.93 GB):
#   * flock: only ONE instance may ever write into $DEST.
#   * every file is downloaded to "<name>.part" and only renamed to its final
#     name after its size matches the API listing exactly.
#   * a final file whose size != expected is considered corrupt and deleted
#     (never resumed); resume only happens on a trusted ".part" prefix.
#   * small files (configs/tokenizer) are fetched first so AutoProcessor/AutoConfig
#     work long before the ~62 GiB of shards finish.
#
# Safe to re-run / resume at any time.
# =============================================================================
set -uo pipefail

REPO_ROOT="${REPO_ROOT:-/home/ma-user/work/rvqa}"
DEST="${DEST:-$REPO_ROOT/llm_models/Qwen3-VL-32B-Instruct}"
MS_MODEL="${MS_MODEL:-Qwen/Qwen3-VL-32B-Instruct}"
MS_REV="${MS_REV:-master}"

export HTTPS_PROXY="${HTTPS_PROXY:-http://proxy.modelarts.com:80}"
export HTTP_PROXY="${HTTP_PROXY:-http://proxy.modelarts.com:80}"
export https_proxy="$HTTPS_PROXY"
export http_proxy="$HTTP_PROXY"

LIST_API="https://modelscope.cn/api/v1/models/${MS_MODEL}/repo/files?Revision=${MS_REV}&Recursive=true"
FILE_API="https://modelscope.cn/api/v1/models/${MS_MODEL}/repo?Revision=${MS_REV}&FilePath="

echo "==> DEST = $DEST"
echo "==> repo = $MS_MODEL @ $MS_REV"
echo "==> proxy = $HTTPS_PROXY"

mkdir -p "$DEST"

# --- single-writer lock -----------------------------------------------------
LOCKFILE="$DEST/.download.lock"
exec 9>"$LOCKFILE"
if ! flock -n 9; then
  echo "FATAL: another downloader already holds $LOCKFILE — refusing to run twice" >&2
  exit 3
fi

LIST_JSON="$DEST/.modelscope_files.json"
MANIFEST="$DEST/.download_manifest.tsv"

echo "==> listing files"
curl -sSL --fail --retry 5 --retry-delay 5 -m 90 "$LIST_API" -o "$LIST_JSON" || {
  echo "FATAL: could not list $MS_MODEL"; exit 1; }

# "<size>\t<path>", directories dropped, sorted by size ASCENDING so the small
# config/tokenizer files land first.
python3 - "$LIST_JSON" <<'PY' | sort -n -k1,1 > "$MANIFEST"
import json, sys
data = json.load(open(sys.argv[1]))
files = data.get("Data", {}).get("Files") or data.get("Data", [])
if isinstance(files, dict):
    files = files.get("Files", [])
for f in files:
    path = f.get("Path") or f.get("Name") or ""
    if not path or f.get("Type") == "tree" or path.endswith("/"):
        continue
    print(f'{int(f.get("Size") or 0)}\t{path}')
PY

TOTAL_FILES=$(wc -l < "$MANIFEST")
TOTAL_BYTES=$(awk -F'\t' '{s+=$1} END{print s+0}' "$MANIFEST")
python3 -c "print(f'    {$TOTAL_FILES} files, total {$TOTAL_BYTES/2**30:.2f} GiB')"

# --- one file, verified -----------------------------------------------------
# returns 0 on success (final file present, exact size)
fetch_one () {
  local size="$1" path="$2"
  local out="$DEST/$path" part="$DEST/$path.part"

  if [[ -f "$out" && "$(stat -c %s "$out")" == "$size" ]]; then
    echo "    SKIP (complete) $path"
    return 0
  fi
  if [[ -f "$out" ]]; then
    echo "    removing size-mismatched file ($(stat -c %s "$out") != $size) $path"
    rm -f "$out"
  fi
  # a trusted .part may only be resumed if it is a proper prefix (smaller)
  if [[ -f "$part" && "$(stat -c %s "$part")" -ge "$size" ]]; then
    echo "    .part is not a valid prefix ($(stat -c %s "$part") >= $size) — restarting $path"
    rm -f "$part"
  fi

  local attempt
  for attempt in 1 2 3; do
    echo "    GET $path  ($(( size / 1048576 )) MiB, attempt $attempt)"
    curl -L --fail --retry 8 --retry-delay 5 --retry-all-errors \
         -C - --speed-time 240 --speed-limit 20480 \
         --no-progress-meter -o "$part" "${FILE_API}${path}" || true
    local have
    have=$(stat -c %s "$part" 2>/dev/null || echo 0)
    if [[ "$have" == "$size" ]]; then
      mv -f "$part" "$out"
      return 0
    fi
    echo "    size mismatch after attempt $attempt: $have != $size"
    if [[ "$have" -gt "$size" ]]; then
      rm -f "$part"   # corrupted/overshot — cannot resume
    fi
  done
  echo "    FATAL: could not fetch $path"
  return 1
}

# --- main loop --------------------------------------------------------------
DONE_BYTES=0
N=0
FAILED=0
while IFS=$'\t' read -r size path; do
  N=$((N + 1))
  echo "[$N/$TOTAL_FILES] $path"
  if fetch_one "$size" "$path"; then
    DONE_BYTES=$((DONE_BYTES + size))
  else
    FAILED=$((FAILED + 1))
  fi
  python3 -c "print(f'      -> {$DONE_BYTES/2**30:.2f}/{$TOTAL_BYTES/2**30:.2f} GiB verified')"
done < "$MANIFEST"

# --- full verification ------------------------------------------------------
echo
echo "==> verifying every file against the API listing"
python3 - "$MANIFEST" "$DEST" <<'PY'
import os, sys
bad = []
n = 0
total = 0
expected_total = 0
for line in open(sys.argv[1]):
    size, path = line.rstrip("\n").split("\t", 1)
    size = int(size)
    expected_total += size
    n += 1
    f = os.path.join(sys.argv[2], path)
    if not os.path.isfile(f) or os.path.getsize(f) != size:
        bad.append((path, os.path.getsize(f) if os.path.isfile(f) else -1, size))
    else:
        total += size
print(f"    files ok: {n - len(bad)}/{n}   bytes: {total/2**30:.2f} / {expected_total/2**30:.2f} GiB")
for p, got, want in bad[:10]:
    print(f"      BAD {p}: got {got} want {want}")
if bad:
    print("    RESULT: INCOMPLETE — re-run this script to resume")
    sys.exit(1)
print("    RESULT: COMPLETE")
PY
rc=$?
[[ $FAILED -eq 0 ]] || rc=1
exit $rc
