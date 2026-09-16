#!/usr/bin/env bash
# =============================================================================
# pull_models_remote.sh  —  RUN ON THE REMOTE Ascend box (ModelArts)
# =============================================================================
# Pulls the three model weights from HuggingFace into the exact paths the
# migrated configs expect.
#
# WHY curl AND NOT `hf download` (diagnosed on this box, 2026-09-16):
#   ModelArts forces HTTPS_PROXY=http://proxy.modelarts.com:80.
#   - curl through that proxy            -> 200, 13.6 MB/s   OK
#   - raw socket CONNECT through proxy   -> "200 Connection established"  OK
#   - python requests/urllib3 via proxy  -> "Tunnel connection failed: 503 Service Unavailable"
#   So EVERY python-based HF client (huggingface_hub, `hf download`,
#   snapshot_download) fails on this box, while curl works fine.
#   => this script shells out to curl for the transfers.
#
# ALSO MEASURED (why we pull here instead of uploading from the origin):
#   origin -> remote upload (ssh/rsync)      11.3 MB/s
#   remote <- huggingface.co (1 stream)      13.6 MB/s
#   remote <- huggingface.co (4x / 8x conc.) 11.2 / 13.5 MB/s  <- no speedup, ~13 MB/s cap
#   remote <- hf-mirror.com                   3.8 MB/s
#   remote <- modelscope.cn                  12.7 MB/s
#   Total to move also drops from ~110 GB to ~94 GiB, because the local
#   MiniCPM-V-2_6 dir carries duplicate weight formats.
#
# Usage:
#   bash pull_models_remote.sh                    # all three
#   bash pull_models_remote.sh 7b                 # just Qwen2.5-VL-7B (unblocks the smoke test)
#   bash pull_models_remote.sh 7b 32b minicpm     # explicit subset
#   HF_TOKEN=hf_xxx bash pull_models_remote.sh minicpm
#
# MiniCPM-V-2_6 is gated=auto: accept the licence at
#   https://huggingface.co/openbmb/MiniCPM-V-2_6
# then export HF_TOKEN before running.
# =============================================================================
set -uo pipefail

# All roots default to empty; export them for your machine.
REPO_ROOT="${REPO_ROOT:-}"
PY="${PY:-}"
HF_BASE="${HF_BASE:-https://huggingface.co}"
HF_TOKEN="${HF_TOKEN:-}"

export HTTPS_PROXY="${HTTPS_PROXY:-http://proxy.modelarts.com:80}"
export HTTP_PROXY="${HTTP_PROXY:-http://proxy.modelarts.com:80}"
export https_proxy="$HTTPS_PROXY"
export http_proxy="$HTTP_PROXY"

SELECT=("$@")
[[ ${#SELECT[@]} -eq 0 ]] && SELECT=(7b 32b minicpm)

want () { local x; for x in "${SELECT[@]}"; do [[ "$x" == "$1" ]] && return 0; done; return 1; }

echo "==> REPO_ROOT = $REPO_ROOT"
echo "==> proxy     = $HTTPS_PROXY   (curl path, NOT python)"
echo "==> HF_TOKEN  = $([[ -n "$HF_TOKEN" ]] && echo set || echo '(not set — gated repos will fail)')"

mkdir -p "$REPO_ROOT/llm_models" "$REPO_ROOT/models/llm_models"

# -----------------------------------------------------------------------------
# curl-based repo fetcher
# -----------------------------------------------------------------------------
fetch_repo () {
  local repo="$1" dest="$2" gated="$3"
  mkdir -p "$dest"

  local auth=()
  [[ -n "$HF_TOKEN" ]] && auth=(-H "Authorization: Bearer $HF_TOKEN")

  local api="$HF_BASE/api/models/$repo?blobs=true"
  echo "    listing $api"
  if ! curl -sL --fail --retry 3 --max-time 60 "${auth[@]}" "$api" -o /tmp/_hf_list.json; then
    echo "    FAILED to list $repo (gated=$gated)"
    return 1
  fi
  cp /tmp/_hf_list.json "$dest/.hf_api.json" 2>/dev/null || true

  REPO="$repo" DEST="$dest" HF_BASE="$HF_BASE" HF_TOKEN="$HF_TOKEN" \
  "$PY" - <<'PYEOF'
import json, os, subprocess, sys, time, pathlib

repo   = os.environ["REPO"]
dest   = pathlib.Path(os.environ["DEST"])
base   = os.environ["HF_BASE"].rstrip("/")
token  = os.environ["HF_TOKEN"]

with open("/tmp/_hf_list.json") as fh:
    data = json.load(fh)

siblings = [s for s in data.get("siblings", []) if s.get("rfilename")]
if not siblings:
    print("    no files listed; aborting"); sys.exit(1)

hdr = ["-H", f"Authorization: Bearer {token}"] if token else []
total = sum(s.get("size") or 0 for s in siblings)
print(f"    {len(siblings)} files, expected {total/2**30:.1f} GiB")

t0 = time.time()
done = 0
for s in siblings:
    name = s["rfilename"]
    size = s.get("size")
    out  = dest / name
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.exists():
        have = out.stat().st_size
        if size and have == size:
            print(f"    SKIP (complete) {name}"); done += have; continue
        if size and have < size:
            print(f"    RESUME {name}  {have}/{size}")

    url = f"{base}/{repo}/resolve/main/{name}"
    cmd = ["curl", "-L", "--fail", "--retry", "5", "--retry-delay", "5",
           "--retry-all-errors", "-C", "-", "--speed-time", "120",
           "--speed-limit", "10240",
           "--progress-bar", "-o", str(out)] + hdr + [url]
    t1 = time.time()
    r = subprocess.run(cmd)
    if r.returncode != 0:
        # a fully-complete file makes -C - return 33/416; treat that as success
        if size and out.exists() and out.stat().st_size == size:
            print(f"    OK (was already complete) {name}")
        else:
            print(f"    FAILED {name} (curl rc={r.returncode})"); sys.exit(2)
    el = max(time.time() - t1, 1e-6)
    dl = (out.stat().st_size if out.exists() else 0)
    done += (size or dl)
    spd = (size or dl) / el / 2**20 if size else 0
    print(f"    OK {name}  {dl/2**20:8.1f} MiB  {el:5.1f}s  {spd:6.2f} MiB/s  "
          f"[{done/2**30:.1f}/{total/2**30:.1f} GiB]")

print(f"    finished in {time.time()-t0:.0f}s")
PYEOF
}

run_one () {
  local tag="$1" repo="$2" dest="$3" gated="$4" note="$5"
  want "$tag" || return 0
  echo
  echo "=============================================================="
  echo "==> [$tag] $repo"
  echo "    -> $dest"
  [[ -n "$note" ]] && echo "    NOTE: $note"
  echo "=============================================================="
  if fetch_repo "$repo" "$dest" "$gated"; then
    echo "    DONE  size=$(du -sh "$dest" 2>/dev/null | cut -f1)"
  else
    echo "    FAILED: $repo"
    return 1
  fi
}

FAILED=()
run_one 7b      "Qwen/Qwen2.5-VL-7B-Instruct"  "$REPO_ROOT/llm_models/Qwen2.5-VL-7B-Instruct"        "false" "" \
  || FAILED+=("7b")
run_one 32b     "Qwen/Qwen2.5-VL-32B-Instruct" "$REPO_ROOT/models/llm_models/Qwen2.5-VL-32B-Instruct" "false" "" \
  || FAILED+=("32b")
run_one minicpm "openbmb/MiniCPM-V-2_6"        "$REPO_ROOT/models/llm_models/MiniCPM-V-2_6"           "auto"  "gated — needs HF_TOKEN + accepted licence" \
  || FAILED+=("minicpm")

# -----------------------------------------------------------------------------
# verification
# -----------------------------------------------------------------------------
echo
echo "=============================================================="
echo "==> verification"
echo "=============================================================="
"$PY" - "$REPO_ROOT" <<'PYEOF'
import pathlib, sys, json
root = pathlib.Path(sys.argv[1])
targets = {
    "Qwen2.5-VL-7B-Instruct":  root / "llm_models/Qwen2.5-VL-7B-Instruct",
    "Qwen2.5-VL-32B-Instruct": root / "models/llm_models/Qwen2.5-VL-32B-Instruct",
    "MiniCPM-V-2_6":           root / "models/llm_models/MiniCPM-V-2_6",
}
allok = True
for name, d in targets.items():
    if not d.exists():
        print(f"  --       {name:26s} not pulled"); continue
    api = d / ".hf_api.json"
    st = sorted(d.rglob("*.safetensors"))
    big = [p for p in st if p.stat().st_size > 100_000_000]
    cfg = (d / "config.json").exists()
    total = sum(p.stat().st_size for p in st)
    # completeness check against the API listing
    bad = []
    if api.exists():
        want = {s["rfilename"]: s.get("size") for s in json.loads(api.read_text()).get("siblings", [])}
        for fn, sz in want.items():
            f = d / fn
            if not f.exists() or (sz and f.stat().st_size != sz):
                bad.append(fn)
    status = "OK " if (big and cfg and not bad) else "BAD"
    if not (big and cfg and not bad): allok = False
    print(f"  {status}      {name:26s} safetensors={len(st):2d} ({total/2**30:6.1f} GiB) "
          f"config={cfg} incomplete={len(bad)}")
    if bad:
        print(f"           first missing/short: {bad[:3]}")
print()
print("RESULT:", "ALL GOOD" if allok else "CHECK ABOVE")
PYEOF

echo
if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "!! failed: ${FAILED[*]}"
  echo "   MiniCPM-V-2_6 tip: accept the licence at"
  echo "     https://huggingface.co/openbmb/MiniCPM-V-2_6"
  echo "   then:  HF_TOKEN=hf_xxx bash \$0 minicpm"
  exit 1
fi
echo "==> done. Next on this box:"
echo "  bash $REPO_ROOT/rvqa_bundle/03_remote_unpack.sh    # unpack code+data, rewrite paths"
echo "  bash $REPO_ROOT/rvqa_bundle/setup_envs_ascend.sh   # conda clone PyTorch-2.7.1 + pins"
echo "  bash $REPO_ROOT/rvqa_bundle/smoke_test_ascend.sh   # NPU smoke test"
