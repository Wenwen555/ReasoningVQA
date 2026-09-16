#!/usr/bin/env python3
"""Build a ReasoningVQA LoRA-target inventory for a local model.

This derives the inventory schema from
``pipelines/train/train_qwen_vl_sft.py`` which requires:

  * ``status`` == "passed"
  * ``model_config_sha256`` == sha256(raw bytes of <model>/config.json)
  * ``targets`` : list of {"name": <module name>, "family": one of
    "language" | "visual_blocks" | "merger"}

The trainer builds ``LoraConfig(target_modules=sorted(names))`` and splits the
LoRA parameters into three AdamW groups by ``family``.  It fails if any of the
three families is empty.

No weights are loaded.  Parameter/module names and shapes are read straight
from the safetensors headers (metadata only).  Works for bf16 and for AWQ
(int4) checkpoints because AWQ quantized linears are exposed as
``<module>.qweight`` / ``.qzeros`` / ``.scales`` and are mapped back to their
logical module names.

Usage:
    python tools/build_model_inventory.py --model <model_dir> --output <out.json>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

# Suffixes that terminate a tensor key which belongs to a *quantized* linear
# (AWQ / GPTQ style).  ".weight"/".bias" are handled separately.
QUANT_SUFFIXES = (".qweight", ".qzeros", ".scales", ".g_idx", ".weight_format")

SCHEMA_VERSION = "reasoningvqa-bench-mm-lora-inventory-v1"

# --------------------------------------------------------------------------- #
# Architecture-specific LoRA target selection.
#
# Each family maps to a list of fully-anchored regexes matched against the
# *logical module name* (tensor suffix stripped).  The Qwen regexes accept both
# the transformers<5 naming (model.layers / visual.*) and the transformers>=5
# naming (model.language_model.layers / model.visual.*), so a single spec covers
# Qwen2.5-VL and Qwen3-VL.
# --------------------------------------------------------------------------- #
QWEN_SPEC = {
    # attention + MLP projections of the language model
    "language": [
        r"(?:^|\.)model\.(?:language_model\.)?layers\.\d+\.self_attn\.(?:q_proj|k_proj|v_proj|o_proj)$",
        r"(?:^|\.)model\.(?:language_model\.)?layers\.\d+\.mlp\.(?:gate_proj|up_proj|down_proj)$",
    ],
    # vision transformer blocks (attention + MLP projections)
    "visual_blocks": [
        r"(?:^|\.)visual\.blocks\.\d+\.attn\.(?:qkv|proj)$",
        r"(?:^|\.)visual\.blocks\.\d+\.mlp\.(?:gate_proj|up_proj|down_proj|linear_fc1|linear_fc2)$",
    ],
    # projector / merger that maps vision features into the LM embedding space
    "merger": [
        r"(?:^|\.)visual\.merger\.(?:mlp\.\d+|linear_fc1|linear_fc2)$",
        r"(?:^|\.)visual\.deepstack_merger_list\.\d+\.(?:linear_fc1|linear_fc2)$",
    ],
}

MINICPM_SPEC = {
    # language model lives under llm.model.layers
    "language": [
        r"(?:^|\.)llm\.model\.layers\.\d+\.self_attn\.(?:q_proj|k_proj|v_proj|o_proj)$",
        r"(?:^|\.)llm\.model\.layers\.\d+\.mlp\.(?:gate_proj|up_proj|down_proj)$",
    ],
    # SigLIP vision tower lives under vpm.encoder.layers
    "visual_blocks": [
        r"(?:^|\.)vpm\.encoder\.layers\.\d+\.self_attn\.(?:q_proj|k_proj|v_proj|out_proj)$",
        r"(?:^|\.)vpm\.encoder\.layers\.\d+\.mlp\.(?:fc1|fc2)$",
    ],
    # Perceiver resampler is the visual->language merger
    "merger": [
        r"(?:^|\.)resampler\.(?:kv_proj|proj|attn\.(?:q_proj|k_proj|v_proj|out_proj)|fc1|fc2)$",
    ],
}

FAMILIES = ("language", "visual_blocks", "merger")

SPECS = {
    "qwen2_5_vl": QWEN_SPEC,
    "qwen2_vl": QWEN_SPEC,
    "qwen3_vl": QWEN_SPEC,
    "minicpmv": MINICPM_SPEC,
    # MiniCPM's config also declares architecture "MiniCPMV" but model_type is
    # "minicpmv"; keep an alias in case a checkpoint reports it directly.
    "minicpm_v": MINICPM_SPEC,
}


# --------------------------------------------------------------------------- #
# safetensors header reading (stdlib only)
# --------------------------------------------------------------------------- #
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_header(path: Path) -> dict:
    with path.open("rb") as handle:
        raw = handle.read(8)
        if len(raw) != 8:
            raise ValueError(f"{path}: truncated safetensors header")
        size = int.from_bytes(raw, "little")
        if size <= 0 or size > 200_000_000:
            raise ValueError(f"{path}: implausible safetensors header size {size}")
        return json.loads(handle.read(size))


def collect_tensors(model: Path) -> dict[str, tuple]:
    """Return {tensor_name: shape} from safetensors headers (no data read)."""
    shards = sorted(model.glob("*.safetensors"))
    if not shards:
        raise ValueError(f"{model}: no *.safetensors shards found")
    tensors: dict[str, tuple] = {}
    for shard in shards:
        header = _read_header(shard)
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            if isinstance(meta, dict) and "shape" in meta:
                tensors[name] = tuple(meta["shape"])
    return tensors


def module_of(tensor_name: str) -> str | None:
    """Map a tensor key to the logical module name it belongs to."""
    if tensor_name.endswith(".weight") or tensor_name.endswith(".bias"):
        return tensor_name.rsplit(".", 1)[0]
    for suffix in QUANT_SUFFIXES:
        if tensor_name.endswith(suffix):
            return tensor_name[: -len(suffix)]
    return None


def build_modules(tensors: dict[str, tuple]) -> dict[str, dict]:
    modules: dict[str, dict] = {}
    for name, shape in tensors.items():
        module = module_of(name)
        if module is None:
            continue
        entry = modules.setdefault(module, {})
        if name.endswith(".weight"):
            entry["weight_shape"] = shape
        elif name.endswith(".qweight"):
            entry["qweight_shape"] = shape
        elif name.endswith(".scales"):
            entry["scales_shape"] = shape
    return modules


def linear_dims(entry: dict, bits: int | None):
    """Return (in_features, out_features, quantized) or None if not linear."""
    weight_shape = entry.get("weight_shape")
    if weight_shape is not None and len(weight_shape) == 2:
        return weight_shape[0], weight_shape[1], False
    qweight_shape = entry.get("qweight_shape")
    if qweight_shape is not None and len(qweight_shape) == 2:
        out_features = int(qweight_shape[1] * 32 // bits) if bits else None
        return qweight_shape[0], out_features, True
    return None


def load_config(model: Path) -> dict:
    with (model / "config.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_spec(config: dict) -> tuple[str, dict]:
    model_type = str(config.get("model_type", "")).lower()
    if model_type in SPECS:
        return model_type, SPECS[model_type]
    # fall back on declared architectures
    for arch in config.get("architectures", []) or []:
        low = str(arch).lower()
        if "qwen" in low:
            return model_type or "qwen", QWEN_SPEC
        if "minicpm" in low:
            return model_type or "minicpmv", MINICPM_SPEC
    raise ValueError(
        f"unsupported model_type={model_type!r} architectures={config.get('architectures')!r}; "
        f"known={sorted(SPECS)}"
    )


def classify(modules: dict[str, dict], spec: dict, bits: int | None):
    picked: dict[str, str] = {}
    for module in modules:
        matches = [family for family, patterns in spec.items()
                   if any(re.search(p, module) for p in patterns)]
        if not matches:
            continue
        if len(matches) != 1:
            raise ValueError(f"module {module!r} matches multiple families {matches}")
        dims = linear_dims(modules[module], bits)
        if dims is None:
            # guard against accidentally matching a non-linear module (norm etc.)
            continue
        picked[module] = matches[0]
    return picked


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model = args.model.resolve()
    if not (model / "config.json").is_file():
        raise SystemExit(f"missing {model / 'config.json'}")

    config = load_config(model)
    config_sha = sha256_file(model / "config.json")
    model_type, spec = resolve_spec(config)
    bits = None
    quant_config = config.get("quantization_config") or {}
    if quant_config.get("bits"):
        bits = int(quant_config["bits"])

    tensors = collect_tensors(model)
    modules = build_modules(tensors)
    picked = classify(modules, spec, bits)

    targets = []
    family_counts = {family: 0 for family in FAMILIES}
    for module in sorted(picked):
        family = picked[module]
        dims = linear_dims(modules[module], bits)
        in_features, out_features, quantized = dims
        targets.append({
            "name": module,
            "family": family,
            "in_features": in_features,
            "out_features": out_features,
            "quantized": quantized,
        })
        family_counts[family] += 1

    # hard contract: the trainer fails when any family is empty
    empty = [family for family in FAMILIES if family_counts[family] == 0]
    if empty:
        raise SystemExit(
            f"ERROR: empty LoRA target families {empty} for {model} "
            f"(model_type={model_type}, target_module_count={len(targets)}) — "
            "the trainer requires all of language/visual_blocks/merger to be non-empty"
        )

    inventory = {
        "status": "passed",
        "schema_version": SCHEMA_VERSION,
        "model": str(model),
        "model_type": model_type,
        "architectures": list(config.get("architectures", []) or []),
        "model_config_sha256": config_sha,
        "target_module_count": len(targets),
        "family_counts": family_counts,
        "targets": targets,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"[build_model_inventory] model={model}")
    print(f"[build_model_inventory] model_type={model_type} config_sha256={config_sha}")
    print(f"[build_model_inventory] total targets={len(targets)}")
    for family in FAMILIES:
        examples = [t["name"] for t in targets if t["family"] == family][:3]
        print(f"[build_model_inventory]   {family:14s} {family_counts[family]:5d}   e.g. {examples}")
    print(f"[build_model_inventory] wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
