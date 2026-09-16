#!/usr/bin/env python3
"""Verify a ReasoningVQA LoRA inventory against a local model without loading weights.

Checks (all metadata only):
  * required fields present and ``status == "passed"``
  * ``model_config_sha256`` matches sha256(raw bytes of <model>/config.json)
  * every target ``family`` is one of language / visual_blocks / merger
  * every target ``name`` is backed by a real state-dict tensor
    (``<name>.weight`` or an AWQ/GPTQ quantized suffix)
  * all three families are non-empty
  * the target set equals the set the generator would select (self-consistency)
  * target names are unique and each is a real 2-D linear module

Usage:
    python tools/verify_model_inventory.py --model <model_dir> --inventory <inventory.json>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_model_inventory as bmi  # noqa: E402

REQUIRED = ("status", "model_config_sha256", "targets")
VALID_FAMILIES = set(bmi.FAMILIES)


def fail(msg: str) -> None:
    raise SystemExit(f"FAIL: {msg}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    args = parser.parse_args()

    model = args.model.resolve()
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))

    errors: list[str] = []

    for field in REQUIRED:
        if field not in inventory:
            errors.append(f"missing required field {field!r}")

    if inventory.get("status") != "passed":
        errors.append(f"status is {inventory.get('status')!r}, expected 'passed'")

    actual_sha = bmi.sha256_file(model / "config.json")
    if inventory.get("model_config_sha256") != actual_sha:
        errors.append(
            f"model_config_sha256 mismatch: inventory={inventory.get('model_config_sha256')} "
            f"actual={actual_sha}"
        )

    targets = inventory.get("targets")
    if not isinstance(targets, list) or not targets:
        errors.append("targets must be a non-empty list")
        targets = []

    names = []
    seen = set()
    for item in targets:
        if not isinstance(item, dict) or "name" not in item or "family" not in item:
            errors.append(f"target entry lacks name/family: {item!r}")
            continue
        name = item["name"]
        family = item["family"]
        if family not in VALID_FAMILIES:
            errors.append(f"invalid family {family!r} for {name!r}")
        if name in seen:
            errors.append(f"duplicate target {name!r}")
        seen.add(name)
        names.append(name)

    config = bmi.load_config(model)
    model_type, spec = bmi.resolve_spec(config)
    bits = None
    quant_config = config.get("quantization_config") or {}
    if quant_config.get("bits"):
        bits = int(quant_config["bits"])

    tensors = bmi.collect_tensors(model)
    modules = bmi.build_modules(tensors)

    # every target name must be backed by a real tensor and be a real linear
    for name in names:
        entry = modules.get(name)
        if entry is None:
            errors.append(f"target {name!r} not found in state dict")
            continue
        if bmi.linear_dims(entry, bits) is None:
            errors.append(f"target {name!r} is not a 2-D linear module")

    # self-consistency: same selection the generator makes
    expected = bmi.classify(modules, spec, bits)
    if set(expected) != set(names):
        missing = sorted(set(expected) - set(names))
        extra = sorted(set(names) - set(expected))
        errors.append(f"target set differs from generator selection; missing={missing[:5]} extra={extra[:5]}")

    family_counts = {family: 0 for family in bmi.FAMILIES}
    for name in names:
        family = next((t["family"] for t in targets if t.get("name") == name), None)
        if family in family_counts:
            family_counts[family] += 1
    empty = [family for family in bmi.FAMILIES if family_counts[family] == 0]
    if empty:
        errors.append(f"empty families (trainer would fail): {empty}")

    if errors:
        for err in errors:
            print(f"FAIL: {err}", file=sys.stderr)
        return 1

    print(f"PASS: {args.inventory}")
    print(f"  model={model}")
    print(f"  model_type={model_type}")
    print(f"  model_config_sha256={actual_sha} (matches inventory)")
    print(f"  target_module_count={len(names)}")
    for family in bmi.FAMILIES:
        examples = [t["name"] for t in targets if t["family"] == family][:3]
        print(f"  {family:14s} {family_counts[family]:5d}   e.g. {examples}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
