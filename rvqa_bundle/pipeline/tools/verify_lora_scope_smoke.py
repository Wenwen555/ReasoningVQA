#!/usr/bin/env python3
"""CPU-only end-to-end smoke verification for ``--lora-scope`` in the trainer.

This never touches a GPU: CUDA is hard-disabled for the process and every
``torch.cuda.*`` call the trainer makes is stubbed.  The transformers loader and
processor are monkeypatched to return a tiny CPU module tree whose module names
are exactly the real inventory target names, so the *actual* target-selection,
LoRA-wrapping, family-completeness, param-group and end-of-run integrity code
paths execute end to end and reach the first optimizer step.

Run with::

    CUDA_VISIBLE_DEVICES="" RVQA_SMOKE_INV_QWEN25VL32B=/path/to/inventory.json \
        RVQA_SMOKE_MODEL_QWEN25VL32B=/path/to/model \
        python tools/verify_lora_scope_smoke.py
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import types
from collections import Counter
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "pipelines" / "train" / "train_qwen_vl_sft.py"


def _env_path(name: str) -> Path | None:
    """Return ``Path(env[name])`` when set, else ``None``."""
    value = os.environ.get(name, "").strip()
    return Path(value) if value else None


# (label, inventory env var, model env var); empty pairs are skipped.
SMOKE_DATASETS = [
    ("qwen25vl32b", "RVQA_SMOKE_INV_QWEN25VL32B", "RVQA_SMOKE_MODEL_QWEN25VL32B"),
    ("minicpmv26", "RVQA_SMOKE_INV_MINICPMV26", "RVQA_SMOKE_MODEL_MINICPMV26"),
]

import torch  # noqa: E402
import transformers  # noqa: E402
from PIL import Image  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

# CUDA is disabled; neutralise the handful of cuda calls the trainer issues.
torch.cuda.reset_peak_memory_stats = lambda *a, **k: None
torch.cuda.max_memory_allocated = lambda *a, **k: 0
torch.cuda.manual_seed_all = lambda *a, **k: None

import peft  # noqa: E402

# The tiny base model is a plain nn.Module, so stub the two trainer calls that
# only make sense for a real PreTrainedModel.  Neither is under test here.
peft.PeftModel.gradient_checkpointing_enable = lambda self, **k: None
peft.PeftModel.enable_input_require_grads = lambda self: None


class DummyConfig(dict):
    def __init__(self):
        super().__init__(tie_word_embeddings=False, model_type="dummy")

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


class DummyBase(torch.nn.Module):
    """Module tree whose leaves are exactly the inventory target linear names."""

    def __init__(self, names):
        super().__init__()
        self.config = DummyConfig()
        for full in names:
            parts = full.split(".")
            cur = self
            for index, part in enumerate(parts):
                if index == len(parts) - 1:
                    setattr(cur, part, torch.nn.Linear(4, 4))
                else:
                    child = getattr(cur, part, None)
                    if child is None:
                        child = torch.nn.Module()
                        setattr(cur, part, child)
                    cur = child

    def forward(self, input_ids=None, labels=None, use_cache=False, data=None, **kwargs):
        # MiniCPM-V forwards the whole processor batch through ``data``.
        if data is not None:
            input_ids = data["input_ids"]
        loss = input_ids.float().sum() * 0.0
        for param in self.parameters():
            if param.requires_grad:
                loss = loss + param.float().sum()
        return types.SimpleNamespace(loss=loss)

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return {}

    @property
    def device(self):
        return torch.device("cpu")


class FakeProcessor:
    def __init__(self):
        # Real MiniCPM processors expose the chat template on the tokenizer.
        self.tokenizer = self

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs):
        return "PROMPT" if add_generation_prompt else "PROMPT_ANSWER"

    def __call__(self, text=None, images=None, padding=True, return_tensors="pt", **kwargs):
        value = text[0] if isinstance(text, list) else text
        ids = [1, 2, 3] if value == "PROMPT" else [1, 2, 3, 4, 5]
        tensor = torch.tensor([ids], dtype=torch.long)
        return transformers.BatchFeature(
            data={"input_ids": tensor, "attention_mask": torch.ones_like(tensor)}
        )

    def save_pretrained(self, path):
        pass


def load_trainer():
    spec = importlib.util.spec_from_file_location("train_qwen_vl_sft_under_test", TRAINER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_trainer(names, model_dir, inventory_path, scope, workdir):
    workdir.mkdir(parents=True, exist_ok=True)
    base = DummyBase(names)
    transformers.AutoModel.from_pretrained = staticmethod(lambda *a, **k: base)
    if getattr(transformers, "AutoModelForImageTextToText", None) is not None:
        transformers.AutoModelForImageTextToText.from_pretrained = staticmethod(lambda *a, **k: base)
    transformers.AutoProcessor.from_pretrained = staticmethod(lambda *a, **k: FakeProcessor())

    image_path = workdir / "image.png"
    Image.new("RGB", (8, 8), (255, 0, 0)).save(image_path)
    manifest = workdir / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"sample_id": "s1", "image_path": str(image_path), "question": "q", "sft_target": "a"}) + "\n",
        encoding="utf-8",
    )
    output = workdir / f"out-{scope or 'default'}"
    sys.argv = [
        "train_qwen_vl_sft.py",
        "--model", str(model_dir),
        "--manifest", str(manifest),
        "--inventory", str(inventory_path),
        "--output", str(output),
        "--max-rows", "1",
        "--gradient-accumulation", "1",
    ]
    if scope is not None:
        sys.argv += ["--lora-scope", scope]
    load_trainer().main()
    return output


def classify_adapter(adapter_dir, family_by_name):
    state = load_file(adapter_dir / "adapter_model.safetensors")
    modules = Counter()
    families = Counter()
    for key in state:
        module = key
        for prefix in ("base_model.model.", "base_model."):
            if module.startswith(prefix):
                module = module[len(prefix):]
        for suffix in (".lora_A.weight", ".lora_B.weight", ".lora_embedding_A", ".lora_embedding_B"):
            if module.endswith(suffix):
                module = module[: -len(suffix)]
                break
        modules[module] += 1
        families[family_by_name.get(module, "UNKNOWN")] += 1
    return modules, families


def grouped_by_suffix(modules):
    groups = Counter()
    for name in modules:
        parts = name.rsplit(".", 2)
        groups[".".join(parts[-2:]) if len(parts) >= 2 else name] += 1
    return dict(sorted(groups.items()))


def check_dataset(label, inventory_path, model_dir, work):
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    names = [item["name"] for item in inventory["targets"]]
    family_by_name = {item["name"]: item["family"] for item in inventory["targets"]}
    report = {
        "inventory": str(inventory_path),
        "target_counts": dict(Counter(family_by_name.values())),
    }

    out = run_trainer(names, model_dir, inventory_path, "language", work / f"{label}-lang")
    result = json.loads((out / "result.json").read_text(encoding="utf-8"))
    modules, families = classify_adapter(out / "adapter", family_by_name)
    report["language_scope"] = {
        "result_trainable_groups": result["trainable_tensor_groups"],
        "result_optimizer_steps": result["optimizer_steps"],
        "result_status": result["status"],
        "adapter_lora_modules": len(modules),
        "adapter_families": dict(families),
        "adapter_groups_by_suffix": grouped_by_suffix(modules),
        "vision_or_merger_modules": sorted(
            name for name in modules if family_by_name.get(name) in ("visual_blocks", "merger")
        ),
    }

    out = run_trainer(names, model_dir, inventory_path, "all", work / f"{label}-all")
    result = json.loads((out / "result.json").read_text(encoding="utf-8"))
    _, families_all = classify_adapter(out / "adapter", family_by_name)
    report["all_scope"] = {
        "result_trainable_groups": result["trainable_tensor_groups"],
        "result_optimizer_steps": result["optimizer_steps"],
        "result_status": result["status"],
        "adapter_families": dict(families_all),
    }
    return report, family_by_name


def main():
    datasets = [
        (label, _env_path(inv_env), _env_path(model_env))
        for label, inv_env, model_env in SMOKE_DATASETS
        if _env_path(inv_env) and _env_path(model_env)
    ]
    if not datasets:
        raise SystemExit(
            "no smoke dataset configured; set RVQA_SMOKE_INV_QWEN25VL32B and "
            "RVQA_SMOKE_MODEL_QWEN25VL32B (optionally the MiniCPM pair)"
        )

    report = {}
    family_by_qwen = None

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        for label, inventory_path, model_dir in datasets:
            report[label], family_by_qwen = check_dataset(label, inventory_path, model_dir, work)

        # --- Default (no --lora-scope) must equal the historical all-family run
        label, inventory_path, model_dir = datasets[0]
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
        family_by_name = {item["name"]: item["family"] for item in inventory["targets"]}
        names = [item["name"] for item in inventory["targets"]]
        out = run_trainer(names, model_dir, inventory_path, None, work / "default")
        result = json.loads((out / "result.json").read_text(encoding="utf-8"))
        run_config = json.loads((out / "run_config.json").read_text(encoding="utf-8"))
        _, families_default = classify_adapter(out / "adapter", family_by_name)
        report["default_scope"] = {
            "run_config_lora_scope": run_config["lora_scope"],
            "result_lora_scope": result["lora_scope"],
            "result_trainable_groups": result["trainable_tensor_groups"],
            "adapter_families": dict(families_default),
            "matches_all_scope": (
                result["trainable_tensor_groups"]
                == report[label]["all_scope"]["result_trainable_groups"]
            ),
        }

        # --- Preflight catches a prefix mismatch (modern inventory, legacy model)
        qwen_model = datasets[0][2]
        synthetic = work / "mismatched_inventory.json"
        sha = __import__("hashlib").sha256((qwen_model / "config.json").read_bytes()).hexdigest()
        synthetic.write_text(json.dumps({
            "status": "passed",
            "model_config_sha256": sha,
            "model_type": "qwen2_5_vl",
            "targets": [
                {"name": "model.language_model.layers.0.self_attn.q_proj", "family": "language"},
                {"name": "model.language_model.layers.0.mlp.gate_proj", "family": "language"},
                {"name": "model.visual.blocks.0.attn.qkv", "family": "visual_blocks"},
                {"name": "model.visual.merger.mlp.0", "family": "merger"},
            ],
        }) + "\n", encoding="utf-8")
        qwen_names = [item["name"] for item in json.loads(datasets[0][1].read_text(encoding="utf-8"))["targets"]]
        try:
            run_trainer(qwen_names, qwen_model, synthetic, "language", work / "preflight")
            report["preflight"] = "NO ERROR RAISED (unexpected)"
        except RuntimeError as exc:
            report["preflight"] = str(exc)

    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
