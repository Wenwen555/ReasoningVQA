#!/usr/bin/env python3
"""Response-only BF16 Qwen-VL MM-LoRA SFT for frozen formal-v0.2.4."""

import argparse
import hashlib
import json
import math
import os
import random
import time
from collections import Counter
from pathlib import Path


OPEN_PROMPT = "Answer the question using the image. Return only the short answer.\nQuestion: {question}"

# MiniCPM-V processors embed images through this textual tag rather than a
# typed-content list; the tag must appear once per image in the user string.
MINICPM_IMAGE_TAG = "(<image>./</image>)"
PROMPT_STYLES = ("auto", "qwen", "minicpm")


def parse_gpus(value):
    """Parse the csv of GPU device ids accepted by ``--gpus``.

    An omitted or empty value yields ``[0]`` so the historical single-process,
    single-GPU ``device_map={"": 0}`` behaviour is the default.
    """
    if value is None:
        return [0]
    devices = []
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            devices.append(int(part))
        except ValueError as exc:
            raise ValueError(f"--gpus must be a csv of integer device ids, got {value!r}") from exc
    if not devices:
        return [0]
    if any(device < 0 for device in devices) or len(set(devices)) != len(devices):
        raise ValueError(f"--gpus must list unique non-negative device ids, got {value!r}")
    return devices


def build_device_plan(gpus, max_memory_per_gpu):
    """Return ``(device_map, max_memory, multi_gpu)`` for the requested GPUs.

    One device keeps the historical ``{"": 0}`` mapping verbatim.  Two or more
    devices shard the model with ``device_map="balanced"`` under an explicit
    per-device memory budget, keyed by the local visible index: the caller pins
    ``CUDA_VISIBLE_DEVICES`` to ``gpus`` so local index ``i`` addresses
    ``gpus[i]``.  This helper is torch-free and asserted in the CPU-only test.
    """
    if len(gpus) <= 1:
        return {"": 0}, None, False
    max_memory = {index: max_memory_per_gpu for index in range(len(gpus))}
    return "balanced", max_memory, True


def build_param_groups(trainable, families, ordered_families, family_lrs):
    """Partition trainable tensors into per-family optimizer groups.

    Returns ``(groups, family_devices)``.  ``family_devices`` records the
    devices each family's parameters occupy so a sharded (``device_map=
    "balanced"``) run can report where every group lives.  Kept torch-free
    (it only reads ``tensor.device``) so it is CPU-unit-testable.
    """
    groups = []
    family_devices = {}
    for family_name in ordered_families:
        if family_name not in family_lrs:
            raise RuntimeError(f"no learning rate configured for trainable family {family_name!r}")
        family_params = [value for name, value in trainable if families[name] == family_name]
        family_devices[family_name] = sorted({str(value.device) for value in family_params})
        groups.append({
            "params": family_params,
            "lr": family_lrs[family_name],
            "family": family_name,
        })
    return groups, family_devices


def detect_prompt_style(model_dir, requested):
    """Resolve the chat/prompt rendering path for ``--prompt-style``.

    ``auto`` keeps the historical Qwen behaviour for every non-MiniCPM model
    and only switches to the MiniCPM string-content path for MiniCPM configs.
    """
    if requested != "auto":
        return requested
    try:
        config = json.loads((Path(model_dir) / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "qwen"
    model_type = str(config.get("model_type", "")).lower()
    architectures = " ".join(str(item).lower() for item in config.get("architectures", []) or [])
    if "minicpm" in model_type or "minicpm" in architectures:
        return "minicpm"
    return "qwen"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def force_default_lora_dispatch(model):
    if getattr(model, "quantization_method", None) is not None or getattr(model.config, "quantization_config", None) is not None:
        return False
    from peft.tuners.lora.model import LoraModel, dispatch_default

    def default_only(config, adapter_name, target, **kwargs):
        value = dispatch_default(target, adapter_name, config, **kwargs)
        if value is None:
            raise TypeError(type(target))
        return value

    LoraModel._create_new_module = staticmethod(default_only)
    return True


def apply_chat_template(processor, messages, add_generation_prompt):
    """Render a chat template, tolerating processors without the helper.

    Transformers 4.40 (the MiniCPM-V-2_6 env) exposes the template only through
    ``processor.tokenizer``; newer processors expose it directly.
    """
    template = getattr(processor, "apply_chat_template", None)
    if template is None:
        template = processor.tokenizer.apply_chat_template
    return template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)


def prepare_minicpm(processor, row, device, max_slice_nums=1, max_input_length=6144):
    import torch
    from PIL import Image

    with Image.open(row["image_path"]) as source:
        image = source.convert("RGB")
    prompt_text = row["sft_prompt"] if "sft_prompt" in row else OPEN_PROMPT.format(question=row["question"])
    # MiniCPM-V's chat template concatenates message['content'] as a plain
    # string, so the user turn must be a string carrying the textual image tag
    # instead of the typed-content list used by Qwen.
    user_content = "\n".join([MINICPM_IMAGE_TAG, prompt_text])
    user = [{"role": "user", "content": user_content}]
    full_messages = user + [{"role": "assistant", "content": str(row["sft_target"])}]
    try:
        prompt = apply_chat_template(processor, user, True)
        full_text = apply_chat_template(processor, full_messages, False)
        shared = {
            "images": [image],
            "max_slice_nums": max_slice_nums,
            "use_image_id": False,
            "return_tensors": "pt",
            "max_length": max_input_length,
        }
        prompt_inputs = processor([prompt], **shared)
        full = processor([full_text], **shared)
    finally:
        image.close()
    prompt_n = int(prompt_inputs["attention_mask"][0].sum())
    full_n = int(full["attention_mask"][0].sum())
    if not 0 < prompt_n < full_n or not torch.equal(
        prompt_inputs["input_ids"][0, :prompt_n], full["input_ids"][0, :prompt_n]
    ):
        raise RuntimeError(f"response boundary mismatch {prompt_n}/{full_n}")
    full["input_ids"] = full["input_ids"].long()
    full["attention_mask"] = full["attention_mask"].long()
    # MiniCPMV.forward reads position_ids from the processor batch.
    full["position_ids"] = full["attention_mask"].cumsum(-1).sub(1).clamp_min(0)
    labels = full["input_ids"].clone()
    labels[full["attention_mask"] == 0] = -100
    labels[:, :prompt_n] = -100
    full = full.to(device)
    return full, labels.to(device), prompt_n, full_n - prompt_n


def prepare(processor, row, device, prompt_style="qwen", max_slice_nums=1, max_input_length=6144):
    if prompt_style == "minicpm":
        return prepare_minicpm(processor, row, device, max_slice_nums, max_input_length)
    import torch
    from PIL import Image

    with Image.open(row["image_path"]) as source:
        image = source.convert("RGB")
    prompt_text = row["sft_prompt"] if "sft_prompt" in row else OPEN_PROMPT.format(question=row["question"])
    user = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt_text},
    ]}]
    full_messages = user + [{"role": "assistant", "content": [
        {"type": "text", "text": str(row["sft_target"])}
    ]}]
    try:
        prompt = processor.apply_chat_template(user, tokenize=False, add_generation_prompt=True)
        full_text = processor.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)
        prompt_inputs = processor(text=[prompt], images=[image], padding=True, return_tensors="pt")
        full = processor(text=[full_text], images=[image], padding=True, return_tensors="pt")
    finally:
        image.close()
    prompt_n = int(prompt_inputs["attention_mask"][0].sum())
    full_n = int(full["attention_mask"][0].sum())
    if not 0 < prompt_n < full_n or not torch.equal(
        prompt_inputs["input_ids"][0, :prompt_n], full["input_ids"][0, :prompt_n]
    ):
        raise RuntimeError(f"response boundary mismatch {prompt_n}/{full_n}")
    labels = full["input_ids"].clone()
    labels[full["attention_mask"] == 0] = -100
    labels[:, :prompt_n] = -100
    moved = {key: value.to(device) if hasattr(value, "to") else value for key, value in full.items()}
    return moved, labels.to(device), prompt_n, full_n - prompt_n


def forward_loss(model, inputs, labels, prompt_style):
    if prompt_style == "minicpm":
        # MiniCPMV.forward takes the whole processor batch under ``data``; the
        # PEFT causal-LM wrapper would instead forward it as ``input_ids``.
        return model.base_model(data=inputs, labels=labels, use_cache=False).loss
    return model(**inputs, labels=labels, use_cache=False).loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--init-adapter", type=Path)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--language-lr", type=float, default=1e-5)
    parser.add_argument("--visual-lr", type=float, default=2e-6)
    parser.add_argument("--merger-lr", type=float, default=2e-6)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-scope", choices=("all", "language"), default="all")
    parser.add_argument("--prompt-style", choices=PROMPT_STYLES, default="auto")
    parser.add_argument("--max-slice-nums", type=int, default=1)
    parser.add_argument("--max-input-length", type=int, default=6144)
    parser.add_argument(
        "--gpus",
        type=parse_gpus,
        default="0",
        help="csv of device ids; one id (default 0) keeps the historical "
        "single-GPU device_map={\"\":0}, two or more shard with device_map=balanced",
    )
    parser.add_argument(
        "--max-memory-per-gpu",
        default="21GiB",
        help="per-device budget for device_map=balanced; ignored on the single-GPU path",
    )
    parser.add_argument(
        "--disable-gradient-clipping",
        action="store_true",
        help="skip gradient clipping (the reference multi-GPU run requires this)",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--seed", type=int, default=20260802)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.epochs < 1 or args.gradient_accumulation < 1:
        raise ValueError("invalid schedule")
    if args.max_slice_nums < 1 or args.max_input_length < 1:
        raise ValueError("MiniCPM input limits must be positive")
    device_map, max_memory, multi_gpu = build_device_plan(args.gpus, args.max_memory_per_gpu)
    gradient_clip_norm = None if args.disable_gradient_clipping else 1.0
    multi_gpu_notes = []
    if multi_gpu:
        multi_gpu_notes = [
            f"device_map='balanced' over {len(args.gpus)} devices, max_memory keys "
            f"0..{len(args.gpus) - 1}",
            "AdamW uses foreach=False because the fused/foreach optimizer kernels assume one device",
            "clip_grad_norm_ reduces per-device gradient norms on the first device; use "
            "--disable-gradient-clipping to skip clipping as the reference run does",
            "no torch.autocast is used; bf16 weights come from torch_dtype=torch.bfloat16",
            "peak_allocated_bytes only reports the default (first) CUDA device",
        ]
    prompt_style = detect_prompt_style(args.model, args.prompt_style)
    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.max_rows is not None:
        rows = rows[:args.max_rows]
    if not rows or len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("empty or duplicate manifest")
    if any("sft_target" not in row for row in rows):
        raise ValueError("manifest is missing sft_target")
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    if inventory.get("status") != "passed" or inventory.get("model_config_sha256") != sha256(args.model / "config.json"):
        raise ValueError("inventory/model contract failed")
    inventory_family = {item["name"]: item["family"] for item in inventory["targets"]}
    if args.lora_scope == "language":
        target_family = {name: family for name, family in inventory_family.items() if family == "language"}
        if not target_family:
            raise ValueError("inventory has no 'language' targets; cannot run --lora-scope language")
    else:
        target_family = dict(inventory_family)
    target_modules = sorted(target_family)
    selected_families = set(target_family.values())

    args.output.mkdir(parents=True)
    (args.output / "status").write_text("running\n", encoding="utf-8")
    run_config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    run_config.update({
        "manifest_sha256": sha256(args.manifest),
        "inventory_sha256": sha256(args.inventory),
        "model_config_sha256": sha256(args.model / "config.json"),
        "prompt": "row.sft_prompt if present, otherwise OPEN_PROMPT",
        "target_field": "sft_target",
        "prompt_style": prompt_style,
        "max_slice_nums": args.max_slice_nums,
        "max_input_length": args.max_input_length,
        "gpus": list(args.gpus),
        "multi_gpu": multi_gpu,
        "device_map": device_map,
        "max_memory": max_memory,
        "gradient_clip_norm": gradient_clip_norm,
    })
    (args.output / "run_config.json").write_text(
        json.dumps(run_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if multi_gpu:
        # Pin the requested physical devices before torch enumerates CUDA so the
        # local indices (0..n-1) match the max_memory keys.  The single-GPU path
        # never touches CUDA_VISIBLE_DEVICES, preserving existing runs.
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(device) for device in args.gpus)

    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModel, AutoProcessor, get_cosine_schedule_with_warmup
    try:
        from transformers import AutoModelForImageTextToText
    except ImportError:  # transformers < 4.46 (e.g. the MiniCPM-V-2_6 env pins 4.40.0)
        AutoModelForImageTextToText = None

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    processor_kwargs = dict(local_files_only=True, trust_remote_code=args.trust_remote_code)
    if prompt_style == "qwen":
        processor_kwargs.update(min_pixels=256 * 28 * 28, max_pixels=512 * 28 * 28)
    try:
        processor = AutoProcessor.from_pretrained(args.model, **processor_kwargs)
    except TypeError:
        # Non-Qwen processors (e.g. MiniCPM-V) do not accept min/max_pixels.
        processor = AutoProcessor.from_pretrained(
            args.model, local_files_only=True, trust_remote_code=args.trust_remote_code
        )
    load_kwargs = dict(
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        local_files_only=True,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    if max_memory is not None:
        load_kwargs["max_memory"] = max_memory
    # Primary path stays AutoModelForImageTextToText (Qwen2.5-VL / Qwen3-VL).
    # Fall back to AutoModel + trust_remote_code for architectures whose remote
    # class is not exposed through AutoModelForImageTextToText (MiniCPM-V-2_6 on
    # transformers 4.40.0 does not even define AutoModelForImageTextToText).
    loader_name = None
    load_errors = []
    base = None
    if AutoModelForImageTextToText is not None:
        try:
            base = AutoModelForImageTextToText.from_pretrained(
                args.model, trust_remote_code=args.trust_remote_code, **load_kwargs
            )
            loader_name = "AutoModelForImageTextToText"
        except Exception as exc:  # noqa: BLE001 - fall through to the AutoModel fallback
            load_errors.append(f"AutoModelForImageTextToText: {type(exc).__name__}: {exc}")
    if base is None:
        try:
            base = AutoModel.from_pretrained(args.model, trust_remote_code=True, **load_kwargs)
            loader_name = "AutoModel"
        except Exception as exc:  # noqa: BLE001 - report every attempt
            load_errors.append(f"AutoModel: {type(exc).__name__}: {exc}")
            raise RuntimeError("model load failed; attempts: " + " | ".join(load_errors)) from exc
    base.config.use_cache = False
    # Preflight: every inventory target must resolve to a real module in the
    # loaded model. A name mismatch is almost always a transformers version
    # prefix change (legacy 'model.layers.*' vs 'model.language_model.*').
    model_module_names = set(dict(base.named_modules()))
    missing_targets = [name for name in sorted(inventory_family) if name not in model_module_names]
    if missing_targets:
        legacy_inventory = any(name.startswith("model.layers.") for name in inventory_family)
        modern_inventory = any("model.language_model." in name for name in inventory_family)
        legacy_model = any(name.startswith("model.layers.") for name in model_module_names)
        modern_model = any("model.language_model." in name for name in model_module_names)
        hint = ""
        if legacy_inventory and modern_model:
            hint = (
                " The loaded model exposes 'model.language_model.*' names while the inventory uses the "
                "legacy 'model.layers.*' names: rebuild the inventory with the same transformers version "
                "used to load the model."
            )
        elif modern_inventory and legacy_model:
            hint = (
                " The loaded model exposes the legacy 'model.layers.*' names while the inventory uses "
                "'model.language_model.*': rebuild the inventory with the same transformers version used "
                "to load the model."
            )
        raise RuntimeError(
            f"inventory/model module-name mismatch: {len(missing_targets)}/{len(inventory_family)} "
            f"target(s) from {args.inventory} do not resolve to modules in {args.model}; first missing: "
            f"{missing_targets[:5]}. Likely cause: a transformers version prefix change (e.g. "
            f"'model.layers.*' vs 'model.language_model.*'), not a bad checkpoint.{hint}"
        )
    patched = force_default_lora_dispatch(base)
    if args.init_adapter is not None:
        model = PeftModel.from_pretrained(base, str(args.init_adapter), is_trainable=True, local_files_only=True)
    else:
        model = get_peft_model(base, LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
            inference_mode=False,
        ))
    model.config.use_cache = False
    # On a sharded model ``model.device`` is the first parameter's device; route
    # inputs through the input-embedding device so accelerate's dispatch hooks
    # move them to the consuming shard.  Single-GPU keeps ``model.device``.
    input_device = model.device
    if multi_gpu:
        try:
            input_device = model.get_input_embeddings().weight.device
        except (AttributeError, NotImplementedError):
            input_device = model.device
    # MiniCPMV itself does not implement checkpointing; enable it on its LLM.
    minicpm_llm = getattr(model.base_model, "llm", None)
    gradient_target = (
        minicpm_llm
        if prompt_style == "minicpm" and hasattr(minicpm_llm, "gradient_checkpointing_enable")
        else model
    )
    gradient_target.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    gradient_target.enable_input_require_grads()
    trainable = [(name, value) for name, value in model.named_parameters() if value.requires_grad]
    families = {}
    for name, _ in trainable:
        matches = {family for target, family in target_family.items() if target in name}
        if len(matches) != 1:
            raise RuntimeError(f"cannot classify trainable tensor: {name} {matches}")
        families[name] = matches.pop()
    counts = Counter(families.values())
    if set(counts) != selected_families:
        raise RuntimeError(
            f"incomplete trainable scope {dict(counts)}; expected {sorted(selected_families)} "
            f"for --lora-scope {args.lora_scope}"
        )
    before = {name: value.detach().float().cpu().clone() for name, value in trainable}
    family_lrs = {
        "language": args.language_lr,
        "visual_blocks": args.visual_lr,
        "merger": args.merger_lr,
    }
    # Preserve the historical language/visual_blocks/merger group order for the
    # default all-family run; language-only runs build a single language group.
    family_order = ("language", "visual_blocks", "merger")
    ordered_families = [f for f in family_order if f in selected_families]
    ordered_families += sorted(selected_families - set(family_order))
    groups, family_devices = build_param_groups(trainable, families, ordered_families, family_lrs)
    optimizer_kwargs = dict(betas=(0.9, 0.95), weight_decay=0.0)
    if multi_gpu:
        # Fused/foreach AdamW kernels assume every parameter lives on one device.
        optimizer_kwargs["foreach"] = False
    optimizer = torch.optim.AdamW(groups, **optimizer_kwargs)
    total_micro = len(rows) * args.epochs
    total_updates = math.ceil(total_micro / args.gradient_accumulation)
    warmup_steps = 0 if total_updates < 10 else math.ceil(total_updates * 0.03)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_updates)
    order = []
    for epoch in range(args.epochs):
        indices = list(range(len(rows)))
        random.Random(args.seed + epoch).shuffle(indices)
        order.extend((epoch + 1, index) for index in indices)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    losses, updates, micro = [], 0, 0
    started = time.time()
    with (args.output / "step_metrics.jsonl").open("x", encoding="utf-8", newline="\n") as log:
        for position, (epoch, index) in enumerate(order, 1):
            row = rows[index]
            inputs, labels, prompt_n, target_n = prepare(
                processor,
                row,
                input_device,
                prompt_style=prompt_style,
                max_slice_nums=args.max_slice_nums,
                max_input_length=args.max_input_length,
            )
            loss = forward_loss(model, inputs, labels, prompt_style)
            scalar = float(loss.detach().float().cpu())
            if not math.isfinite(scalar):
                raise FloatingPointError("non-finite loss")
            (loss / args.gradient_accumulation).backward()
            losses.append(scalar)
            micro += 1
            should_step = micro % args.gradient_accumulation == 0 or position == len(order)
            if should_step:
                # A non-finite max_norm is the no-clip sentinel: clip_grad_norm_
                # still returns the total norm but multiplies gradients by
                # clamp(inf) == 1.0.  It reduces per-device norms on the first
                # device, so it is valid for sharded parameters.
                clip_max_norm = gradient_clip_norm if gradient_clip_norm is not None else float("inf")
                grad_norm = float(torch.nn.utils.clip_grad_norm_(
                    [value for _, value in trainable], clip_max_norm
                ).detach().float().cpu())
                if not math.isfinite(grad_norm):
                    raise FloatingPointError("non-finite grad")
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                updates += 1
            record = {
                "position": position,
                "epoch": epoch,
                "sample_id": row["sample_id"],
                "loss": scalar,
                "prompt_tokens": prompt_n,
                "target_tokens": target_n,
                "optimizer_step": updates,
                "stepped": should_step,
            }
            log.write(json.dumps(record, sort_keys=True) + "\n")
            log.flush()
            print(json.dumps(record), flush=True)
            del inputs, labels, loss

    changed, finite = Counter(), Counter()
    for name, value in trainable:
        current = value.detach().float().cpu()
        family_name = families[name]
        finite[family_name] += int(torch.isfinite(current).all())
        changed[family_name] += int(not torch.equal(current, before[name]))
    present_families = sorted(family for family, count in counts.items() if count)
    if any(changed[name] == 0 or finite[name] != counts[name] for name in present_families):
        raise RuntimeError(
            f"parameter integrity failed changed={dict(changed)} finite={dict(finite)} "
            f"families={present_families}"
        )
    adapter = args.output / "adapter"
    model.save_pretrained(adapter, safe_serialization=True)
    processor.save_pretrained(adapter)
    unloaded = model.unload()
    reloaded = PeftModel.from_pretrained(unloaded, str(adapter), is_trainable=False, local_files_only=True)
    reload_count = sum("lora_" in name for name, _ in reloaded.named_parameters())
    result = {
        "status": "passed",
        "algorithm": "formal-v024-generic-qwen-vl-mm-lora-response-only-ce-v1",
        "model_type": inventory.get("model_type"),
        "lora_scope": args.lora_scope,
        "prompt_style": prompt_style,
        "base_model_loader": loader_name,
        "rows": len(rows),
        "epochs": args.epochs,
        "micro_steps": micro,
        "optimizer_steps": updates,
        "warmup_steps": warmup_steps,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "loss_mean": sum(losses) / len(losses),
        "trainable_tensors": len(trainable),
        "trainable_tensor_groups": dict(counts),
        "changed_tensor_groups": dict(changed),
        "finite_tensor_groups": dict(finite),
        "reloaded_lora_parameter_tensors": reload_count,
        "manifest_sha256": sha256(args.manifest),
        "inventory_sha256": sha256(args.inventory),
        "adapter_sha256": sha256(adapter / "adapter_model.safetensors"),
        "dispatch_patch": patched,
        "gpus": list(args.gpus),
        "multi_gpu": multi_gpu,
        "device_map": device_map,
        "max_memory": max_memory,
        "gradient_clip_norm": gradient_clip_norm,
        "trainable_tensor_devices": family_devices,
        "multi_gpu_notes": multi_gpu_notes,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "elapsed_seconds": time.time() - started,
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output / "status").write_text("passed\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
