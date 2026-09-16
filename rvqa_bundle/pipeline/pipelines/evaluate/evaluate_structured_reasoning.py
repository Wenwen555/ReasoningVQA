#!/usr/bin/env python3
"""Generate structured trajectories and compute deterministic rubric components."""

import argparse
import hashlib
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from PIL import Image

class Generator:
    """Minimal Qwen3-VL loader matching the successful Stage5 training route."""
    def __init__(self, model_path, adapter=""):
        from transformers import AutoModelForImageTextToText, AutoProcessor
        from peft import PeftModel
        self.processor = AutoProcessor.from_pretrained(
            model_path, local_files_only=True, min_pixels=256 * 28 * 28, max_pixels=512 * 28 * 28
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map={"": 0}, local_files_only=True,
            low_cpu_mem_usage=True, attn_implementation="sdpa"
        ).eval()
        if adapter:
            # The optional server torchao package is older than PEFT's probe
            # requirement; this ordinary BF16 LoRA adapter does not use it.
            try:
                import peft.tuners.lora.torchao as peft_torchao
                peft_torchao.is_torchao_available = lambda: False
            except (ImportError, AttributeError):
                pass
            self.model = PeftModel.from_pretrained(
                self.model, adapter, is_trainable=False, local_files_only=True
            ).eval()
        self.input_device = self.model.get_input_embeddings().weight.device

    def generate(self, image, prompt, max_tokens):
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image}, {"type": "text", "text": prompt}
        ]}]
        rendered = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[rendered], images=[image], padding=True, return_tensors="pt")
        inputs = {k: v.to(self.input_device) if hasattr(v, "to") else v for k, v in inputs.items()}
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
        trim = [y[len(x):] for x, y in zip(inputs["input_ids"], out)]
        return self.processor.batch_decode(trim, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def extract_json(text):
    text = str(text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if match:
            try: return json.loads(match.group(0))
            except Exception: pass
    return None


def text_generate(gen, prompt, max_tokens):
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    rendered = gen.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = gen.processor(text=[rendered], padding=True, return_tensors="pt")
    inputs = {k: v.to(gen.input_device) if hasattr(v, "to") else v for k, v in inputs.items()}
    with torch.inference_mode():
        out = gen.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
    trim = [y[len(x):] for x, y in zip(inputs["input_ids"], out)]
    return gen.processor.batch_decode(trim, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()


def score_row(row, parsed):
    required = {"visual_features", "root_choice", "evidence_ids", "hops", "answer_choice"}
    schema = int(isinstance(parsed, dict) and set(parsed) == required and isinstance(parsed.get("visual_features"), list)
                 and isinstance(parsed.get("evidence_ids"), list) and isinstance(parsed.get("hops"), list))
    if not schema:
        return {"schema_valid": 0, "root_correct": 0, "evidence_f1": 0.0, "evidence_exact": 0,
                "path_edge_accuracy": 0.0, "path_continuous": 0, "path_exact": 0, "answer_correct": 0,
                "deterministic_score_without_rotation": 0.0, "full_chain": 0}
    root = int(parsed["root_choice"] == row["gold_root_choice"])
    pred_ev, gold_ev = set(map(str, parsed["evidence_ids"])), set(map(str, row["gold_evidence_ids"]))
    precision = len(pred_ev & gold_ev) / len(pred_ev) if pred_ev else 0.0
    recall = len(pred_ev & gold_ev) / len(gold_ev) if gold_ev else 0.0
    ev_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    ev_exact = int(parsed["evidence_ids"] == row["gold_evidence_ids"])
    pred_hops, gold_hops = parsed["hops"], row["gold_path"]
    edge_hits = sum(int(i < len(pred_hops) and pred_hops[i] == edge) for i, edge in enumerate(gold_hops))
    edge_acc = edge_hits / len(gold_hops) if gold_hops else 0.0
    # A pre-structured baseline may emit a list of strings. Treat malformed hop
    # elements as a zero path score instead of crashing the entire evaluation.
    hops_are_objects = all(isinstance(x, dict) for x in pred_hops)
    continuous = int(len(pred_hops) == len(gold_hops) and hops_are_objects and all(
        pred_hops[i-1].get("object") == pred_hops[i].get("subject") for i in range(1, len(pred_hops))
    ))
    path_exact = int(pred_hops == gold_hops)
    answer = int(parsed["answer_choice"] == row["gold_answer_choice"])
    score = 5 * schema + 20 * root + 15 * ev_f1 + 20 * edge_acc + 5 * continuous + 10 * answer
    full = int(schema and root and ev_exact and path_exact and answer)
    return {"schema_valid": schema, "root_correct": root, "evidence_f1": ev_f1, "evidence_exact": ev_exact,
            "path_edge_accuracy": edge_acc, "path_continuous": continuous, "path_exact": path_exact,
            "answer_correct": answer, "deterministic_score_without_rotation": score, "full_chain": full}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--adapter", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mode", choices=("correct-image", "no-image", "cyclic-shuffled-image"), default="correct-image")
    p.add_argument("--max-rows", type=int)
    p.add_argument("--max-new-tokens", type=int, default=256)
    a = p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    rows = [json.loads(x) for x in a.manifest.read_text(encoding="utf-8").splitlines() if x.strip()]
    if a.max_rows: rows = rows[:a.max_rows]
    a.output.mkdir(parents=True)
    (a.output / "status").write_text("running\n")
    contract = {"manifest": str(a.manifest), "manifest_sha256": sha256(a.manifest), "model": str(a.model),
                "adapter": str(a.adapter) if a.adapter else None, "mode": a.mode, "rows": len(rows),
                "max_new_tokens": a.max_new_tokens, "rubric_version": "hybrid-100-v17.2"}
    (a.output / "contract.json").write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    gen = Generator(str(a.model), str(a.adapter) if a.adapter else "")
    output, started = [], time.time()
    with (a.output / "predictions.jsonl").open("x", encoding="utf-8", newline="\n") as f:
        for i, row in enumerate(rows):
            image_id = None
            if a.mode == "no-image":
                raw = text_generate(gen, row["sft_prompt"], a.max_new_tokens)
            else:
                source = row if a.mode == "correct-image" else rows[(i + 12) % len(rows)]
                image_id = source["image_id"]
                with Image.open(source["image_path"]) as im:
                    raw = gen.generate(im.convert("RGB"), row["sft_prompt"], a.max_new_tokens)
            parsed = extract_json(raw)
            rec = {"sample_id": row["sample_id"], "group_id": row["group_id"], "hop_count": row["hop_count"],
                   "rotation": row["rotation"], "mode": a.mode, "used_image_id": image_id, "raw_output": raw,
                   "parsed_output": parsed, "gold_root_choice": row["gold_root_choice"],
                   "gold_evidence_ids": row["gold_evidence_ids"], "gold_path": row["gold_path"],
                   "gold_answer_choice": row["gold_answer_choice"], **score_row(row, parsed)}
            output.append(rec); f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n"); f.flush()
            print(json.dumps({"row": i+1, "total": len(rows), "schema": rec["schema_valid"], "root": rec["root_correct"],
                              "path": rec["path_exact"], "answer": rec["answer_correct"], "full": rec["full_chain"]}), flush=True)

    groups = defaultdict(list)
    for x in output: groups[(x["group_id"], x["hop_count"])].append(x)
    rotation_pass = {}
    for key, values in groups.items():
        rotation_pass[key] = int(len(values) == 4 and {x["rotation"] for x in values} == {0,1,2,3}
                                 and all(x["root_correct"] and x["answer_correct"] for x in values))
    for x in output:
        x["rotation_consistency"] = rotation_pass[(x["group_id"], x["hop_count"])]
        x["deterministic_score_80"] = x["deterministic_score_without_rotation"] + 5 * x["rotation_consistency"]
        x["external_visual_score_20"] = None
        x["hybrid_score_100"] = None
    # Rewrite once to include group-level rotation component.
    with (a.output / "predictions.scored.jsonl").open("x", encoding="utf-8", newline="\n") as f:
        for x in output: f.write(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n")
    n = len(output)
    summary = {"status": "passed", "rows": n, "mode": a.mode, "schema_valid": sum(x["schema_valid"] for x in output),
               "root_correct": sum(x["root_correct"] for x in output), "evidence_exact": sum(x["evidence_exact"] for x in output),
               "path_exact": sum(x["path_exact"] for x in output), "answer_correct": sum(x["answer_correct"] for x in output),
               "full_chain": sum(x["full_chain"] for x in output), "rotation_groups_passed": sum(rotation_pass.values()),
               "rotation_groups": len(rotation_pass), "deterministic_score_80_mean": sum(x["deterministic_score_80"] for x in output)/n,
               "external_visual_score_20_mean": None, "hybrid_score_100_mean": None,
               "elapsed_seconds": time.time()-started}
    (a.output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (a.output / "status").write_text("passed\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
