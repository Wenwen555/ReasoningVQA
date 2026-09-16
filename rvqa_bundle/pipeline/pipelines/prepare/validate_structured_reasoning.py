#!/usr/bin/env python3
"""Independent validator; deliberately does not import the v17.2 builder."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def load(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--dev", type=Path, required=True)
    p.add_argument("--formal-train", type=Path, required=True)
    p.add_argument("--formal-eval", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    train, dev = load(a.train), load(a.dev)
    formal = load(a.formal_train) + load(a.formal_eval)
    errors = []
    if len(train) != 480: errors.append(f"train_rows={len(train)}")
    if len(dev) != 240: errors.append(f"dev_rows={len(dev)}")
    if len({x.get("sample_id") for x in train + dev}) != len(train) + len(dev): errors.append("duplicate_sample_id")
    if {x.get("schema_version") for x in train + dev} != {"spec009-structured-mc-reasoning20-v17.2"}: errors.append("schema_version")
    if set(x.get("view_role") for x in train) != {"anchor", "same-root-unseen-view"}: errors.append("train_view_roles")
    if set(x.get("view_role") for x in dev) != {"third-view-fresh"}: errors.append("dev_view_roles")
    if len({x["root_qid"] for x in train}) != 20 or len({x["root_qid"] for x in dev}) != 20: errors.append("root_count")
    if {x["root_qid"] for x in train} != {x["root_qid"] for x in dev}: errors.append("root_set_mismatch")
    if {x["image_sha256"] for x in train} & {x["image_sha256"] for x in dev}: errors.append("train_dev_image_sha_overlap")
    if {x["image_id"] for x in train} & {x["image_id"] for x in dev}: errors.append("train_dev_image_id_overlap")
    for row in train + dev:
        if not Path(row["image_path"]).is_file() or digest(row["image_path"]) != row["image_sha256"]:
            errors.append(f'image_contract:{row.get("sample_id")}')
            break

    edge_truth = set()
    for row in formal:
        subject_label = row["root_label"]
        for i, edge in enumerate(row["path"]):
            obj = edge.get("object_qid") or edge.get("object_literal")
            edge_truth.add((edge["subject_qid"], edge["property_qid"], obj, edge["object_label"]))
            subject_label = edge["object_label"]

    per_item = defaultdict(list)
    for row in train + dev:
        sid = row["sample_id"]
        try:
            target = json.loads(row["sft_target"])
        except Exception as exc:
            errors.append(f"target_json:{sid}:{exc}")
            continue
        if set(target) != {"visual_features", "root_choice", "evidence_ids", "hops", "answer_choice"}:
            errors.append(f"target_schema:{sid}")
        if len(row["root_options"]) != 4 or len({x["id"] for x in row["root_options"]}) != 4:
            errors.append(f"root_options:{sid}")
        if len(row["answer_options"]) != 4 or len({x["text"] for x in row["answer_options"]}) != 4:
            errors.append(f"answer_options:{sid}")
        if target.get("root_choice") != row["gold_root_choice"] or target.get("answer_choice") != row["gold_answer_choice"]:
            errors.append(f"target_choice:{sid}")
        if target.get("evidence_ids") != row["gold_evidence_ids"] or target.get("hops") != row["gold_path"]:
            errors.append(f"target_path:{sid}")
        cards = {x["evidence_id"]: x for x in row["evidence_cards"]}
        if len(cards) != 4 * row["hop_count"]:
            errors.append(f"card_count:{sid}")
        for card in cards.values():
            edge = (card["subject_qid"], card["property_qid"], card["object_id"], card["object_label"])
            if edge not in edge_truth:
                errors.append(f"unverified_edge:{sid}:{card['evidence_id']}")
        selected = [cards.get(x) for x in target.get("evidence_ids", [])]
        if None in selected or len(selected) != row["hop_count"]:
            errors.append(f"selected_card_count:{sid}")
        else:
            selected.sort(key=lambda x: x["edge_index"])
            for i, (card, hop) in enumerate(zip(selected, target["hops"])):
                expected = {"subject": card["subject_qid"], "relation": card["property_qid"], "object": card["object_id"]}
                if hop != expected: errors.append(f"hop_card_mismatch:{sid}:{i}")
                if i and selected[i-1]["object_id"] != card["subject_qid"]: errors.append(f"path_discontinuity:{sid}:{i}")
            if any(card["branch_root_choice"] != target["root_choice"] for card in selected): errors.append(f"branch_mismatch:{sid}")
        answer = next((x["text"] for x in row["answer_options"] if x["id"] == target.get("answer_choice")), None)
        if answer != row["gold_answer_text"]: errors.append(f"answer_mapping:{sid}")
        key = (row["split"], row["group_id"], row["view_role"], row["hop_count"])
        per_item[key].append(row)

    for key, rows in per_item.items():
        if len(rows) != 4 or {x["rotation"] for x in rows} != {0, 1, 2, 3}:
            errors.append(f"rotation_set:{key}")
        if Counter(x["gold_root_choice"] for x in rows) != Counter({x: 1 for x in "R1 R2 R3 R4".split()}):
            errors.append(f"root_rotation_balance:{key}")
        if Counter(x["gold_answer_choice"] for x in rows) != Counter({x: 1 for x in "C1 C2 C3 C4".split()}):
            errors.append(f"answer_rotation_balance:{key}")

    report = {
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "train_rows": len(train),
        "dev_rows": len(dev),
        "roots": len({x["root_qid"] for x in train}),
        "train_hops": dict(Counter(x["hop_count"] for x in train)),
        "dev_hops": dict(Counter(x["hop_count"] for x in dev)),
        "train_sha256": digest(a.train),
        "dev_sha256": digest(a.dev),
        "verified_edge_cards": sum(len(x["evidence_cards"]) for x in train + dev),
    }
    a.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    raise SystemExit(0 if not errors else 1)


if __name__ == "__main__":
    main()
