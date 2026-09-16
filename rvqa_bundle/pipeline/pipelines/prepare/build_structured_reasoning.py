#!/usr/bin/env python3
"""Build create-only 20-root structured multimodal reasoning curriculum."""

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


LETTERS = "ABCD"


def load_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def write_jsonl(path, rows):
    with path.open("x", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def edge_text(card):
    obj = card["object_label"]
    return f'{card["evidence_id"]}: {card["subject_label"]} [{card["subject_qid"]}] --{card["property_label"]} [{card["property_qid"]}]--> {obj} [{card["object_id"]}]'


def target_edge(edge):
    return {
        "subject": edge["subject_qid"],
        "relation": edge["property_qid"],
        "object": edge.get("object_qid") or edge.get("object_literal"),
    }


def build_row(group, image, path_rows, hop, rotation, split, seed):
    # The same four visually similar roots each receive a real Wikidata chain.
    roots = [{"root_qid": group["root_qid"], "root_label": group["root_label"], "role": "gold"}]
    roots += [{"root_qid": x["root_qid"], "root_label": x["root_label"], "role": "visual-hard-negative"}
              for x in group["hard_negatives"]]
    root_shift = rotation
    roots = roots[root_shift:] + roots[:root_shift]
    root_options = [{"id": f"R{i+1}", **x} for i, x in enumerate(roots)]
    gold_root_choice = next(x["id"] for x in root_options if x["role"] == "gold")

    cards = []
    branch_cards = defaultdict(list)
    endpoint_pool = []
    for root in root_options:
        source = path_rows[(root["root_qid"], hop)]
        endpoint_pool.append(source["answer_text"])
        subject_label = root["root_label"]
        for j, edge in enumerate(source["path"]):
            obj_id = edge.get("object_qid") or edge.get("object_literal")
            card = {
                "branch_root_choice": root["id"],
                "subject_qid": edge["subject_qid"],
                "subject_label": subject_label,
                "property_qid": edge["property_qid"],
                "property_label": edge["property_label"],
                "object_id": obj_id,
                "object_label": edge["object_label"],
                "edge_index": j,
            }
            cards.append(card)
            subject_label = edge["object_label"]

    rng = random.Random(seed + int(group["group_id"].rsplit("-", 1)[1]) * 100 + hop * 10 + rotation)
    rng.shuffle(cards)
    for i, card in enumerate(cards):
        card["evidence_id"] = f"E{i+1}"
        branch_cards[card["branch_root_choice"]].append(card)
    for value in branch_cards.values():
        value.sort(key=lambda x: x["edge_index"])

    gold_source = path_rows[(group["root_qid"], hop)]
    # Keep exactly four unique endpoint choices. If decoy chains share an endpoint,
    # fill from the original model-blind choices for the gold formal item.
    answer_values = []
    for value in endpoint_pool + gold_source["choices"]:
        if value not in answer_values:
            answer_values.append(value)
        if len(answer_values) == 4:
            break
    if gold_source["answer_text"] not in answer_values or len(answer_values) != 4:
        raise ValueError((group["group_id"], hop, answer_values))
    # Freeze the correct answer at each of C1/C2/C3/C4 exactly once per
    # canonical item. Shuffle only the three decoys so root rotation cannot
    # accidentally cancel answer rotation.
    gold_answer = gold_source["answer_text"]
    decoys = [x for x in answer_values if x != gold_answer]
    rng.shuffle(decoys)
    answer_values = list(decoys)
    answer_values.insert(rotation, gold_answer)
    answer_options = [{"id": f"C{i+1}", "text": x} for i, x in enumerate(answer_values)]
    gold_answer_choice = next(x["id"] for x in answer_options if x["text"] == gold_source["answer_text"])

    prompt = (
        "Use the image and the evidence cards to solve the task. First identify which visually similar root entity is shown. "
        "Then select only the evidence cards belonging to that root and follow them in order. "
        "Return exactly one JSON object with keys visual_features, root_choice, evidence_ids, hops, answer_choice. "
        "visual_features must contain short visible cues; hops must use QID/PID/object IDs exactly as provided.\n"
        f"Required hop count: {hop}.\nRoot candidates:\n" +
        "\n".join(f'{x["id"]}. {x["root_label"]}' for x in root_options) +
        "\nEvidence cards:\n" + "\n".join(edge_text(x) for x in cards) +
        "\nAnswer candidates:\n" + "\n".join(f'{x["id"]}. {x["text"]}' for x in answer_options)
    )
    gold_cards = branch_cards[gold_root_choice]
    target = {
        "visual_features": [f'visible {group["root_class_label"]}'],
        "root_choice": gold_root_choice,
        "evidence_ids": [x["evidence_id"] for x in gold_cards],
        "hops": [target_edge(x) for x in gold_source["path"]],
        "answer_choice": gold_answer_choice,
    }
    sample_id = f'{group["group_id"]}::{split}::{image["view_role"]}::hop{hop}::rot{rotation}'
    return {
        "schema_version": "spec009-structured-mc-reasoning20-v17.2",
        "sample_id": sample_id,
        "split": split,
        "group_id": group["group_id"],
        "hop_count": hop,
        "rotation": rotation,
        "image_id": image["image_id"],
        "image_path": image["image_path"],
        "image_sha256": image["image_sha256"],
        "view_role": image["view_role"],
        "root_qid": group["root_qid"],
        "root_label": group["root_label"],
        "root_class_label": group["root_class_label"],
        "root_options": root_options,
        "evidence_cards": cards,
        "answer_options": answer_options,
        "gold_root_choice": gold_root_choice,
        "gold_evidence_ids": target["evidence_ids"],
        "gold_path": target["hops"],
        "gold_answer_choice": gold_answer_choice,
        "gold_answer_text": gold_source["answer_text"],
        "question_source_sample_id": gold_source["sample_id"],
        "sft_prompt": prompt,
        "sft_target": json.dumps(target, ensure_ascii=False, separators=(",", ":")),
        "human_visual_audit": group.get("human_visual_audit", "pending"),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--groups", type=Path, required=True)
    p.add_argument("--formal-train", type=Path, required=True)
    p.add_argument("--formal-eval", type=Path, required=True)
    p.add_argument("--third-view", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--train-filename", default="train.jsonl")
    p.add_argument("--dev-filename", default="dev.jsonl")
    p.add_argument("--seed", type=int, default=20260806)
    a = p.parse_args()
    for name in (a.train_filename, a.dev_filename):
        if Path(name).name != name or not name.endswith(".jsonl"):
            raise ValueError("output filenames must be plain .jsonl basenames")
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    groups = load_jsonl(a.groups)
    formal = load_jsonl(a.formal_train) + load_jsonl(a.formal_eval)
    path_rows = {(x["root_qid"], x["hop_count"]): x for x in formal}
    thirds = {x["group_id"]: x for x in load_jsonl(a.third_view)}
    if len(groups) != 20 or len(thirds) != 20:
        raise ValueError("expected exactly 20 groups and third views")

    train, dev = [], []
    for group in groups:
        candidate_qids = [group["root_qid"]] + [x["root_qid"] for x in group["hard_negatives"]]
        if any((qid, hop) not in path_rows for qid in candidate_qids for hop in (1, 2, 3)):
            raise ValueError(f'missing real path capacity: {group["group_id"]}')
        train_images = [dict(group["anchor"], view_role="anchor"), group["alternate"]]
        for image in train_images:
            for hop in (1, 2, 3):
                for rotation in range(4):
                    train.append(build_row(group, image, path_rows, hop, rotation, "train", a.seed))
        image = thirds[group["group_id"]]
        for hop in (1, 2, 3):
            for rotation in range(4):
                dev.append(build_row(group, image, path_rows, hop, rotation, "dev-third-view", a.seed + 100000))

    train_path = a.output / a.train_filename
    dev_path = a.output / a.dev_filename
    write_jsonl(train_path, train)
    write_jsonl(dev_path, dev)
    summary = {
        "status": "built",
        "schema_version": "spec009-structured-mc-reasoning20-v17.2",
        "roots": len(groups),
        "candidate_roots": len({x["root_qid"] for g in groups for x in ([{"root_qid": g["root_qid"]}] + g["hard_negatives"])}),
        "train_rows": len(train),
        "dev_rows": len(dev),
        "train_filename": a.train_filename,
        "dev_filename": a.dev_filename,
        "train_views": dict(Counter(x["view_role"] for x in train)),
        "dev_views": dict(Counter(x["view_role"] for x in dev)),
        "train_hops": dict(Counter(x["hop_count"] for x in train)),
        "dev_hops": dict(Counter(x["hop_count"] for x in dev)),
        "root_choice_positions": dict(Counter(x["gold_root_choice"] for x in train)),
        "answer_choice_positions": dict(Counter(x["gold_answer_choice"] for x in train)),
        "sources": {"groups": str(a.groups), "formal_train": str(a.formal_train), "formal_eval": str(a.formal_eval), "third_view": str(a.third_view)},
        "source_sha256": {"groups": sha256(a.groups), "formal_train": sha256(a.formal_train), "formal_eval": sha256(a.formal_eval), "third_view": sha256(a.third_view)},
    }
    (a.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary["train_sha256"] = sha256(train_path)
    summary["dev_sha256"] = sha256(dev_path)
    (a.output / "freeze.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
