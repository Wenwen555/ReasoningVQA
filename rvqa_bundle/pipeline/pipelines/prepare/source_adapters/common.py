"""Shared helpers used by source adapters, never by downstream training code."""

from __future__ import annotations

from typing import Any

from pipelines.prepare.canonical_schema import (
    DEFAULT_ANSWER_MODE,
    SCHEMA_VERSION,
    answer_options,
    build_prompt,
    build_target,
    correct_answer,
    nonempty,
    source_path,
    stable_id,
    validate_row,
)


def image_value(row: dict[str, Any], key: str) -> Any:
    if row.get(key) not in (None, ""):
        return row[key]
    image = row.get("image")
    return image.get(key) if isinstance(image, dict) else None


def source_record_id(row: dict[str, Any]) -> str:
    for key in ("sample_id", "question_id", "source_sample_id", "unit_id"):
        if row.get(key) not in (None, ""):
            return str(row[key])
    image = image_value(row, "image_id")
    question = str(row.get("question", "")).strip()
    return f"{image}|{question}" if image and question else ""


def canonical_row(
    row: dict[str, Any],
    *,
    source_dataset: str,
    split: str | None,
    task_type: str,
    provenance: dict[str, Any],
    existing_prompt: Any = None,
    answer_mode: str = DEFAULT_ANSWER_MODE,
) -> dict[str, Any]:
    source_id = nonempty(source_record_id(row), "source_record_id")
    question = str(row.get("question") or existing_prompt or "").strip()
    question = nonempty(question, "question")
    answer = nonempty(correct_answer(row), "answer")
    options = answer_options(row)
    reasoning_path = source_path(row)
    evidence = row.get("evidence_refs") if isinstance(row.get("evidence_refs"), dict) else {}
    image_id = nonempty(image_value(row, "image_id"), "image_id")
    image_path = nonempty(image_value(row, "image_path"), "image_path")
    resolved_split = nonempty(split or row.get("split"), "split")
    accepted_answers = [answer]
    for value in row.get("accepted_answers", []):
        text = str(value).strip()
        if text and text not in accepted_answers:
            accepted_answers.append(text)
    result = {
        "schema_version": SCHEMA_VERSION,
        "sample_id": stable_id(source_dataset, source_id, row),
        "source_dataset": source_dataset,
        "source_record_id": source_id,
        "split": resolved_split,
        "task_type": task_type,
        "image_id": image_id,
        "image_path": image_path,
        "image_sha256": image_value(row, "image_sha256"),
        "question": question,
        "answer": answer,
        "answers": accepted_answers,
        "answer_options": options,
        "reasoning_path": reasoning_path,
        "evidence_refs": {
            "kg_path": evidence.get("kg_path") or (evidence.get("kg") or {}).get("path") or [],
            "scene_graph_path": row.get("scene_graph_path") or evidence.get("scene_graph_path") or [],
            "linking_evidence": row.get("linking_evidence") or evidence.get("linking_evidence"),
        },
        "sft_prompt": build_prompt(question, options, existing_prompt, answer_mode=answer_mode),
        "sft_target": build_target(answer, reasoning_path, answer_mode=answer_mode),
        "provenance": provenance,
    }
    errors = validate_row(result, answer_mode=answer_mode)
    if errors:
        raise ValueError(f"adapter produced invalid canonical row: {errors}")
    return result
