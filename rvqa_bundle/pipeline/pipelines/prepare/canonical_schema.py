"""Canonical row contract shared by all ReasoningVQA source adapters."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "reasoningvqa.canonical.v1"
SOURCE_DATASETS = {"inaturalist", "gldv2", "visual_genome"}
ANSWER_MODES = ("open", "json")
DEFAULT_ANSWER_MODE = "open"
# Verbatim historical GLDv2 open-ended prompt. It is kept byte-identical so the
# new open-ended runs stay comparable to the existing base-vs-SFT numbers.
OPEN_PROMPT = (
    "Answer the question using the image. Return only the short answer.\n"
    "Question: {question}"
)
REQUIRED_FIELDS = {
    "schema_version",
    "sample_id",
    "source_dataset",
    "source_record_id",
    "split",
    "task_type",
    "image_id",
    "image_path",
    "question",
    "answer",
    "answers",
    "answer_options",
    # ``reasoning_path`` is intentionally not required: the open-ended mode drops
    # the reasoning trace. It stays a tolerated optional field so older JSON rows
    # keep validating without exploding.
    "evidence_refs",
    "sft_prompt",
    "sft_target",
    "provenance",
}


def normalize_answer_mode(value: Any = None) -> str:
    mode = str(value if value is not None else DEFAULT_ANSWER_MODE).strip().lower()
    if mode not in ANSWER_MODES:
        raise ValueError(f"unknown answer mode {value!r}; expected one of {ANSWER_MODES}")
    return mode


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_id(source_dataset: str, source_record_id: str, row: dict[str, Any]) -> str:
    identifier = str(source_record_id).strip()
    if identifier:
        raw = f"{SCHEMA_VERSION}|{source_dataset}|{identifier}"
    else:
        raw = f"{SCHEMA_VERSION}|{source_dataset}|{canonical_json(row)}"
    return f"{source_dataset}:{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:24]}"


def nonempty(value: Any, name: str) -> str:
    text = str(value if value is not None else "").strip()
    if not text:
        raise ValueError(f"{name} must be non-empty")
    return text


def answer_label(value: Any) -> str:
    if isinstance(value, dict):
        for key in ("label", "text", "value", "name"):
            if str(value.get(key, "")).strip():
                return str(value[key]).strip()
        return ""
    return str(value if value is not None else "").strip()


def correct_answer(row: dict[str, Any]) -> str:
    for key in ("gold_answer_text", "program_answer", "answer_normalized"):
        value = answer_label(row.get(key))
        if value:
            return value
    value = answer_label(row.get("final_answer"))
    if value:
        return value
    options = row.get("answers")
    correct = row.get("correct")
    if isinstance(options, list) and isinstance(correct, list) and len(options) == len(correct):
        selected = [answer_label(option) for option, flag in zip(options, correct) if flag]
        if len(selected) == 1 and selected[0]:
            return selected[0]
    return answer_label(row.get("answer"))


def answer_options(row: dict[str, Any]) -> list[str]:
    values = row.get("answer_options") or row.get("answers") or row.get("choices") or []
    result = []
    if isinstance(values, list):
        for value in values:
            text = answer_label(value)
            if text and text not in result:
                result.append(text)
    return result


def _entity_id(value: Any) -> str:
    if isinstance(value, dict):
        for key in ("qid", "id", "value", "object_id", "subject_id", "label", "name"):
            if str(value.get(key, "")).strip():
                return str(value[key]).strip()
        return ""
    return str(value if value is not None else "").strip()


def normalize_edge(edge: Any) -> dict[str, str]:
    if not isinstance(edge, dict):
        raise ValueError(f"reasoning edge must be a mapping: {edge!r}")
    subject = edge.get(
        "subject", edge.get("subject_qid", edge.get("subject_id", edge.get("root")))
    )
    relation = edge.get(
        "relation",
        edge.get("predicate", edge.get("property_qid", edge.get("property"))),
    )
    obj = edge.get(
        "object",
        edge.get(
            "object_qid",
            edge.get("object_id", edge.get("object_literal", edge.get("tail"))),
        ),
    )
    normalized = {
        "subject": _entity_id(subject),
        "relation": _entity_id(relation),
        "object": _entity_id(obj),
    }
    if not all(normalized.values()):
        raise ValueError(f"incomplete reasoning edge: {edge!r}")
    return normalized


def normalize_path(values: Any) -> list[dict[str, str]]:
    if values in (None, ""):
        return []
    if not isinstance(values, list):
        raise ValueError("reasoning path must be a list")
    return [normalize_edge(value) for value in values]


def source_path(row: dict[str, Any]) -> list[dict[str, str]]:
    if isinstance(row.get("gold_path"), list):
        return normalize_path(row["gold_path"])
    evidence = row.get("evidence_refs") if isinstance(row.get("evidence_refs"), dict) else {}
    scene = row.get("scene_graph_path") or evidence.get("scene_graph_path") or []
    kg = row.get("path")
    if not isinstance(kg, list) or (kg and not isinstance(kg[0], dict)):
        kg = evidence.get("kg_path") or (evidence.get("kg") or {}).get("path") or []
    return normalize_path(scene) + normalize_path(kg)


def build_prompt(
    question: str,
    options: list[str],
    existing: Any = None,
    answer_mode: Any = None,
) -> str:
    """Render the SFT prompt for the requested answer mode.

    ``open`` (default) yields the historical open-ended prompt verbatim: the
    answer candidates are deliberately omitted and no JSON output is requested.
    ``json`` keeps the previous multiple-choice + ``{answer, reasoning_path}``
    contract for backward compatibility, and still honours an explicit
    ``existing`` prompt.
    """
    mode = normalize_answer_mode(answer_mode)
    if mode == "open":
        return OPEN_PROMPT.format(question=question)
    if str(existing or "").strip():
        body = str(existing).strip()
    else:
        body = "Use the image to answer the question.\nQuestion: " + question
        if options:
            body += "\nAnswer candidates:\n" + "\n".join(
                f"{index + 1}. {value}" for index, value in enumerate(options)
            )
    return (
        body
        + "\nReturn exactly one JSON object with keys answer and reasoning_path. "
        "answer must be the final answer text. reasoning_path must be a list of "
        "objects with subject, relation, and object; use an empty list when no path is supplied."
    )


def build_target(
    answer: str,
    reasoning_path: list[dict[str, str]],
    answer_mode: Any = None,
) -> str:
    """Return the SFT target for the requested answer mode.

    ``open`` (default) is the bare answer string, which makes ``open_exact``
    scoring valid directly. ``json`` keeps the legacy serialized object.
    """
    mode = normalize_answer_mode(answer_mode)
    if mode == "open":
        return str(answer)
    return canonical_json({"answer": answer, "reasoning_path": reasoning_path})


def validate_row(
    row: dict[str, Any], check_image: bool = False, answer_mode: Any = None
) -> list[str]:
    mode = normalize_answer_mode(answer_mode)
    errors = []
    missing = sorted(REQUIRED_FIELDS - set(row))
    if missing:
        return [f"missing_fields:{','.join(missing)}"]
    if row.get("schema_version") != SCHEMA_VERSION:
        errors.append("schema_version")
    for key in (
        "sample_id",
        "source_record_id",
        "split",
        "task_type",
        "image_id",
        "image_path",
        "question",
        "answer",
        "sft_prompt",
        "sft_target",
    ):
        if not isinstance(row.get(key), str) or not row[key].strip():
            errors.append(f"nonempty:{key}")
    if row.get("source_dataset") not in SOURCE_DATASETS:
        errors.append("source_dataset")
    if not isinstance(row.get("answers"), list) or row.get("answer") not in row.get("answers", []):
        errors.append("answers")
    if not isinstance(row.get("answer_options"), list):
        errors.append("answer_options")
    path = row.get("reasoning_path")
    if mode == "json" and not isinstance(path, list):
        errors.append("reasoning_path")
    elif path is not None:
        if not isinstance(path, list):
            errors.append("reasoning_path")
        else:
            try:
                if normalize_path(path) != path:
                    errors.append("reasoning_path_normalization")
            except ValueError:
                errors.append("reasoning_path_edges")
    if not isinstance(row.get("evidence_refs"), dict):
        errors.append("evidence_refs")
    if not isinstance(row.get("provenance"), dict):
        errors.append("provenance")
    if mode == "open":
        if row.get("sft_target") != row.get("answer"):
            errors.append("sft_target")
    else:
        try:
            target = json.loads(row.get("sft_target", ""))
            expected = {"answer": row.get("answer"), "reasoning_path": row.get("reasoning_path")}
            if target != expected:
                errors.append("sft_target")
        except (TypeError, json.JSONDecodeError):
            errors.append("sft_target_json")
    image_sha = row.get("image_sha256")
    if image_sha is not None and (
        not isinstance(image_sha, str)
        or len(image_sha) != 64
        or any(char not in "0123456789abcdef" for char in image_sha.lower())
    ):
        errors.append("image_sha256")
    if check_image and not Path(str(row.get("image_path", ""))).is_file():
        errors.append("image_missing")
    return errors
