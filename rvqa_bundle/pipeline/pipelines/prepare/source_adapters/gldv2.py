"""Adapter for both materialized GLDv2 QA and frozen structured curriculum rows."""

from .common import canonical_row


SOURCE_DATASET = "gldv2"


def matches(row):
    source = str(row.get("source", "")).lower()
    schema = str(row.get("schema_version", "")).lower()
    image = row.get("image") if isinstance(row.get("image"), dict) else {}
    return (
        "gldv2" in source
        or "gldv2" in str(image.get("source", "")).lower()
        or schema.startswith("spec009-structured-mc-reasoning")
    )


def adapt(row, split=None, answer_mode="open"):
    if not matches(row):
        raise ValueError("row is not a GLDv2 source record")
    structured = str(row.get("schema_version", "")).startswith("spec009-")
    provenance = {
        "adapter": SOURCE_DATASET,
        "adapter_version": "1",
        "source": row.get("source", "gldv2-structured" if structured else None),
        "source_schema_version": row.get("schema_version"),
        "recipe": row.get("recipe"),
        "root": row.get("root") or {
            "qid": row.get("root_qid"),
            "label": row.get("root_label"),
        },
        "group_id": row.get("group_id"),
        "view_role": row.get("view_role"),
        "image_attribution": row.get("image_attribution"),
        "question_source_sample_id": row.get("question_source_sample_id"),
    }
    return canonical_row(
        row,
        source_dataset=SOURCE_DATASET,
        split=split,
        task_type="landmark_structured_reasoning" if structured else "landmark_knowledge_reasoning",
        provenance=provenance,
        existing_prompt=row.get("sft_prompt") if structured else None,
        answer_mode=answer_mode,
    )
