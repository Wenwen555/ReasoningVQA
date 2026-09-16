"""Adapter for Visual Genome scene-graph grounded reasoning manifests."""

from .common import canonical_row


SOURCE_DATASET = "visual_genome"


def matches(row):
    # The materialized subsets use the short source code "VG"; the frozen
    # structured manifests use the full "visual_genome" name.
    source = str(row.get("source", "")).lower()
    image = row.get("image") if isinstance(row.get("image"), dict) else {}
    return (
        source in {SOURCE_DATASET, "vg"}
        or str(image.get("source", "")).lower() == SOURCE_DATASET
    )


def adapt(row, split=None, answer_mode="open"):
    if not matches(row):
        raise ValueError("row is not a Visual Genome source record")
    provenance = {
        "adapter": SOURCE_DATASET,
        "adapter_version": "1",
        "source": row.get("source"),
        "source_schema_version": row.get("schema_version"),
        "recipe": row.get("recipe"),
        "root": row.get("root"),
        "linking_evidence": row.get("linking_evidence"),
        "image_attribution": row.get("image_attribution"),
        "hop_components": row.get("hop_components"),
    }
    return canonical_row(
        row,
        source_dataset=SOURCE_DATASET,
        split=split,
        task_type="scene_graph_knowledge_reasoning",
        provenance=provenance,
        answer_mode=answer_mode,
    )
