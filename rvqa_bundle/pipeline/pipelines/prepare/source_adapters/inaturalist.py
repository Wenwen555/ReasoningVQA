"""Adapter for iNaturalist taxonomy and Wikidata reasoning manifests."""

from .common import canonical_row


SOURCE_DATASET = "inaturalist"


def matches(row):
    # The materialized subsets use the short source code "iNat"; the frozen
    # structured manifests use "iNaturalist2021..." and an inaturalist block.
    source = str(row.get("source", "")).lower()
    return "inat" in source or isinstance(row.get("inaturalist"), dict)


def adapt(row, split=None, answer_mode="open"):
    if not matches(row):
        raise ValueError("row is not an iNaturalist source record")
    metadata = row.get("inaturalist") if isinstance(row.get("inaturalist"), dict) else {}
    provenance = {
        "adapter": SOURCE_DATASET,
        "adapter_version": "1",
        "source": row.get("source"),
        "source_schema_version": row.get("schema_version"),
        "recipe": row.get("recipe"),
        "generator_prompt_version": row.get("generator_prompt_version"),
        "root_concept": row.get("root_concept"),
        "taxonomy": metadata.get("taxonomy"),
        "entity_id": row.get("entity_id"),
        "property_route": row.get("route_properties"),
    }
    return canonical_row(
        row,
        source_dataset=SOURCE_DATASET,
        split=split,
        task_type="taxonomy_knowledge_reasoning",
        provenance=provenance,
        answer_mode=answer_mode,
    )
