"""Registry with a complete, validated source-adapter contract."""

from . import gldv2, inaturalist, visual_genome


ADAPTERS = {
    module.SOURCE_DATASET: module
    for module in (inaturalist, gldv2, visual_genome)
}


def get_adapter(name):
    try:
        module = ADAPTERS[str(name)]
    except KeyError as exc:
        raise ValueError(f"unknown source adapter {name!r}; available={sorted(ADAPTERS)}") from exc
    if not callable(getattr(module, "matches", None)) or not callable(getattr(module, "adapt", None)):
        raise TypeError(f"incomplete source adapter: {name}")
    return module
