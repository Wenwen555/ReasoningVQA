from pathlib import Path


def require(mapping, key):
    if key not in mapping:
        raise ValueError(f"missing required key: {key}")
    return mapping[key]


def require_mapping(mapping, key):
    value = require(mapping, key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    return value


def resolve_script(config, script):
    source = require_mapping(config, "source")
    root = Path(str(require(source, "pipeline_root")))
    path = Path(str(script))
    resolved_root = root.resolve()
    resolved = path.resolve() if path.is_absolute() else (resolved_root / path).resolve()
    if not resolved.is_relative_to(resolved_root):
        raise ValueError(f"pipeline script escapes source.pipeline_root: {script}")
    return resolved
