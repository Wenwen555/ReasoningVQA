def _mapping(parent, key):
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    return value


def _nonempty_string(mapping, key, qualified=None):
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{qualified or key} must be a non-empty string")
    return value


def validate_pipeline_config(config):
    if not isinstance(config, dict):
        raise ValueError("config root must be a mapping")
    source = _mapping(config, "source")
    output = _mapping(config, "output")
    runtime = _mapping(config, "runtime")
    model = _mapping(config, "model")
    actions = _mapping(config, "actions")
    _nonempty_string(source, "pipeline_root", "source.pipeline_root")
    _nonempty_string(output, "experiment_root", "output.experiment_root")
    _nonempty_string(runtime, "python", "runtime.python")
    _nonempty_string(model, "backend", "model.backend")
    datasets = config.get("datasets")
    if datasets is not None:
        if not isinstance(datasets, dict):
            raise ValueError("datasets must be a mapping")
        sources = datasets.get("sources")
        if not isinstance(sources, list) or not sources:
            raise ValueError("datasets.sources must be a non-empty list")
        names = set()
        for index, source_config in enumerate(sources, 1):
            prefix = f"datasets.sources[{index}]"
            if not isinstance(source_config, dict):
                raise ValueError(f"{prefix} must be a mapping")
            for key in ("name", "adapter", "input"):
                _nonempty_string(source_config, key, f"{prefix}.{key}")
            if "image_root" in source_config:
                _nonempty_string(source_config, "image_root", f"{prefix}.image_root")
            if source_config["adapter"] not in {"inaturalist", "gldv2", "visual_genome"}:
                raise ValueError(f"{prefix}.adapter is unsupported: {source_config['adapter']!r}")
            if source_config["name"] in names:
                raise ValueError(f"duplicate dataset source name: {source_config['name']}")
            names.add(source_config["name"])
            for key in ("allow_mixed_source", "check_images"):
                if key in source_config and not isinstance(source_config[key], bool):
                    raise ValueError(f"{prefix}.{key} must be boolean")
        manifests = datasets.get("manifests", [])
        if not isinstance(manifests, list):
            raise ValueError("datasets.manifests must be a list")
        manifest_names = set()
        for index, manifest in enumerate(manifests, 1):
            prefix = f"datasets.manifests[{index}]"
            if not isinstance(manifest, dict):
                raise ValueError(f"{prefix} must be a mapping")
            _nonempty_string(manifest, "name", f"{prefix}.name")
            inputs = manifest.get("inputs")
            if not isinstance(inputs, list) or not inputs or any(item not in names for item in inputs):
                raise ValueError(f"{prefix}.inputs must reference dataset source names")
            if manifest["name"] in manifest_names:
                raise ValueError(f"duplicate canonical manifest name: {manifest['name']}")
            manifest_names.add(manifest["name"])
    for action in actions:
        if action not in {"prepare", "train", "evaluate"}:
            raise ValueError(f"unsupported actions key: {action}")
    for action in ("prepare", "train", "evaluate"):
        steps = actions.get(action, [])
        if not isinstance(steps, list):
            raise ValueError(f"actions.{action} must be a list")
        names = set()
        for index, step in enumerate(steps, 1):
            prefix = f"actions.{action}[{index}]"
            if not isinstance(step, dict):
                raise ValueError(f"{prefix} must be a mapping")
            name = step.get("name", f"{action}-{index}")
            if not isinstance(name, str) or not name:
                raise ValueError(f"{prefix}.name must be a non-empty string")
            if name in names:
                raise ValueError(f"duplicate step name in actions.{action}: {name}")
            names.add(name)
            has_operation = "operation" in step
            has_script = "script" in step
            if has_operation == has_script:
                raise ValueError(f"{prefix} must declare exactly one of operation or script")
            _nonempty_string(step, "operation" if has_operation else "script", prefix)
            args = step.get("args", [])
            if not isinstance(args, list) or any(isinstance(arg, (dict, list)) for arg in args):
                raise ValueError(f"{prefix}.args must be a flat list")
            env = step.get("env", {})
            if not isinstance(env, dict) or any(isinstance(value, (dict, list)) for value in env.values()):
                raise ValueError(f"{prefix}.env must be a scalar mapping")
            if "python" in step and (not isinstance(step["python"], str) or not step["python"]):
                raise ValueError(f"{prefix}.python must be a non-empty string")
