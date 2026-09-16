from pathlib import Path

from .common import Command, build_action_commands


def _format(value, context):
    try:
        return str(value).format_map(context)
    except KeyError as exc:
        raise ValueError(f"unknown config placeholder: {exc.args[0]}") from exc


def _canonical_commands(config, backend):
    datasets = config["datasets"]
    pipeline_root = Path(str(config["source"]["pipeline_root"])).resolve()
    context = {
        **backend.context(config),
        "pipeline_root": str(pipeline_root),
        "experiment_root": str(config["output"]["experiment_root"]),
    }
    python = str(config["runtime"]["python"])
    adapt_script = pipeline_root / "prepare" / "adapt_source.py"
    validate_script = pipeline_root / "prepare" / "validate_canonical_manifest.py"
    compose_script = pipeline_root / "prepare" / "compose_canonical_manifests.py"
    for script in (adapt_script, validate_script, compose_script):
        if not script.is_file():
            raise FileNotFoundError(script)
    commands = []
    outputs = {}
    source_configs = {}
    for source in datasets["sources"]:
        name = source["name"]
        adapter = source["adapter"]
        output = _format(
            source.get("output", f"{{experiment_root}}/data/canonical/{name}.jsonl"),
            context,
        )
        summary = _format(
            source.get("summary", f"{{experiment_root}}/data/canonical/{name}.adaptation.json"),
            context,
        )
        command = [
            python,
            str(adapt_script),
            "--adapter",
            adapter,
            "--input",
            _format(source["input"], context),
            "--output",
            output,
            "--summary",
            summary,
        ]
        if source.get("split"):
            command += ["--split", str(source["split"])]
        if source.get("image_root"):
            command += ["--image-root", _format(source["image_root"], context)]
        if source.get("allow_mixed_source", False):
            command.append("--allow-mixed-source")
        commands.append(Command(name=f"adapt-{name}", argv=command, env={}))
        validation = [
            python,
            str(validate_script),
            "--manifest",
            output,
            "--output",
            _format(
                source.get("validation", f"{{experiment_root}}/data/canonical/{name}.validation.json"),
                context,
            ),
            "--expected-source",
            adapter,
        ]
        if source.get("split"):
            validation += ["--expected-split", str(source["split"])]
        if source.get("check_images", False):
            validation.append("--check-images")
        commands.append(Command(name=f"validate-{name}", argv=validation, env={}))
        outputs[name] = output
        source_configs[name] = source
    for manifest in datasets.get("manifests", []):
        name = manifest["name"]
        input_names = manifest["inputs"]
        output = _format(
            manifest.get("output", f"{{experiment_root}}/data/manifests/{name}.jsonl"),
            context,
        )
        summary = _format(
            manifest.get("summary", f"{{experiment_root}}/data/manifests/{name}.composition.json"),
            context,
        )
        command = [python, str(compose_script)]
        for input_name in input_names:
            command += ["--input", outputs[input_name]]
        command += ["--output", output, "--summary", summary]
        commands.append(Command(name=f"compose-{name}", argv=command, env={}))
        validation = [
            python,
            str(validate_script),
            "--manifest",
            output,
            "--output",
            _format(
                manifest.get("validation", f"{{experiment_root}}/data/manifests/{name}.validation.json"),
                context,
            ),
        ]
        for value in sorted({source_configs[item]["adapter"] for item in input_names}):
            validation += ["--expected-source", value]
        for value in sorted({source_configs[item].get("split") for item in input_names} - {None, ""}):
            validation += ["--expected-split", str(value)]
        if manifest.get("check_images", False):
            validation.append("--check-images")
        commands.append(Command(name=f"validate-{name}", argv=validation, env={}))
    return commands


def build_commands(config, backend):
    if "datasets" in config:
        return _canonical_commands(config, backend)
    return build_action_commands(config, backend, "prepare")
