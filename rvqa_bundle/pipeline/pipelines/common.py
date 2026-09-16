import os
from dataclasses import dataclass

from model_backends.common import require_mapping, resolve_script


@dataclass(frozen=True)
class Command:
    name: str
    argv: list[str]
    env: dict[str, str]


def _format(value, context):
    try:
        return str(value).format_map(context)
    except KeyError as exc:
        raise ValueError(f"unknown config placeholder: {exc.args[0]}") from exc


def build_action_commands(config, backend, action):
    actions = require_mapping(config, "actions")
    steps = actions.get(action, [])
    if not isinstance(steps, list):
        raise ValueError(f"actions.{action} must be a list")
    source = require_mapping(config, "source")
    output = require_mapping(config, "output")
    context = {
        **backend.context(config),
        "pipeline_root": str(source["pipeline_root"]),
        "experiment_root": str(output["experiment_root"]),
    }
    commands = []
    for index, step in enumerate(steps, 1):
        if not isinstance(step, dict):
            raise ValueError(f"actions.{action}[{index}] must be a mapping")
        name = str(step.get("name", f"{action}-{index}"))
        operation = step.get("operation")
        explicit_script = step.get("script")
        if operation is not None and explicit_script is not None:
            raise ValueError(f"step {name}: use operation or script, not both")
        if operation is not None:
            script = backend.resolve_operation(config, action, str(operation))
        elif explicit_script is not None:
            script = resolve_script(config, explicit_script)
        else:
            raise ValueError(f"step {name}: operation or script is required")
        if "grpo" in script.name.lower():
            raise ValueError(f"GRPO is outside the current pipeline: {script.name}")
        python = _format(step.get("python", "{python}"), context)
        args = step.get("args", [])
        if not isinstance(args, list):
            raise ValueError(f"step {name}: args must be a list")
        argv = [python, str(script), *[_format(arg, context) for arg in args]]
        env = {str(k): _format(v, context) for k, v in step.get("env", {}).items()}
        commands.append(Command(name=name, argv=argv, env=env))
    return commands


def execute(command, dry_run=False):
    if dry_run:
        return
    import subprocess

    env = os.environ.copy()
    env.update(command.env)
    subprocess.run(command.argv, check=True, env=env)
