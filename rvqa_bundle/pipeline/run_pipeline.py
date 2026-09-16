#!/usr/bin/env python3
import argparse
import os
import shlex
from pathlib import Path

import yaml

from model_backends import get_backend
from model_backends.common import require_mapping
from pipelines import build_commands
from pipelines.common import execute
from config_schema import validate_pipeline_config


def load_config(path):
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("config root must be a mapping")
    return value


def validate_config(config):
    validate_pipeline_config(config)
    backend_name = str(config["model"].get("backend", ""))
    if not backend_name:
        raise ValueError("model.backend is required")
    backend = get_backend(backend_name)
    backend.context(config)
    return backend


def freeze_config(config_path, config):
    root = Path(str(config["output"]["experiment_root"]))
    root.mkdir(parents=True, exist_ok=True)
    target = root / "pipeline_config.yaml"
    payload = Path(config_path).read_bytes()
    if target.exists() and target.read_bytes() != payload:
        raise ValueError(f"refusing to replace different frozen config: {target}")
    if not target.exists():
        target.write_bytes(payload)


def main():
    parser = argparse.ArgumentParser(description="ReasoningVQA public pipeline")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--print-command", action="store_true")
    parser.add_argument("action", choices=("prepare", "train", "evaluate", "all"))
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    backend = validate_config(config)
    if not args.dry_run and not args.print_command:
        freeze_config(config_path, config)

    actions = ("prepare", "train", "evaluate") if args.action == "all" else (args.action,)
    for action in actions:
        for index, command in enumerate(build_commands(config, backend, action), 1):
            prefix = " ".join(f"{k}={shlex.quote(v)}" for k, v in command.env.items())
            rendered = shlex.join(command.argv)
            print(f"[{action}:{index}:{command.name}] {prefix} {rendered}".strip(), flush=True)
            if not args.print_command:
                execute(command, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
