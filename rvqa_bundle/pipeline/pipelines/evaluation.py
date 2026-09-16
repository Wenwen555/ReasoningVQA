from .common import build_action_commands


def build_commands(config, backend):
    return build_action_commands(config, backend, "evaluate")
