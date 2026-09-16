from . import data, evaluation, training

BUILDERS = {
    "prepare": data.build_commands,
    "train": training.build_commands,
    "evaluate": evaluation.build_commands,
}


def build_commands(config, backend, action):
    return BUILDERS[action](config, backend)
