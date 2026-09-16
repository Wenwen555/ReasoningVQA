NAME = "qwen_vl"

OPERATIONS = {
    "train": {
        "canonical_sft": "train/train_qwen_vl_sft.py",
        "structured_sft": "train/train_qwen_vl_sft.py",
    },
    "evaluate": {
        "canonical_generation": "evaluate/evaluate_canonical_reasoning.py",
        "structured_generation": "evaluate/evaluate_structured_reasoning.py",
    },
}


def context(config):
    runtime = config["runtime"]
    model = config["model"]
    return {
        "python": str(runtime["python"]),
        "model": str(model["checkpoint"]),
        "inventory": str(model.get("inventory", "")),
    }
