"""CPU-only static checks for the multi-GPU (model-parallel) device plan.

These tests never import torch and never touch a GPU.  They assert the exact
single-GPU default the trainer has always used, the sharded plan built for
``--gpus 8,9``, and that the three LoRA-family optimizer groups stay correctly
partitioned and device-reported when the model is sharded.
"""

import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "pipelines" / "train" / "train_qwen_vl_sft.py"


def load_trainer():
    spec = importlib.util.spec_from_file_location("train_qwen_vl_sft_static", TRAINER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


trainer = load_trainer()


class FakeTensor:
    """Minimal tensor stand-in exposing only ``device``."""

    def __init__(self, device):
        self.device = device


class ParseGpusTests(unittest.TestCase):
    def test_default_is_historical_single_gpu(self):
        self.assertEqual(trainer.parse_gpus(None), [0])
        self.assertEqual(trainer.parse_gpus(""), [0])
        self.assertEqual(trainer.parse_gpus("0"), [0])

    def test_csv_is_parsed(self):
        self.assertEqual(trainer.parse_gpus("8,9"), [8, 9])
        self.assertEqual(trainer.parse_gpus(" 4 , 5 ,6 "), [4, 5, 6])

    def test_invalid_values_are_rejected(self):
        for value in ("a", "8,8", "-1", "8,-2"):
            with self.assertRaises(ValueError):
                trainer.parse_gpus(value)


class BuildDevicePlanTests(unittest.TestCase):
    def test_single_gpu_default_is_unchanged(self):
        self.assertEqual(
            trainer.build_device_plan([0], "21GiB"), ({"": 0}, None, False)
        )
        self.assertEqual(
            trainer.build_device_plan(trainer.parse_gpus(None), "21GiB"),
            ({"": 0}, None, False),
        )

    def test_two_gpus_use_balanced_sharding(self):
        device_map, max_memory, multi_gpu = trainer.build_device_plan([8, 9], "21GiB")
        self.assertEqual(device_map, "balanced")
        self.assertEqual(max_memory, {0: "21GiB", 1: "21GiB"})
        self.assertTrue(multi_gpu)

    def test_budget_is_applied_per_visible_device(self):
        device_map, max_memory, multi_gpu = trainer.build_device_plan([4, 5, 6], "18GiB")
        self.assertEqual(device_map, "balanced")
        self.assertEqual(max_memory, {0: "18GiB", 1: "18GiB", 2: "18GiB"})
        self.assertTrue(multi_gpu)


class ParamGroupTests(unittest.TestCase):
    def test_language_scope_stays_one_group_when_sharded(self):
        trainable = [
            ("model.layers.0.q_proj", FakeTensor("cuda:0")),
            ("model.layers.1.q_proj", FakeTensor("cuda:1")),
        ]
        families = {name: "language" for name, _ in trainable}
        groups, devices = trainer.build_param_groups(
            trainable, families, ["language"], {"language": 1e-5}
        )
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["family"], "language")
        self.assertEqual(len(groups[0]["params"]), 2)
        self.assertEqual(devices, {"language": ["cuda:0", "cuda:1"]})

    def test_three_families_partition_and_report_devices(self):
        trainable = [
            ("model.language_model.layers.0.q_proj", FakeTensor("cuda:0")),
            ("model.visual.blocks.0.attn.qkv", FakeTensor("cuda:1")),
            ("model.visual.merger.mlp.0", FakeTensor("cuda:1")),
        ]
        families = {
            "model.language_model.layers.0.q_proj": "language",
            "model.visual.blocks.0.attn.qkv": "visual_blocks",
            "model.visual.merger.mlp.0": "merger",
        }
        lrs = {"language": 1e-5, "visual_blocks": 2e-6, "merger": 2e-6}
        groups, devices = trainer.build_param_groups(
            trainable, families, ["language", "visual_blocks", "merger"], lrs
        )
        self.assertEqual([g["family"] for g in groups], ["language", "visual_blocks", "merger"])
        self.assertEqual([len(g["params"]) for g in groups], [1, 1, 1])
        self.assertEqual(
            devices,
            {"language": ["cuda:0"], "visual_blocks": ["cuda:1"], "merger": ["cuda:1"]},
        )

    def test_unconfigured_family_is_rejected(self):
        with self.assertRaises(RuntimeError):
            trainer.build_param_groups(
                [("x", FakeTensor("cpu"))], {"x": "mystery"}, ["mystery"], {}
            )


class CliFlagTests(unittest.TestCase):
    def test_help_exposes_multi_gpu_flags(self):
        completed = subprocess.run(
            [sys.executable, str(TRAINER), "--help"],
            capture_output=True,
            text=True,
            check=True,
        )
        for flag in ("--gpus", "--max-memory-per-gpu", "--disable-gradient-clipping", "--lora-scope"):
            self.assertIn(flag, completed.stdout)


if __name__ == "__main__":
    unittest.main()
