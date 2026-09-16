import sys
import unittest
import copy
from types import ModuleType
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model_backends import get_backend
from model_backends.adapter import BackendAdapter
from pipelines import build_commands
from run_pipeline import load_config, validate_config


class PipelineContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(ROOT / "configs" / "example_pipeline.yaml")
        # The published example config ships with empty machine-specific paths;
        # fill them with local stand-ins so the contracts can be exercised.
        cls.config["source"]["pipeline_root"] = str(ROOT / "pipelines")
        cls.config["output"]["experiment_root"] = str(ROOT / ".pytest-experiment")
        cls.config["runtime"]["python"] = sys.executable
        cls.config["model"]["checkpoint"] = "/models/Qwen3-VL-8B-Instruct"
        cls.config["model"]["inventory"] = str(ROOT / ".pytest-inventory.json")
        cls.backend = validate_config(cls.config)

    def test_registered_backend(self):
        self.assertIs(get_backend("qwen_vl"), self.backend)
        self.assertIsInstance(self.backend, BackendAdapter)

    def test_actions_expand_in_order(self):
        self.assertEqual(len(build_commands(self.config, self.backend, "prepare")), 2)
        self.assertEqual(len(build_commands(self.config, self.backend, "train")), 1)
        self.assertEqual(len(build_commands(self.config, self.backend, "evaluate")), 2)

    def test_scripts_are_self_contained(self):
        pipeline_root = ROOT / "pipelines"
        for action in ("prepare", "train", "evaluate"):
            for command in build_commands(self.config, self.backend, action):
                script = Path(command.argv[1]).resolve()
                self.assertTrue(script.is_relative_to(pipeline_root))
                self.assertTrue(script.is_file())

    def test_model_placeholders_expand(self):
        command = build_commands(self.config, self.backend, "train")[0]
        self.assertIn("/models/Qwen3-VL-8B-Instruct", command.argv)
        self.assertEqual(command.env["CUDA_VISIBLE_DEVICES"], "0")

    def test_grpo_is_rejected(self):
        config = dict(self.config)
        config["actions"] = {"train": [{"script": "train_grpo.py"}]}
        with self.assertRaisesRegex(ValueError, "GRPO"):
            build_commands(config, self.backend, "train")

    def test_grpo_operation_is_rejected(self):
        config = dict(self.config)
        config["actions"] = {"train": [{"operation": "train_grpo"}]}
        with self.assertRaisesRegex(ValueError, "GRPO"):
            build_commands(config, self.backend, "train")

    def test_unknown_backend_operation_is_actionable(self):
        config = dict(self.config)
        config["actions"] = {"train": [{"operation": "unknown"}]}
        with self.assertRaisesRegex(ValueError, "available=.*structured_sft"):
            build_commands(config, self.backend, "train")

    def test_incomplete_backend_is_rejected(self):
        module = ModuleType("incomplete")
        module.NAME = "incomplete"
        with self.assertRaisesRegex(TypeError, "context"):
            BackendAdapter.from_module(module)

    def test_duplicate_step_names_are_rejected_during_validation(self):
        config = copy.deepcopy(self.config)
        config["actions"]["train"] = [
            {"name": "duplicate", "operation": "structured_sft"},
            {"name": "duplicate", "operation": "structured_sft"},
        ]
        with self.assertRaisesRegex(ValueError, "duplicate step name"):
            validate_config(config)

    def test_malformed_environment_is_rejected_during_validation(self):
        config = copy.deepcopy(self.config)
        config["actions"]["train"][0]["env"] = []
        with self.assertRaisesRegex(ValueError, "env must be a scalar mapping"):
            validate_config(config)

    def test_unknown_action_is_rejected_during_validation(self):
        config = copy.deepcopy(self.config)
        config["actions"]["grpo"] = []
        with self.assertRaisesRegex(ValueError, "unsupported actions key"):
            validate_config(config)

    def test_portable_template_has_no_shenying_paths(self):
        template = (ROOT / "configs" / "pipeline.template.yaml").read_text(encoding="utf-8")
        self.assertNotIn("/data/wenjt", template)
        config = load_config(ROOT / "configs" / "pipeline.template.yaml")
        config["source"]["pipeline_root"] = str(ROOT / "pipelines")
        config["output"]["experiment_root"] = str(ROOT / ".pytest-experiment")
        config["runtime"]["python"] = sys.executable
        config["model"]["checkpoint"] = str(ROOT / ".pytest-model")
        config["model"]["inventory"] = str(ROOT / ".pytest-inventory.json")
        for source in config["datasets"]["sources"]:
            source["input"] = str(ROOT / "datasets" / f"{source['name']}.jsonl")
        backend = validate_config(config)
        prepare = build_commands(config, backend, "prepare")
        self.assertEqual(len(prepare), 16)
        self.assertEqual(sum("adapt_source.py" in command.argv[1] for command in prepare), 6)
        self.assertEqual(sum("compose_canonical_manifests.py" in command.argv[1] for command in prepare), 2)
        self.assertEqual(len(build_commands(config, backend, "train")), 1)
        self.assertEqual(len(build_commands(config, backend, "evaluate")), 2)

    def test_canonical_downstream_does_not_branch_on_source_names(self):
        paths = [
            ROOT / "pipelines" / "train" / "train_qwen_vl_sft.py",
            ROOT / "pipelines" / "evaluate" / "evaluate_canonical_reasoning.py",
            ROOT / "pipelines" / "evaluate" / "validate_canonical_predictions.py",
        ]
        text = "\n".join(path.read_text(encoding="utf-8").lower() for path in paths)
        for source_name in ("inaturalist", "gldv2", "visual_genome"):
            self.assertNotIn(source_name, text)

    def test_mainline_artifact_names_do_not_encode_row_counts(self):
        config_text = (ROOT / "configs" / "example_pipeline.yaml").read_text(encoding="utf-8")
        builder_text = (ROOT / "pipelines" / "prepare" / "build_structured_reasoning.py").read_text(encoding="utf-8")
        self.assertNotIn("train480", config_text + builder_text)
        self.assertNotIn("dev240", config_text + builder_text)

    def test_script_escape_is_rejected(self):
        config = copy.deepcopy(self.config)
        config["actions"]["prepare"] = [{"script": "../outside.py"}]
        with self.assertRaisesRegex(ValueError, "escapes source.pipeline_root"):
            build_commands(config, self.backend, "prepare")

    def test_unknown_placeholder_is_actionable(self):
        config = copy.deepcopy(self.config)
        config["actions"]["prepare"] = [
            {"script": "prepare/build_structured_reasoning.py", "args": ["{missing}"]}
        ]
        with self.assertRaisesRegex(ValueError, "unknown config placeholder: missing"):
            build_commands(config, self.backend, "prepare")


if __name__ == "__main__":
    unittest.main()
