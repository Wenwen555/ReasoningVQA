import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipelines.prepare.canonical_schema import SCHEMA_VERSION, validate_row
from pipelines.prepare.source_adapters import ADAPTERS, get_adapter
from pipelines.prepare.source_adapters import gldv2, inaturalist, visual_genome


class SourceAdapterContracts(unittest.TestCase):
    @staticmethod
    def invoke(function, *arguments):
        original = sys.argv
        try:
            sys.argv = ["test", *map(str, arguments)]
            return function()
        finally:
            sys.argv = original

    def setUp(self):
        self.inaturalist = {
            "schema_version": "knowledge_reasoning_vqa.v1",
            "source": "iNaturalist2021_val+wikidata_multihop_clean_mc",
            "split": "val",
            "question_id": 0,
            "question": "What use is associated with the species shown in the image?",
            "image_id": 2686906,
            "image_path": "/datasets/inaturalist/species.jpg",
            "answers": ["medicinal plant", "afforestation", "fodder", "vegetable"],
            "correct": [1, 0, 0, 0],
            "final_answer": {"label": "medicinal plant", "qid": "Q188840"},
            "entity_id": "Q147783",
            "root_concept": {"wikidata_qid": "Q147783", "label": "Poa pratensis"},
            "inaturalist": {"taxonomy": {"kingdom": "Plantae", "family": "Poaceae"}},
            "route_properties": ["P366"],
            "evidence_refs": {"kg_path": [{
                "root": {"qid": "Q147783"},
                "property": {"id": "P366"},
                "tail": {"qid": "Q188840"},
            }]},
        }
        self.visual_genome = {
            "schema_version": "reasoningvqa-v1",
            "source": "visual_genome",
            "split": "train",
            "question_id": 1001,
            "question": "What type of object is shown?",
            "image_id": "107961",
            "image_path": "/datasets/visual-genome/107961.jpg",
            "image_sha256": "8" * 64,
            "answers": ["hand tool", "footwear"],
            "correct": [1, 0],
            "final_answer": {"label": "hand tool", "qid": "Q2578402"},
            "root": {"label": "bat", "qid": "Q3817547"},
            "linking_evidence": {"method": "wordnet", "wikidata_qid": "Q3817547"},
            "scene_graph_path": [{
                "subject_id": "3798726",
                "predicate": "holding",
                "object_id": "1073640",
                "relationship_id": "3200827",
            }],
            "evidence_refs": {"kg_path": [{
                "root": {"qid": "Q3817547"},
                "property": {"id": "P279"},
                "tail": {"qid": "Q2578402"},
            }]},
        }
        self.gldv2 = {
            "schema_version": "spec009-structured-mc-reasoning20-v17.2",
            "sample_id": "visual-v161-00::train::anchor::hop1::rot0",
            "split": "train",
            "image_id": "4b63544aa75957ce",
            "image_path": "/datasets/gldv2/4b63544aa75957ce.jpg",
            "image_sha256": "d" * 64,
            "root_qid": "Q1907813",
            "root_label": "Oele Molle",
            "group_id": "visual-v161-00",
            "view_role": "anchor",
            "gold_answer_text": "Molenplein",
            "gold_path": [{"subject": "Q1907813", "relation": "P669", "object": "Q19348380"}],
            "answer_options": [
                {"id": "C1", "text": "Molenplein"},
                {"id": "C2", "text": "windmill"},
            ],
            "sft_prompt": "Use the image and evidence cards to solve the landmark task.",
        }

    def test_registry_is_exact_and_complete(self):
        self.assertEqual(set(ADAPTERS), {"inaturalist", "gldv2", "visual_genome"})
        for name in ADAPTERS:
            self.assertIs(get_adapter(name), ADAPTERS[name])

    def test_each_realistic_source_shape_normalizes_to_one_schema(self):
        cases = (
            (inaturalist, self.inaturalist, "inaturalist", "taxonomy_knowledge_reasoning"),
            (gldv2, self.gldv2, "gldv2", "landmark_structured_reasoning"),
            (visual_genome, self.visual_genome, "visual_genome", "scene_graph_knowledge_reasoning"),
        )
        keys = None
        for adapter, source, name, task in cases:
            with self.subTest(adapter=name):
                row = adapter.adapt(source)
                self.assertEqual(row["schema_version"], SCHEMA_VERSION)
                self.assertEqual(row["source_dataset"], name)
                self.assertEqual(row["task_type"], task)
                self.assertEqual(validate_row(row), [])
                self.assertEqual(row["sft_target"], row["answer"])
                self.assertEqual(
                    row["sft_prompt"],
                    "Answer the question using the image. Return only the short answer.\n"
                    f"Question: {row['question']}",
                )
                self.assertNotIn("Answer candidates:", row["sft_prompt"])
                self.assertNotIn("reasoning_path", row["sft_prompt"])
                self.assertTrue(row["answer_options"])
                keys = set(row) if keys is None else keys
                self.assertEqual(set(row), keys)

    def test_json_mode_keeps_the_legacy_contract(self):
        row = visual_genome.adapt(self.visual_genome, answer_mode="json")
        self.assertEqual(validate_row(row, answer_mode="json"), [])
        self.assertIn("Answer candidates:", row["sft_prompt"])
        self.assertIn("reasoning_path", row["sft_prompt"])
        self.assertEqual(
            json.loads(row["sft_target"]),
            {"answer": row["answer"], "reasoning_path": row["reasoning_path"]},
        )
        # The open contract rejects the legacy serialized target.
        self.assertIn("sft_target", validate_row(row, answer_mode="open"))

    def test_adapter_rejects_the_wrong_source(self):
        with self.assertRaisesRegex(ValueError, "not a GLDv2"):
            gldv2.adapt(self.inaturalist)

    def test_split_override_is_explicit(self):
        row = visual_genome.adapt(self.visual_genome, split="dev")
        self.assertEqual(row["split"], "dev")

    def test_stable_sample_id(self):
        first = inaturalist.adapt(self.inaturalist)
        second = inaturalist.adapt(dict(self.inaturalist))
        self.assertEqual(first["sample_id"], second["sample_id"])
        changed = dict(self.inaturalist, question_id=1)
        self.assertNotEqual(first["sample_id"], inaturalist.adapt(changed)["sample_id"])

    def test_canonical_manifest_validator_cli(self):
        from pipelines.prepare.validate_canonical_manifest import main

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "canonical.jsonl"
            manifest.write_text(
                json.dumps(inaturalist.adapt(self.inaturalist)) + "\n", encoding="utf-8"
            )
            output = root / "validation.json"
            original = sys.argv
            try:
                sys.argv = [
                    "test",
                    "--manifest",
                    str(manifest),
                    "--output",
                    str(output),
                    "--expected-source",
                    "inaturalist",
                    "--expected-split",
                    "val",
                    "--expected-rows",
                    "1",
                ]
                with self.assertRaises(SystemExit) as exit_status:
                    main()
                self.assertEqual(exit_status.exception.code, 0)
            finally:
                sys.argv = original
            self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["status"], "passed")

    def test_adapt_and_compose_clis_are_create_only(self):
        from pipelines.prepare.adapt_source import main as adapt_main
        from pipelines.prepare.compose_canonical_manifests import main as compose_main

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.jsonl"
            source.write_text(json.dumps(self.inaturalist) + "\n", encoding="utf-8")
            canonical = root / "canonical.jsonl"
            adaptation = root / "adaptation.json"
            self.invoke(
                adapt_main,
                "--adapter",
                "inaturalist",
                "--input",
                source,
                "--output",
                canonical,
                "--summary",
                adaptation,
                "--image-root",
                root / "images",
            )
            relocated = json.loads(canonical.read_text(encoding="utf-8"))
            self.assertEqual(
                relocated["image_path"],
                str(root / "images" / Path(self.inaturalist["image_path"]).name),
            )
            with self.assertRaises(FileExistsError):
                self.invoke(
                    adapt_main,
                    "--adapter",
                    "inaturalist",
                    "--input",
                    source,
                    "--output",
                    canonical,
                    "--summary",
                    adaptation,
                )
            composed = root / "composed.jsonl"
            composition = root / "composition.json"
            self.invoke(
                compose_main,
                "--input",
                canonical,
                "--output",
                composed,
                "--summary",
                composition,
            )
            self.assertEqual(composed.read_text(encoding="utf-8"), canonical.read_text(encoding="utf-8"))

    def test_canonical_prediction_is_independently_rescored(self):
        from pipelines.evaluate.validate_canonical_predictions import main as validate_predictions

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            row = inaturalist.adapt(self.inaturalist)
            manifest = root / "manifest.jsonl"
            manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
            prediction = {
                "sample_id": row["sample_id"],
                "source_dataset": row["source_dataset"],
                "task_type": row["task_type"],
                "mode": "correct-image",
                "used_image_id": row["image_id"],
                "raw_output": row["sft_target"],
                "answer_correct": 1,
            }
            predictions = root / "predictions.jsonl"
            predictions.write_text(json.dumps(prediction) + "\n", encoding="utf-8")
            output = root / "validation.json"
            with self.assertRaises(SystemExit) as exit_status:
                self.invoke(
                    validate_predictions,
                    "--manifest",
                    manifest,
                    "--predictions",
                    predictions,
                    "--output",
                    output,
                    "--mode",
                    "correct-image",
                    "--expected-rows",
                    "1",
                )
            self.assertEqual(exit_status.exception.code, 0)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["status"], "passed")

    def test_canonical_prediction_json_mode_is_independently_rescored(self):
        from pipelines.evaluate.validate_canonical_predictions import main as validate_predictions

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            row = inaturalist.adapt(self.inaturalist, answer_mode="json")
            manifest = root / "manifest.jsonl"
            manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
            prediction = {
                "sample_id": row["sample_id"],
                "source_dataset": row["source_dataset"],
                "task_type": row["task_type"],
                "mode": "correct-image",
                "used_image_id": row["image_id"],
                "raw_output": row["sft_target"],
                "schema_valid": 1,
                "answer_correct": 1,
                "path_exact": 1,
                "full_chain": 1,
            }
            predictions = root / "predictions.jsonl"
            predictions.write_text(json.dumps(prediction) + "\n", encoding="utf-8")
            output = root / "validation.json"
            with self.assertRaises(SystemExit) as exit_status:
                self.invoke(
                    validate_predictions,
                    "--manifest",
                    manifest,
                    "--predictions",
                    predictions,
                    "--output",
                    output,
                    "--mode",
                    "correct-image",
                    "--answer-mode",
                    "json",
                    "--expected-rows",
                    "1",
                )
            self.assertEqual(exit_status.exception.code, 0)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["answer_correct"], 1)
            self.assertEqual(report["full_chain"], 1)


if __name__ == "__main__":
    unittest.main()
