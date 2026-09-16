import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipelines.evaluate.validate_predictions import main as validate_main


class ValidationHelpers(unittest.TestCase):
    def test_minimal_prediction_is_independently_validated_without_fixed_count(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            hop = {"subject": "A", "predicate": "related_to", "object": "B"}
            row = {
                "sample_id": "sample-1",
                "group_id": "group-1",
                "hop_count": 1,
                "rotation": 0,
                "image_id": "image-1",
                "gold_root_choice": "A",
                "gold_answer_choice": "B",
                "gold_evidence_ids": ["e1"],
                "gold_path": [hop],
            }
            raw = {
                "visual_features": ["visible landmark"],
                "root_choice": "A",
                "evidence_ids": ["e1"],
                "hops": [hop],
                "answer_choice": "B",
            }
            prediction = {
                "sample_id": "sample-1",
                "mode": "correct-image",
                "used_image_id": "image-1",
                "raw_output": json.dumps(raw),
                "schema_valid": 1,
                "root_correct": 1,
                "evidence_f1": 1.0,
                "evidence_exact": 1,
                "path_edge_accuracy": 1.0,
                "path_continuous": 1,
                "path_exact": 1,
                "answer_correct": 1,
                "full_chain": 1,
                "rotation_consistency": 0,
            }
            manifest = root / "dev.jsonl"
            predictions = root / "predictions.jsonl"
            output = root / "validation.json"
            manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
            predictions.write_text(json.dumps(prediction) + "\n", encoding="utf-8")
            original = sys.argv
            try:
                sys.argv = [
                    "validate", "--manifest", str(manifest),
                    "--predictions", str(predictions), "--output", str(output),
                    "--mode", "correct-image",
                ]
                with self.assertRaisesRegex(SystemExit, "0"):
                    validate_main()
            finally:
                sys.argv = original
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["rows"], 1)


if __name__ == "__main__":
    unittest.main()
