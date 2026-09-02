"""Check sequence assignments and question variants in the released records."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from validate_release import repository_files


class RecordTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with (ROOT / "data/vqa_records.jsonl").open(encoding="utf-8") as handle:
            cls.rows = [json.loads(line) for line in handle if line.strip()]

    def test_record_counts_and_uniqueness(self):
        self.assertEqual(len(self.rows), 59552)
        self.assertEqual(len({row["qa_id"] for row in self.rows}), 59552)
        self.assertEqual(len({row["record_id"] for row in self.rows}), 5632)

    def test_sequence_partition(self):
        train = {row["source_sequence_id"] for row in self.rows if row["source_split"] == "train"}
        test = {row["source_sequence_id"] for row in self.rows if row["source_split"] == "test"}
        validation = {row["source_sequence_id"] for row in self.rows if row["validation_role"] == "validation"}
        self.assertEqual(len(train), 23)
        self.assertEqual(len(test), 6)
        self.assertFalse(train & test)
        self.assertEqual(validation, {"train|site1/video3", "train|site2/video2",
                                      "train|site2/video11", "train|site2/video12"})
        self.assertTrue(validation <= train)

    def test_heldout_wording_not_used_in_training(self):
        questions = json.loads((ROOT / "configs/questions.json").read_text())["tasks"]
        for task, spec in questions.items():
            with self.subTest(task=task):
                variants = [spec["canonical"], *spec["train_paraphrases"], spec["heldout_paraphrase"]]
                self.assertEqual(len(variants), 4)
                self.assertEqual(len(set(variants)), 4)

    def test_checkpoint_revisions(self):
        config = json.loads((ROOT / "configs/shared_vqa.json").read_text())
        self.assertEqual(config["model_revision"], "9f341de24bfb00180f1b847274256e9b65a3a32e")
        self.assertEqual(config["text_model_revision"], "d673b8835373c6fa116d6d8006b33d48734e305d")

    def test_inventory_skips_git_and_local_caches(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative in ("README.md", ".github/workflows/validate.yml", ".git/config",
                             ".venv/ignored.txt", "outputs/result.csv", ".cache/ignored.txt"):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            self.assertEqual({path.relative_to(root).as_posix() for path in repository_files(root)},
                             {"README.md", ".github/workflows/validate.yml"})


if __name__ == "__main__":
    unittest.main()
