"""Regression tests for fixed-label metrics and the frozen human comparison."""

import csv
import json
import sys
import unittest
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate_human_reference as human
import evaluate_shared_vqa as shared
from analyse_task_groups import f1_from_counts


class FixedLabelTests(unittest.TestCase):
    def test_absent_class_stays_in_denominator(self):
        labels = ["clear", "degraded"]
        truth = ["clear"] * 8
        self.assertEqual(human.metrics(truth, truth, labels)["macro_f1"], 0.5)
        self.assertEqual(shared.fixed_metrics(truth, truth, labels)["macro_f1_fixed_labels"], 0.5)
        counts = Counter({("clear", "clear"): 8})
        self.assertEqual(human.macro_f1_from_confusion(counts, labels), 0.5)
        self.assertEqual(f1_from_counts(counts, labels), 0.5)

    def test_q5_harmonisation(self):
        for value in ("blurry", "reflective", "mixed"):
            self.assertEqual(human.map_label("q5", value, "human"), "degraded")
            self.assertEqual(shared.harmonise_q5(value), "degraded")
        self.assertEqual(shared.harmonise_q5("clear"), "clear")

    def test_invalid_answers_are_errors(self):
        result = shared.fixed_metrics(["yes", "no"], ["clear", "no"], ["no", "yes"])
        self.assertEqual(result["accuracy"], 0.5)
        self.assertEqual(result["invalid_answer_rate"], 0.5)
        self.assertEqual(result["macro_f1_fixed_labels"], 0.5)

    def test_candidate_and_human_bootstraps_keep_absent_classes(self):
        records = [
            {"sequence_key": str(i), "human": "clear", "prediction": "clear",
             "human_comparison": "clear", "machine_comparison": "clear"}
            for i in range(6)
        ]
        for bootstrap in (human.cluster_bootstrap_human, human.cluster_bootstrap_candidate_human):
            interval = bootstrap(records, ["clear", "degraded"], 7)
            self.assertEqual(interval["macro_f1_ci_low"], 0.5)
            self.assertEqual(interval["macro_f1_ci_high"], 0.5)

    def test_all_twelve_human_results_and_intervals(self):
        references = human.read_csv(ROOT / "data/human_reference/reference.csv")
        predictions = human.read_csv(ROOT / "results/taskwise/predictions.csv")
        rows, aggregates, _ = human.build_harmonised_human_baselines(references, predictions)
        expected = human.read_csv(ROOT / "results/human_reference/taskwise_human_reference_metrics.csv")
        lookup = {(row["task"], row["condition"]): row for row in expected}
        self.assertEqual(len(rows), 12)
        for row in rows:
            saved = lookup[(row["task"], row["condition"])]
            for field in ("accuracy", "macro_f1", "balanced_accuracy", "accuracy_ci_low",
                          "accuracy_ci_high", "macro_f1_ci_low", "macro_f1_ci_high"):
                with self.subTest(task=row["task"], condition=row["condition"], field=field):
                    self.assertAlmostEqual(row[field], float(saved[field]), places=12)
        for condition, value in {"majority_prior": 0.450, "rgb_linear": 0.527,
                                 "rgb_mask_linear": 0.561}.items():
            row = next(row for row in aggregates if row["condition"] == condition)
            self.assertEqual(round(row["mean_macro_f1"], 3), value)

    def test_shared_aggregates_from_frozen_predictions(self):
        config = json.loads((ROOT / "configs/shared_vqa.json").read_text())
        labels = json.loads((ROOT / "data/vqa_summary.json").read_text())["task_labels"]
        counts = defaultdict(Counter)
        with (ROOT / "results/shared_vqa/predictions.csv").open(newline="") as handle:
            for row in csv.DictReader(handle):
                counts[(row["condition"], row["wording"], row["task"])][
                    (row["reference"], row["prediction"])
                ] += 1
        groups = {"all_applicable_tasks": list(labels),
                  "primary_discriminative_tasks": config["primary_tasks"],
                  "perceptual_tasks": config["perceptual_tasks"]}
        expected = human.read_csv(ROOT / "results/shared_vqa/aggregate_metrics.csv")
        for row in expected:
            tasks = groups[row["task_group"]]
            scores = [f1_from_counts(counts[(row["condition"], row["wording"], task)],
                                    labels[task]) for task in tasks]
            with self.subTest(condition=row["condition"], wording=row["wording"], group=row["task_group"]):
                self.assertAlmostEqual(sum(scores) / len(scores),
                                       float(row["mean_macro_f1_fixed_labels"]), places=12)


if __name__ == "__main__":
    unittest.main()
