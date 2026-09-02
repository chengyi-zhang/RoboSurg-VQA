#!/usr/bin/env python3
"""Evaluate shared-VQA predictions against adjudicated references."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE = (
    PROJECT_ROOT
    / "data"
    / "human_reference"
    / "reference.csv"
)
DEFAULT_PREDICTIONS = (
    PROJECT_ROOT
    / "results"
    / "shared_vqa"
    / "predictions.csv"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "shared_vqa_human_reference"

TASKS = {
    "q4.bleeding": ("q4_bleeding", ["no", "yes"]),
    "q5": ("q5_value", ["clear", "degraded"]),
    "q6": ("q6_glare", ["no", "yes"]),
}

CONDITIONS = [
    ("answer_frequency", "canonical"),
    ("question_only", "canonical"),
    ("image_only", "canonical"),
    ("image_question", "canonical"),
    ("image_question", "heldout_paraphrase"),
    ("image_question", "wrong_question"),
]

COMPARISONS = [
    (("image_question", "canonical"), ("answer_frequency", "canonical")),
    (("image_question", "canonical"), ("question_only", "canonical")),
    (("image_question", "canonical"), ("image_only", "canonical")),
    (("image_question", "heldout_paraphrase"), ("image_question", "canonical")),
    (("image_question", "wrong_question"), ("image_question", "canonical")),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260831)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def harmonise_q5(value: str) -> str:
    if value in {"blurry", "mixed", "reflective"}:
        return "degraded"
    return value


def fixed_metrics(
    references: Iterable[str], predictions: Iterable[str], labels: list[str]
) -> dict[str, float]:
    y_true = list(references)
    y_pred = list(predictions)
    if len(y_true) != len(y_pred) or not y_true:
        raise ValueError("Metric inputs must be non-empty and aligned")

    per_class_f1 = []
    per_class_recall = []
    for label in labels:
        tp = sum(a == label and b == label for a, b in zip(y_true, y_pred))
        fp = sum(a != label and b == label for a, b in zip(y_true, y_pred))
        fn = sum(a == label and b != label for a, b in zip(y_true, y_pred))
        f1_denominator = 2 * tp + fp + fn
        per_class_f1.append(0.0 if f1_denominator == 0 else 2 * tp / f1_denominator)
        recall_denominator = tp + fn
        per_class_recall.append(
            0.0 if recall_denominator == 0 else tp / recall_denominator
        )

    return {
        "accuracy": sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true),
        "macro_f1_fixed_labels": float(np.mean(per_class_f1)),
        "balanced_accuracy_fixed_labels": float(np.mean(per_class_recall)),
        "invalid_answer_rate": sum(value not in labels for value in y_pred)
        / len(y_pred),
    }


def task_scores(rows: list[dict[str, str]]) -> tuple[list[dict[str, object]], float]:
    metrics = []
    values = []
    for task, (_, labels) in TASKS.items():
        task_rows = [row for row in rows if row["task"] == task]
        result = fixed_metrics(
            [row["human_reference"] for row in task_rows],
            [row["prediction"] for row in task_rows],
            labels,
        )
        metrics.append(
            {
                "task": task,
                "n": len(task_rows),
                "reference_class_count": len(
                    {row["human_reference"] for row in task_rows}
                ),
                **result,
            }
        )
        values.append(result["macro_f1_fixed_labels"])
    return metrics, float(np.mean(values))


def main() -> None:
    args = parse_args()
    reference_path = args.reference.resolve()
    predictions_path = args.predictions.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    reference_rows = [
        row
        for row in read_csv(reference_path)
        if row["split"] == "test"
        and row["representative_stratum"] == "1"
        and row["global_unusable_frame_yes_no"] == "no"
    ]
    if len(reference_rows) != 40:
        raise RuntimeError(f"Expected 40 held-out representative frames, found {len(reference_rows)}")
    sequences = sorted({row["video_key"] for row in reference_rows})
    if len(sequences) != 6:
        raise RuntimeError(f"Expected six held-out source sequences, found {len(sequences)}")

    prediction_rows = read_csv(predictions_path)
    prediction_lookup = {
        (
            row["condition"],
            row["wording"],
            row["task"],
            row["record_id"],
        ): row["prediction"]
        for row in prediction_rows
    }

    joined_by_condition: dict[tuple[str, str], list[dict[str, str]]] = {}
    joined_all = []
    for condition, wording in CONDITIONS:
        joined = []
        for reference in reference_rows:
            for task, (column, _) in TASKS.items():
                key = (condition, wording, task, reference["record_id"])
                if key not in prediction_lookup:
                    raise RuntimeError(f"Missing frozen prediction: {key}")
                human_value = reference[column]
                prediction = prediction_lookup[key]
                if task == "q5":
                    human_value = harmonise_q5(human_value)
                    prediction = harmonise_q5(prediction)
                row = {
                    "condition": condition,
                    "wording": wording,
                    "sample_id": reference["sample_id"],
                    "record_id": reference["record_id"],
                    "source_sequence_id": reference["video_key"],
                    "site": reference["site"],
                    "task": task,
                    "human_reference": human_value,
                    "prediction": prediction,
                }
                joined.append(row)
                joined_all.append(row)
        joined_by_condition[(condition, wording)] = joined

    point_by_condition = {}
    task_metric_rows = []
    aggregate_rows = []
    for condition, wording in CONDITIONS:
        metrics, aggregate = task_scores(joined_by_condition[(condition, wording)])
        point_by_condition[(condition, wording)] = aggregate
        for row in metrics:
            task_metric_rows.append(
                {"condition": condition, "wording": wording, **row}
            )
        aggregate_rows.append(
            {
                "condition": condition,
                "wording": wording,
                "task_group": "heldout_human_reference_three_tasks",
                "task_count": 3,
                "mean_macro_f1_fixed_labels": aggregate,
            }
        )

    rows_by_sequence_condition: dict[
        tuple[str, str], dict[str, list[dict[str, str]]]
    ] = {}
    for key, rows in joined_by_condition.items():
        grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            grouped[row["source_sequence_id"]].append(row)
        rows_by_sequence_condition[key] = grouped

    rng = np.random.default_rng(args.seed)
    replicate_scores = {
        key: np.empty(args.replicates, dtype=np.float64) for key in CONDITIONS
    }
    for replicate in range(args.replicates):
        selected = rng.choice(sequences, size=len(sequences), replace=True)
        for key in CONDITIONS:
            sampled_rows = []
            for sequence in selected:
                sampled_rows.extend(rows_by_sequence_condition[key][str(sequence)])
            _, score = task_scores(sampled_rows)
            replicate_scores[key][replicate] = score

    for row in aggregate_rows:
        key = (str(row["condition"]), str(row["wording"]))
        low, high = np.percentile(replicate_scores[key], [2.5, 97.5])
        row["ci_low"] = float(low)
        row["ci_high"] = float(high)

    delta_rows = []
    for condition_a, condition_b in COMPARISONS:
        deltas = replicate_scores[condition_a] - replicate_scores[condition_b]
        low, high = np.percentile(deltas, [2.5, 97.5])
        delta_rows.append(
            {
                "condition_a": condition_a[0],
                "wording_a": condition_a[1],
                "condition_b": condition_b[0],
                "wording_b": condition_b[1],
                "estimate_a": point_by_condition[condition_a],
                "estimate_b": point_by_condition[condition_b],
                "delta_macro_f1": point_by_condition[condition_a]
                - point_by_condition[condition_b],
                "ci_low": float(low),
                "ci_high": float(high),
            }
        )

    joined_path = output / "joined_predictions.csv"
    task_path = output / "task_metrics.csv"
    aggregate_path = output / "aggregate_metrics.csv"
    delta_path = output / "bootstrap_deltas.csv"
    write_csv(joined_path, joined_all)
    write_csv(task_path, task_metric_rows)
    write_csv(aggregate_path, aggregate_rows)
    write_csv(delta_path, delta_rows)

    run = {
        "analysis": "shared_vqa_human_reference",
        "heldout_representative_frames": len(reference_rows),
        "source_sequences": sequences,
        "tasks": list(TASKS),
        "fixed_labels": {task: labels for task, (_, labels) in TASKS.items()},
        "q5_harmonisation": {
            "clear": "clear",
            "blurry": "degraded",
            "mixed": "degraded",
            "reflective": "degraded",
        },
        "bootstrap": {
            "replicates": args.replicates,
            "seed": args.seed,
            "unit": "source_sequence",
            "design": "six held-out sequences sampled with replacement",
        },
        "inputs": {
            "human_reference": reference_path.relative_to(PROJECT_ROOT).as_posix(),
            "human_reference_sha256": sha256(reference_path),
            "frozen_predictions": predictions_path.relative_to(PROJECT_ROOT).as_posix(),
            "frozen_predictions_sha256": sha256(predictions_path),
        },
        "outputs": {
            path.name: sha256(path)
            for path in [joined_path, task_path, aggregate_path, delta_path]
        },
    }
    with (output / "run.json").open("w", encoding="utf-8") as handle:
        json.dump(run, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps(run, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
