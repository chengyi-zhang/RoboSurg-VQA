#!/usr/bin/env python3
"""Analyse taskwise baselines, label shift, and paired bootstrap differences."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


import numpy as np


WORKSPACE = Path(__file__).resolve().parents[1]
RESULTS = WORKSPACE / "results" / "taskwise"
PREDICTIONS = RESULTS / "predictions.csv"
ANNOTATIONS = WORKSPACE / "data" / "frame_annotations.jsonl"
MANIFEST = WORKSPACE / "data" / "manifest.jsonl"
Q8_TARGETS = WORKSPACE / "data" / "mask_targets.csv"
TASK_APPLICABILITY = WORKSPACE / "data" / "task_applicability.csv"
Q8_TASK = "q8.mask_center_5pct_v1"
Q8_LABELS: dict[str, str] = {}
BOOTSTRAP_REPLICATES = 5000
BOOTSTRAP_SEED = 20260824
COMPARISONS = [
    ("rgb_linear", "majority_prior"),
    ("mask_linear", "majority_prior"),
    ("rgb_mask_linear", "rgb_linear"),
    ("rgb_mask_linear", "mask_linear"),
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def task_target(row: dict[str, Any], task: str) -> str:
    if task == Q8_TASK:
        return Q8_LABELS[row["record_id"]]
    if task.startswith("q4."):
        return str(row["answers"]["q4"]["value"][task.split(".", 1)[1]])
    return str(row["answers"][task]["value"])


def confusion_counts(
    reference: np.ndarray, prediction: np.ndarray, labels: list[str]
) -> np.ndarray:
    label_index = {label: index for index, label in enumerate(labels)}
    true_index = np.fromiter(
        (label_index[value] for value in reference), dtype=np.int64, count=len(reference)
    )
    pred_index = np.fromiter(
        (label_index[value] for value in prediction), dtype=np.int64, count=len(prediction)
    )
    encoded = true_index * len(labels) + pred_index
    return np.bincount(encoded, minlength=len(labels) ** 2).reshape(len(labels), len(labels))


def macro_f1_from_confusion(confusion: np.ndarray) -> np.ndarray:
    true_positive = np.diagonal(confusion, axis1=-2, axis2=-1)
    denominator = confusion.sum(axis=-2) + confusion.sum(axis=-1)
    per_class = np.divide(
        2.0 * true_positive,
        denominator,
        out=np.zeros_like(denominator, dtype=np.float64),
        where=denominator != 0,
    )
    return per_class.mean(axis=-1)


def macro_f1(reference: np.ndarray, prediction: np.ndarray, labels: list[str]) -> float:
    return float(macro_f1_from_confusion(confusion_counts(reference, prediction, labels)))


def bootstrap_video_counts(videos: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    unique_videos = np.unique(videos)
    rng = np.random.default_rng(seed)
    counts = np.zeros((BOOTSTRAP_REPLICATES, len(unique_videos)), dtype=np.int16)
    for replicate in range(BOOTSTRAP_REPLICATES):
        sampled = rng.choice(unique_videos, size=len(unique_videos), replace=True)
        counts[replicate] = [np.count_nonzero(sampled == video) for video in unique_videos]
    return unique_videos, counts


def main() -> None:
    global Q8_LABELS
    for required in (
        PREDICTIONS,
        ANNOTATIONS,
        MANIFEST,
        Q8_TARGETS,
        TASK_APPLICABILITY,
        RESULTS / "metrics.csv",
    ):
        if not required.is_file():
            raise FileNotFoundError(required)

    prediction_rows = read_csv(PREDICTIONS)
    annotations = read_jsonl(ANNOTATIONS)
    manifest = read_jsonl(MANIFEST)
    annotation_by_id = {row["record_id"]: row for row in annotations}
    manifest_by_id = {row["record_id"]: row for row in manifest}
    q8_rows = read_csv(Q8_TARGETS)
    Q8_LABELS = {row["record_id"]: row["label"] for row in q8_rows}
    applicability = {
        (row["site"], row["target"])
        for row in read_csv(TASK_APPLICABILITY)
        if row["primary_scored"] == "1"
    }
    if set(Q8_LABELS) != set(annotation_by_id):
        raise RuntimeError("Deterministic Q8 records do not match annotations")
    tasks = sorted({row["task"] for row in prediction_rows})
    conditions = sorted({row["condition"] for row in prediction_rows})
    if "q1" in tasks or "q3" in tasks:
        raise RuntimeError("Q1/Q3 unexpectedly appear in scored predictions")

    by_task_condition: dict[tuple[str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in prediction_rows:
        key = (row["task"], row["condition"])
        if row["record_id"] in by_task_condition[key]:
            raise RuntimeError(f"Duplicate prediction: {key} {row['record_id']}")
        by_task_condition[key][row["record_id"]] = row

    task_ids: dict[str, list[str]] = {}
    references: dict[str, np.ndarray] = {}
    predictions: dict[tuple[str, str], np.ndarray] = {}
    videos: dict[str, np.ndarray] = {}
    sites: dict[str, np.ndarray] = {}
    datasets: dict[str, np.ndarray] = {}
    labels: dict[str, list[str]] = {}
    for task in tasks:
        ids = sorted(by_task_condition[(task, "majority_prior")])
        if not ids:
            raise RuntimeError(f"No prediction IDs for {task}")
        for condition in conditions:
            if set(by_task_condition[(task, condition)]) != set(ids):
                raise RuntimeError(f"Condition ID mismatch for {task}: {condition}")
        task_ids[task] = ids
        references[task] = np.asarray(
            [by_task_condition[(task, "majority_prior")][record_id]["reference"] for record_id in ids]
        )
        videos[task] = np.asarray([manifest_by_id[record_id]["video_key"] for record_id in ids])
        sites[task] = np.asarray([annotation_by_id[record_id]["site"] for record_id in ids])
        datasets[task] = np.asarray([annotation_by_id[record_id]["dataset"] for record_id in ids])
        for condition in conditions:
            rows = by_task_condition[(task, condition)]
            condition_reference = np.asarray([rows[record_id]["reference"] for record_id in ids])
            if not np.array_equal(condition_reference, references[task]):
                raise RuntimeError(f"Reference mismatch across conditions for {task}")
            predictions[(task, condition)] = np.asarray(
                [rows[record_id]["prediction"] for record_id in ids]
            )
        eligible_rows = [
            row for row in annotations if (row["site"], task) in applicability
        ]
        labels[task] = sorted({task_target(row, task) for row in eligible_rows})

    evaluable_tasks = [
        task for task in tasks if len(set(references[task])) >= 2 and task != Q8_TASK
    ]
    bootstrap_counts: dict[str, np.ndarray] = {}
    unique_videos: dict[str, np.ndarray] = {}
    video_confusions: dict[tuple[str, str], np.ndarray] = {}
    for task_index, task in enumerate(tasks):
        task_videos, counts = bootstrap_video_counts(videos[task], BOOTSTRAP_SEED + task_index)
        unique_videos[task] = task_videos
        bootstrap_counts[task] = counts
        indices_by_video = {
            video: np.flatnonzero(videos[task] == video) for video in task_videos
        }
        for condition in conditions:
            video_confusions[(task, condition)] = np.stack(
                [
                    confusion_counts(
                        references[task][indices_by_video[video]],
                        predictions[(task, condition)][indices_by_video[video]],
                        labels[task],
                    )
                    for video in task_videos
                ]
            )

    delta_rows: list[dict[str, Any]] = []
    for condition_a, condition_b in COMPARISONS:
        task_boot_deltas: list[np.ndarray] = []
        for task in tasks:
            point_delta = macro_f1(
                references[task], predictions[(task, condition_a)], labels[task]
            ) - macro_f1(references[task], predictions[(task, condition_b)], labels[task])
            confusion_a = np.einsum(
                "bv,vij->bij",
                bootstrap_counts[task],
                video_confusions[(task, condition_a)],
                optimize=True,
            )
            confusion_b = np.einsum(
                "bv,vij->bij",
                bootstrap_counts[task],
                video_confusions[(task, condition_b)],
                optimize=True,
            )
            boot_delta = macro_f1_from_confusion(confusion_a) - macro_f1_from_confusion(confusion_b)
            lower, upper = np.percentile(boot_delta, [2.5, 97.5])
            delta_rows.append(
                {
                    "task": task,
                    "condition_a": condition_a,
                    "condition_b": condition_b,
                    "delta_macro_f1_a_minus_b": point_delta,
                    "ci_2.5": float(lower),
                    "ci_97.5": float(upper),
                    "direction_clear_at_95pct": int(lower > 0 or upper < 0),
                    "test_video_count": len(unique_videos[task]),
                    "bootstrap_design": "task_specific_video_bootstrap",
                }
            )
            if task in evaluable_tasks:
                task_boot_deltas.append(boot_delta)

        point_aggregate = float(
            np.mean(
                [
                    macro_f1(references[task], predictions[(task, condition_a)], labels[task])
                    - macro_f1(references[task], predictions[(task, condition_b)], labels[task])
                    for task in evaluable_tasks
                ]
            )
        )
        aggregate_boot = np.mean(task_boot_deltas, axis=0)
        lower, upper = np.percentile(aggregate_boot, [2.5, 97.5])
        delta_rows.append(
            {
                "task": "MEAN_TEST_EVALUABLE_TASKS",
                "condition_a": condition_a,
                "condition_b": condition_b,
                "delta_macro_f1_a_minus_b": point_aggregate,
                "ci_2.5": float(lower),
                "ci_97.5": float(upper),
                "direction_clear_at_95pct": int(lower > 0 or upper < 0),
                "test_video_count": "task-specific",
                "bootstrap_design": "task_stratified_video_bootstrap",
            }
        )

    delta_path = RESULTS / "paired_bootstrap_deltas.csv"
    with delta_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(delta_rows[0]))
        writer.writeheader()
        writer.writerows(delta_rows)

    shift_rows: list[dict[str, Any]] = []
    for task in tasks:
        eligible = [row for row in annotations if (row["site"], task) in applicability]
        train = Counter(task_target(row, task) for row in eligible if row["split"] == "train")
        test = Counter(task_target(row, task) for row in eligible if row["split"] == "test")
        all_labels = sorted(set(train) | set(test))
        train_total = sum(train.values())
        test_total = sum(test.values())
        tvd = 0.5 * sum(
            abs(train[label] / train_total - test[label] / test_total) for label in all_labels
        )
        shift_rows.append(
            {
                "task": task,
                "eligible_sites": " | ".join(sorted({row["site"] for row in eligible})),
                "train_n": train_total,
                "test_n": test_total,
                "total_variation_distance": tvd,
                "train_classes": len(train),
                "test_classes": len(test),
                "missing_test_labels": " | ".join(sorted(set(train) - set(test))),
            }
        )
    shift_path = RESULTS / "label_shift.csv"
    with shift_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(shift_rows[0]))
        writer.writeheader()
        writer.writerows(shift_rows)

    by_dataset_rows: list[dict[str, Any]] = []
    for task in tasks:
        for dataset in sorted(set(datasets[task])):
            indices = np.flatnonzero(datasets[task] == dataset)
            for condition in conditions:
                by_dataset_rows.append(
                    {
                        "task": task,
                        "condition": condition,
                        "dataset": dataset,
                        "test_n": len(indices),
                        "test_video_count": len(np.unique(videos[task][indices])),
                        "test_classes": len(set(references[task][indices])),
                        "macro_f1_fixed_global_labels": macro_f1(
                            references[task][indices],
                            predictions[(task, condition)][indices],
                            labels[task],
                        ),
                    }
                )
    by_dataset_path = RESULTS / "metrics_by_source.csv"
    with by_dataset_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(by_dataset_rows[0]))
        writer.writeheader()
        writer.writerows(by_dataset_rows)

    analysis_config = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tasks": tasks,
        "test_evaluable_tasks": evaluable_tasks,
        "conditions": conditions,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_base_seed": BOOTSTRAP_SEED,
        "aggregate_bootstrap_design": "task_stratified_video_bootstrap",
        "outputs": [
            delta_path.name,
            shift_path.name,
            by_dataset_path.name,
        ],
    }
    (RESULTS / "analysis.json").write_text(
        json.dumps(analysis_config, indent=2), encoding="utf-8"
    )
    print(json.dumps(analysis_config, indent=2))


if __name__ == "__main__":
    main()
