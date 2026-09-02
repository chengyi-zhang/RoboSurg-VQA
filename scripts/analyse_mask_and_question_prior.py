#!/usr/bin/env python3
"""Audit direct mask-derived targets and the fixed-question prior baseline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


CONDITIONS = ["majority_prior", "rgb_linear", "mask_linear", "rgb_mask_linear"]
TASK_SETS = {
    "candidate_visual_five": [
        "q4.bleeding",
        "q4.occlusion",
        "q5",
        "q6",
        "q7",
    ],
    "discriminative_without_q8": [
        "q2",
        "q4.bleeding",
        "q4.occlusion",
        "q5",
        "q6",
        "q7",
    ],
    "direct_mask_tasks": ["q2", "q8.mask_center_5pct_v1"],
}
COMPARISONS = [
    ("rgb_linear", "majority_prior"),
    ("mask_linear", "majority_prior"),
    ("rgb_mask_linear", "rgb_linear"),
    ("rgb_mask_linear", "mask_linear"),
]
BOOTSTRAP_REPLICATES = 5000
BOOTSTRAP_SEED = 20260825


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().lower()


def metric_bundle(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> dict[str, float]:
    recalls = []
    f1s = []
    for label in labels:
        tp = int(np.sum((y_true == label) & (y_pred == label)))
        fp = int(np.sum((y_true != label) & (y_pred == label)))
        fn = int(np.sum((y_true == label) & (y_pred != label)))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        recalls.append(recall)
        f1s.append(f1)
    return {
        "accuracy": float(np.mean(y_true == y_pred)),
        "macro_f1": float(np.mean(f1s)),
        "balanced_accuracy": float(np.mean(recalls)),
    }


def build_task_condition(rows: list[dict[str, str]]) -> dict[tuple[str, str], list[dict[str, str]]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["task"], row["condition"])].append(row)
    for key in grouped:
        grouped[key].sort(key=lambda row: row["record_id"])
    return grouped


def validate_alignment(grouped: dict[tuple[str, str], list[dict[str, str]]]) -> None:
    tasks = sorted({task for task, _ in grouped})
    for task in tasks:
        base = grouped[(task, CONDITIONS[0])]
        base_ids = [row["record_id"] for row in base]
        base_refs = [row["reference"] for row in base]
        for condition in CONDITIONS[1:]:
            rows = grouped[(task, condition)]
            if [row["record_id"] for row in rows] != base_ids:
                raise RuntimeError(f"Prediction alignment failed for {task}/{condition}")
            if [row["reference"] for row in rows] != base_refs:
                raise RuntimeError(f"Reference alignment failed for {task}/{condition}")


def aggregate_metrics(
    grouped: dict[tuple[str, str], list[dict[str, str]]],
    tasks: list[str],
    condition: str,
    task_labels: dict[str, list[str]],
) -> dict[str, float]:
    bundles = []
    for task in tasks:
        rows = grouped[(task, condition)]
        y_true = np.asarray([row["reference"] for row in rows])
        y_pred = np.asarray([row["prediction"] for row in rows])
        bundles.append(metric_bundle(y_true, y_pred, task_labels[task]))
    return {name: float(np.mean([bundle[name] for bundle in bundles])) for name in bundles[0]}


def paired_task_video_bootstrap(
    grouped: dict[tuple[str, str], list[dict[str, str]]],
    tasks: list[str],
    left: str,
    right: str,
    seed: int,
    task_labels: dict[str, list[str]],
) -> tuple[float, float, float]:
    task_data = []
    for task in tasks:
        left_rows = grouped[(task, left)]
        right_rows = grouped[(task, right)]
        y_true = np.asarray([row["reference"] for row in left_rows])
        left_pred = np.asarray([row["prediction"] for row in left_rows])
        right_pred = np.asarray([row["prediction"] for row in right_rows])
        videos = np.asarray([row["video_key"] for row in left_rows])
        labels = task_labels[task]
        unique_videos = sorted(set(videos))
        indices = {video: np.flatnonzero(videos == video) for video in unique_videos}
        task_data.append((y_true, left_pred, right_pred, labels, unique_videos, indices))

    cluster_sets = [set(item[4]) for item in task_data]
    shared_cluster_set = all(
        cluster_set == cluster_sets[0] for cluster_set in cluster_sets[1:]
    )
    shared_videos = sorted(cluster_sets[0]) if shared_cluster_set else []

    point_left = aggregate_metrics(grouped, tasks, left, task_labels)["macro_f1"]
    point_right = aggregate_metrics(grouped, tasks, right, task_labels)["macro_f1"]
    point = point_left - point_right
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(BOOTSTRAP_REPLICATES):
        shared_sample = (
            rng.choice(shared_videos, size=len(shared_videos), replace=True)
            if shared_cluster_set
            else None
        )
        task_deltas = []
        for y_true, left_pred, right_pred, labels, videos, indices in task_data:
            sampled = (
                shared_sample
                if shared_sample is not None
                else rng.choice(videos, size=len(videos), replace=True)
            )
            sampled_indices = np.concatenate([indices[video] for video in sampled])
            left_score = metric_bundle(y_true[sampled_indices], left_pred[sampled_indices], labels)["macro_f1"]
            right_score = metric_bundle(y_true[sampled_indices], right_pred[sampled_indices], labels)["macro_f1"]
            task_deltas.append(left_score - right_score)
        deltas.append(float(np.mean(task_deltas)))
    return point, float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def leave_one_video_out(
    grouped: dict[tuple[str, str], list[dict[str, str]]],
    tasks: list[str],
    left: str,
    right: str,
    task_labels: dict[str, list[str]],
) -> list[dict[str, object]]:
    videos = sorted({row["video_key"] for row in grouped[(tasks[0], left)]})
    output = []
    for omitted in videos:
        task_deltas = []
        for task in tasks:
            left_rows = [row for row in grouped[(task, left)] if row["video_key"] != omitted]
            right_rows = [row for row in grouped[(task, right)] if row["video_key"] != omitted]
            y_true = np.asarray([row["reference"] for row in left_rows])
            left_pred = np.asarray([row["prediction"] for row in left_rows])
            right_pred = np.asarray([row["prediction"] for row in right_rows])
            labels = task_labels[task]
            task_deltas.append(
                metric_bundle(y_true, left_pred, labels)["macro_f1"]
                - metric_bundle(y_true, right_pred, labels)["macro_f1"]
            )
        output.append(
            {
                "omitted_video_key": omitted,
                "remaining_video_clusters": len(videos) - 1,
                "delta_macro_f1": float(np.mean(task_deltas)),
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args()
    project = args.project.resolve()
    source = project / "results" / "taskwise" / "predictions.csv"
    config_path = project / "results" / "taskwise" / "run.json"
    output = project / "results" / "mask_question_analysis"
    output.mkdir(parents=True, exist_ok=True)

    rows = read_csv(source)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    task_labels = {
        task: [str(label) for label in details["labels"]]
        for task, details in config["task_summary"].items()
    }
    grouped = build_task_condition(rows)
    validate_alignment(grouped)
    available_tasks = sorted({task for task, _ in grouped})
    required_tasks = sorted({task for tasks in TASK_SETS.values() for task in tasks})
    missing = sorted(set(required_tasks) - set(available_tasks))
    if missing:
        raise RuntimeError(f"Missing required tasks: {missing}")

    partition_rows = []
    for task in available_tasks:
        if task in TASK_SETS["direct_mask_tasks"]:
            partition = "direct_mask_derived_diagnostic"
        elif task in TASK_SETS["candidate_visual_five"]:
            partition = "non_direct_mask_derived"
        else:
            partition = "single_class_or_descriptive"
        task_rows = grouped[(task, "majority_prior")]
        references = [row["reference"] for row in task_rows]
        partition_rows.append(
            {
                "task": task,
                "partition": partition,
                "test_n": len(task_rows),
                "test_video_clusters": len({row["video_key"] for row in task_rows}),
                "test_reference_classes": len(set(references)),
                "test_reference_distribution": json.dumps(dict(sorted(Counter(references).items())), sort_keys=True),
                "headline_discrimination_eligible": int(len(set(references)) >= 2 and partition != "direct_mask_derived_diagnostic"),
            }
        )
    write_csv(
        output / "task_partition.csv",
        partition_rows,
        [
            "task",
            "partition",
            "test_n",
            "test_video_clusters",
            "test_reference_classes",
            "test_reference_distribution",
            "headline_discrimination_eligible",
        ],
    )

    aggregate_rows = []
    for set_name, tasks in TASK_SETS.items():
        for condition in CONDITIONS:
            metrics = aggregate_metrics(grouped, tasks, condition, task_labels)
            aggregate_rows.append(
                {
                    "task_set": set_name,
                    "tasks": "|".join(tasks),
                    "task_count": len(tasks),
                    "condition": condition,
                    **metrics,
                }
            )
    write_csv(
        output / "mask_sensitivity_aggregate.csv",
        aggregate_rows,
        ["task_set", "tasks", "task_count", "condition", "accuracy", "macro_f1", "balanced_accuracy"],
    )

    delta_rows = []
    for set_index, set_name in enumerate(TASK_SETS):
        tasks = TASK_SETS[set_name]
        for comparison_index, (left, right) in enumerate(COMPARISONS):
            point, low, high = paired_task_video_bootstrap(
                grouped,
                tasks,
                left,
                right,
                BOOTSTRAP_SEED + set_index * 100 + comparison_index,
                task_labels,
            )
            delta_rows.append(
                {
                    "task_set": set_name,
                    "comparison": f"{left}_minus_{right}",
                    "delta_macro_f1": point,
                    "ci_low": low,
                    "ci_high": high,
                    "bootstrap_unit": (
                        "shared_video_key_across_tasks"
                        if set_name == "candidate_visual_five"
                        else "video_key_within_task"
                    ),
                    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                }
            )
    write_csv(
        output / "paired_video_bootstrap_deltas.csv",
        delta_rows,
        [
            "task_set",
            "comparison",
            "delta_macro_f1",
            "ci_low",
            "ci_high",
            "bootstrap_unit",
            "bootstrap_replicates",
        ],
    )

    loo_rows = []
    for left, right in [
        ("rgb_linear", "majority_prior"),
        ("rgb_mask_linear", "rgb_linear"),
    ]:
        for row in leave_one_video_out(
            grouped, TASK_SETS["candidate_visual_five"], left, right, task_labels
        ):
            loo_rows.append(
                {
                    "task_set": "candidate_visual_five",
                    "comparison": f"{left}_minus_{right}",
                    **row,
                }
            )
    write_csv(
        output / "leave_one_video_out_deltas.csv",
        loo_rows,
        [
            "task_set",
            "comparison",
            "omitted_video_key",
            "remaining_video_clusters",
            "delta_macro_f1",
        ],
    )

    prior_rows = []
    for task in available_tasks:
        task_rows = grouped[(task, "majority_prior")]
        references = np.asarray([row["reference"] for row in task_rows])
        predictions = np.asarray([row["prediction"] for row in task_rows])
        labels = task_labels[task]
        metrics = metric_bundle(references, predictions, labels)
        prior_rows.append(
            {
                "task": task,
                "canonical_question_count": 1,
                "question_only_predictor": "training_majority_label_for_fixed_question",
                "predicted_label": predictions[0],
                "test_n": len(task_rows),
                "test_reference_classes": len(set(references)),
                **metrics,
            }
        )
    write_csv(
        output / "question_prior_audit.csv",
        prior_rows,
        [
            "task",
            "canonical_question_count",
            "question_only_predictor",
            "predicted_label",
            "test_n",
            "test_reference_classes",
            "accuracy",
            "macro_f1",
            "balanced_accuracy",
        ],
    )

    summary = {
        "source_predictions": source.relative_to(project).as_posix(),
        "source_predictions_sha256": sha256(source),
        "prediction_rows": len(rows),
        "task_sets": TASK_SETS,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "fixed_label_source": config_path.relative_to(project).as_posix(),
        "bootstrap_design": "paired conditions; one shared six-video resample applied to every task; equal task weights; percentile interval",
        "outputs": [
            "task_partition.csv",
            "mask_sensitivity_aggregate.csv",
            "paired_video_bootstrap_deltas.csv",
            "leave_one_video_out_deltas.csv",
            "question_prior_audit.csv",
        ],
    }
    (output / "run.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
