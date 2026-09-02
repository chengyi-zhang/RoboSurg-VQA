#!/usr/bin/env python3
"""Train the fixed taskwise baselines and compute held-out metrics."""

from __future__ import annotations

import csv
import hashlib
import json
import time
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


WORKSPACE = Path(__file__).resolve().parents[1]
import numpy as np
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    recall_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


FEATURES = (
    WORKSPACE
    / "features"
    / "taskwise_resnet18_features.npz"
)
ANNOTATIONS = WORKSPACE / "data" / "frame_annotations.jsonl"
APPLICABILITY = WORKSPACE / "data" / "task_applicability.csv"
MANIFEST = WORKSPACE / "data" / "manifest.jsonl"
Q8_TARGETS = WORKSPACE / "data" / "mask_targets.csv"
RESULTS = WORKSPACE / "results" / "taskwise"
Q8_TASK = "q8.mask_center_5pct_v1"
Q8_LABELS: dict[str, str] = {}

TASKS = [
    "q2",
    "q4.bleeding",
    "q4.smoke",
    "q4.occlusion",
    "q5",
    "q6",
    "q7",
    Q8_TASK,
    "q9",
]
CONDITIONS = ["majority_prior", "rgb_linear", "mask_linear", "rgb_mask_linear"]
BOOTSTRAP_REPLICATES = 1000
BOOTSTRAP_SEED = 20260824


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def target(row: dict[str, Any], task: str) -> str:
    if task == Q8_TASK:
        return Q8_LABELS[row["record_id"]]
    if task.startswith("q4."):
        field = task.split(".", 1)[1]
        return str(row["answers"]["q4"]["value"][field])
    return str(row["answers"][task]["value"])


def metric_bundle(
    y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]
) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1_fixed_labels": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "balanced_accuracy_fixed_labels": float(
            recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
    }


def clustered_bootstrap(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    video_keys: np.ndarray,
    labels: list[str],
    seed: int,
) -> dict[str, tuple[float, float]]:
    unique_videos = np.unique(video_keys)
    video_indices = {video: np.flatnonzero(video_keys == video) for video in unique_videos}
    rng = np.random.default_rng(seed)
    values = {
        "accuracy": [],
        "macro_f1_fixed_labels": [],
        "balanced_accuracy_fixed_labels": [],
    }
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled = rng.choice(unique_videos, size=len(unique_videos), replace=True)
        indices = np.concatenate([video_indices[video] for video in sampled])
        metrics = metric_bundle(y_true[indices], y_pred[indices], labels)
        for name, value in metrics.items():
            values[name].append(value)
    return {
        name: (float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5)))
        for name, samples in values.items()
    }


def main() -> None:
    global Q8_LABELS
    RESULTS.mkdir(parents=True, exist_ok=True)
    started = time.time()
    for required in (FEATURES, ANNOTATIONS, APPLICABILITY, MANIFEST, Q8_TARGETS):
        if not required.is_file():
            raise FileNotFoundError(required)

    feature_data = np.load(FEATURES)
    record_ids = feature_data["record_ids"].astype(str)
    rgb_features = feature_data["rgb_features"].astype(np.float32)
    mask_features = feature_data["mask_features"].astype(np.float32)
    annotation_rows = load_jsonl(ANNOTATIONS)
    annotations = {row["record_id"]: row for row in annotation_rows}
    manifest = {row["record_id"]: row for row in load_jsonl(MANIFEST)}
    with APPLICABILITY.open("r", encoding="utf-8", newline="") as handle:
        primary_tasks = {
            (row["site"], row["target"])
            for row in csv.DictReader(handle)
            if row["primary_scored"] == "1"
        }
    with Q8_TARGETS.open("r", encoding="utf-8", newline="") as handle:
        q8_rows = list(csv.DictReader(handle))
    Q8_LABELS = {row["record_id"]: row["label"] for row in q8_rows}
    if len(annotations) != len(annotation_rows):
        raise RuntimeError("Duplicate annotation record IDs")
    if set(record_ids) != set(annotations) or set(record_ids) != set(manifest):
        raise RuntimeError("Feature, annotation, and manifest record IDs do not match")
    if len(Q8_LABELS) != len(q8_rows) or set(record_ids) != set(Q8_LABELS):
        raise RuntimeError("Deterministic Q8 record IDs do not match the frozen corpus")
    if set(Q8_LABELS.values()) != {"yes", "no"}:
        raise RuntimeError("Deterministic Q8 must contain exactly yes/no labels")
    if "q1" in TASKS or "q3" in TASKS:
        raise RuntimeError("Source metadata fields Q1/Q3 must not be scored")

    splits = np.asarray([manifest[record_id]["split"] for record_id in record_ids])
    video_keys = np.asarray([manifest[record_id]["video_key"] for record_id in record_ids])
    sites = np.asarray([annotations[record_id]["site"] for record_id in record_ids])
    datasets = np.asarray([annotations[record_id]["dataset"] for record_id in record_ids])
    feature_sets = {
        "rgb_linear": rgb_features,
        "mask_linear": mask_features,
        "rgb_mask_linear": np.concatenate([rgb_features, mask_features], axis=1),
    }

    metric_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    warning_rows: list[dict[str, Any]] = []
    predictions: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    task_summary: dict[str, dict[str, Any]] = {}

    for task_index, task in enumerate(TASKS):
        eligible = np.asarray(
            [
                (annotations[record_id]["site"], task) in primary_tasks
                for record_id in record_ids
            ]
        )
        eligible_indices = np.flatnonzero(eligible)
        train_indices = np.flatnonzero(eligible & (splits == "train"))
        test_indices = np.flatnonzero(eligible & (splits == "test"))
        if not len(train_indices) or not len(test_indices):
            raise RuntimeError(f"Task has no eligible train/test records: {task}")
        if task == "q2" and set(sites[eligible_indices]) != {"site2"}:
            raise RuntimeError("Q2 must be restricted to source-grounded EndoVis 2018 records")

        all_targets = np.asarray(
            [target(annotations[record_id], task) if flag else "" for record_id, flag in zip(record_ids, eligible)]
        )
        y_train = all_targets[train_indices]
        y_test = all_targets[test_indices]
        labels = sorted(set(y_train) | set(y_test))
        train_counts = Counter(y_train)
        test_counts = Counter(y_test)
        majority_label = sorted(train_counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
        condition_predictions: dict[str, np.ndarray] = {
            "majority_prior": np.repeat(majority_label, len(test_indices))
        }

        for condition, features in feature_sets.items():
            if len(set(y_train)) < 2:
                condition_predictions[condition] = np.repeat(majority_label, len(test_indices))
                warning_rows.append(
                    {
                        "task": task,
                        "condition": condition,
                        "warning": "Single training class; used the fixed training label.",
                    }
                )
                continue
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    solver="lbfgs",
                    max_iter=5000,
                    tol=1e-4,
                ),
            )
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always", ConvergenceWarning)
                model.fit(features[train_indices], y_train)
            for warning in captured:
                warning_rows.append(
                    {"task": task, "condition": condition, "warning": str(warning.message)}
                )
            condition_predictions[condition] = model.predict(features[test_indices]).astype(str)

        task_summary[task] = {
            "eligible_n": int(len(eligible_indices)),
            "train_n": int(len(train_indices)),
            "test_n": int(len(test_indices)),
            "train_video_count": int(len(np.unique(video_keys[train_indices]))),
            "test_video_count": int(len(np.unique(video_keys[test_indices]))),
            "eligible_sites": sorted(set(sites[eligible_indices])),
            "labels": labels,
            "test_evaluable": len(set(y_test)) >= 2,
        }

        for condition in CONDITIONS:
            y_pred = condition_predictions[condition]
            predictions[(condition, task)] = (test_indices, y_pred)
            metrics = metric_bundle(y_test, y_pred, labels)
            metric_rows.append(
                {
                    "task": task,
                    "condition": condition,
                    "eligible_sites": " | ".join(sorted(set(sites[eligible_indices]))),
                    "train_n": len(y_train),
                    "test_n": len(y_test),
                    "train_classes": len(set(y_train)),
                    "test_classes": len(set(y_test)),
                    "fixed_label_count": len(labels),
                    "degenerate_test_missing_class": int(len(set(y_test)) < len(labels)),
                    **metrics,
                }
            )
            precision, recall, f1, support = precision_recall_fscore_support(
                y_test, y_pred, labels=labels, zero_division=0
            )
            matrix = confusion_matrix(y_test, y_pred, labels=labels)
            for label_index, label in enumerate(labels):
                class_rows.append(
                    {
                        "task": task,
                        "condition": condition,
                        "label": label,
                        "train_support": train_counts[label],
                        "test_support": int(support[label_index]),
                        "precision": float(precision[label_index]),
                        "recall": float(recall[label_index]),
                        "f1": float(f1[label_index]),
                        "predicted_support": int(matrix[:, label_index].sum()),
                    }
                )
            intervals = clustered_bootstrap(
                y_test,
                y_pred,
                video_keys[test_indices],
                labels,
                BOOTSTRAP_SEED + task_index,
            )
            for metric, (lower, upper) in intervals.items():
                bootstrap_rows.append(
                    {
                        "task": task,
                        "condition": condition,
                        "metric": metric,
                        "estimate": metrics[metric],
                        "ci_2.5": lower,
                        "ci_97.5": upper,
                        "bootstrap_unit": "video_key",
                        "test_video_count": len(np.unique(video_keys[test_indices])),
                        "replicates": BOOTSTRAP_REPLICATES,
                        "seed": BOOTSTRAP_SEED + task_index,
                    }
                )
            for local_index, global_index in enumerate(test_indices):
                prediction_rows.append(
                    {
                        "record_id": record_ids[global_index],
                        "video_key": video_keys[global_index],
                        "site": sites[global_index],
                        "dataset": datasets[global_index],
                        "task": task,
                        "condition": condition,
                        "reference": y_test[local_index],
                        "prediction": y_pred[local_index],
                    }
                )

    q4_tuple_rows: list[dict[str, Any]] = []
    q4_indices = predictions[("majority_prior", "q4.bleeding")][0]
    q4_truth = np.column_stack(
        [
            np.asarray([target(annotations[record_ids[index]], f"q4.{field}") for index in q4_indices])
            for field in ("bleeding", "smoke", "occlusion")
        ]
    )
    for condition in CONDITIONS:
        q4_parts = [predictions[(condition, f"q4.{field}")] for field in ("bleeding", "smoke", "occlusion")]
        if any(not np.array_equal(indices, q4_indices) for indices, _ in q4_parts):
            raise RuntimeError("Q4 subtask eligibility differs unexpectedly")
        q4_prediction = np.column_stack([prediction for _, prediction in q4_parts])
        q4_tuple_rows.append(
            {
                "condition": condition,
                "test_n": len(q4_indices),
                "tuple_exact_match": float(np.all(q4_truth == q4_prediction, axis=1).mean()),
            }
        )

    def write_csv(name: str, rows: list[dict[str, Any]]) -> Path:
        if not rows:
            raise RuntimeError(f"Refusing to write empty CSV: {name}")
        path = RESULTS / name
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return path

    output_paths = [
        write_csv("metrics.csv", metric_rows),
        write_csv("per_class_metrics.csv", class_rows),
        write_csv("bootstrap_cis.csv", bootstrap_rows),
        write_csv("predictions.csv", prediction_rows),
        write_csv("q4_tuple_metrics.csv", q4_tuple_rows),
    ]

    evaluable_tasks = [
        task
        for task in TASKS
        if task_summary[task]["test_evaluable"] and task != Q8_TASK
    ]
    aggregate: dict[str, dict[str, Any]] = {}
    for condition in CONDITIONS:
        rows = [row for row in metric_rows if row["condition"] == condition]
        aggregate[condition] = {
            "mean_macro_f1_all_scored_tasks": float(
                np.mean([row["macro_f1_fixed_labels"] for row in rows])
            ),
            "mean_macro_f1_test_evaluable_tasks": float(
                np.mean(
                    [
                        row["macro_f1_fixed_labels"]
                        for row in rows
                        if row["task"] in evaluable_tasks
                    ]
                )
            ),
            "test_evaluable_tasks": evaluable_tasks,
        }

    config = {
        "protocol": "taskwise_fixed_baselines",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "split": "frozen video-level train/test split; task-specific applicability",
        "task_summary": task_summary,
        "conditions": {
            "majority_prior": "Most frequent eligible training label per fixed question.",
            "rgb_linear": "Frozen ResNet-18 ImageNet features plus a class-balanced logistic head.",
            "mask_linear": "Deterministic official instrument-mask geometry plus the same head.",
            "rgb_mask_linear": "Concatenated frozen RGB and mask features plus the same head.",
        },
        "head": {
            "model": "LogisticRegression",
            "C": 1.0,
            "class_weight": "balanced",
            "solver": "lbfgs",
            "max_iter": 5000,
            "tol": 0.0001,
            "hyperparameter_search": False,
            "stochastic_training_seeds": "not applicable; fixed deterministic solver",
        },
        "bootstrap": {
            "unit": "video_key",
            "replicates": BOOTSTRAP_REPLICATES,
            "base_seed": BOOTSTRAP_SEED,
        },
        "headline_aggregate_policy": {
            "included_tasks": evaluable_tasks,
            "excluded_deterministic_diagnostics": {
                Q8_TASK: (
                    "The target is thresholded from central instrument-mask occupancy and "
                    "would be circular in an aggregate that compares mask-derived features."
                )
            },
        },
        "feature_file": FEATURES.relative_to(WORKSPACE).as_posix(),
        "feature_file_sha256": sha256(FEATURES),
        "annotations": ANNOTATIONS.relative_to(WORKSPACE).as_posix(),
        "annotations_sha256": sha256(ANNOTATIONS),
        "q8_deterministic_targets": Q8_TARGETS.relative_to(WORKSPACE).as_posix(),
        "q8_deterministic_targets_sha256": sha256(Q8_TARGETS),
        "applicability": APPLICABILITY.relative_to(WORKSPACE).as_posix(),
        "applicability_sha256": sha256(APPLICABILITY),
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "elapsed_seconds": time.time() - started,
        "warnings": warning_rows,
        "aggregate": aggregate,
        "outputs": {path.name: sha256(path) for path in output_paths},
    }
    config_path = RESULTS / "run.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(config, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
