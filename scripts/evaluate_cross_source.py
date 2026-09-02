#!/usr/bin/env python3
"""Run fixed EndoVis17-to-18 and EndoVis18-to-17 transfer diagnostics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import warnings
from collections import Counter
from pathlib import Path
from typing import Any


TASKS = [
    "q4.bleeding",
    "q4.smoke",
    "q4.occlusion",
    "q5",
    "q6",
    "q7",
    "q8.mask_center_5pct_v1",
    "q9",
]
NON_DIRECT_TASKS = ["q4.bleeding", "q4.occlusion", "q5", "q6", "q7"]
CONDITIONS = ["majority_prior", "rgb_linear", "mask_linear", "rgb_mask_linear"]
DIRECTIONS = [("site1", "site2"), ("site2", "site1")]
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20260825


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().lower()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def target(row: dict[str, Any], task: str, q8: dict[str, str]) -> str:
    if task == "q8.mask_center_5pct_v1":
        return q8[row["record_id"]]
    if task.startswith("q4."):
        return str(row["answers"]["q4"]["value"][task.split(".", 1)[1]])
    return str(row["answers"][task]["value"])


def metric_bundle(y_true, y_pred, labels: list[str]) -> dict[str, float]:
    import numpy as np

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


def cluster_ci(y_true, y_pred, videos, labels: list[str], seed: int) -> dict[str, tuple[float, float]]:
    import numpy as np

    unique_videos = sorted(set(videos))
    indices = {video: np.flatnonzero(videos == video) for video in unique_videos}
    rng = np.random.default_rng(seed)
    values = {"accuracy": [], "macro_f1": [], "balanced_accuracy": []}
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled = rng.choice(unique_videos, size=len(unique_videos), replace=True)
        sampled_indices = np.concatenate([indices[video] for video in sampled])
        bundle = metric_bundle(y_true[sampled_indices], y_pred[sampled_indices], labels)
        for key, value in bundle.items():
            values[key].append(value)
    return {
        key: (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))
        for key, vals in values.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    args = parser.parse_args()
    project = args.project.resolve()
    import numpy as np
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    features_path = project / "features" / "taskwise_resnet18_features.npz"
    annotations_path = project / "data" / "frame_annotations.jsonl"
    manifest_path = project / "data" / "manifest.jsonl"
    q8_path = project / "data" / "mask_targets.csv"
    output = project / "results" / "cross_source"
    output.mkdir(parents=True, exist_ok=True)

    feature_data = np.load(features_path)
    record_ids = feature_data["record_ids"].astype(str)
    rgb_features = feature_data["rgb_features"].astype(np.float32)
    mask_features = feature_data["mask_features"].astype(np.float32)
    annotations = {row["record_id"]: row for row in load_jsonl(annotations_path)}
    manifest = {row["record_id"]: row for row in load_jsonl(manifest_path)}
    q8 = {row["record_id"]: row["label"] for row in read_csv(q8_path)}
    if set(record_ids) != set(annotations) or set(record_ids) != set(manifest) or set(record_ids) != set(q8):
        raise RuntimeError("Frozen feature, annotation, manifest, and Q8 IDs must match")

    sites = np.asarray([manifest[record_id]["site"] for record_id in record_ids])
    splits = np.asarray([manifest[record_id]["split"] for record_id in record_ids])
    videos = np.asarray([manifest[record_id]["video_key"] for record_id in record_ids])
    feature_sets = {
        "rgb_linear": rgb_features,
        "mask_linear": mask_features,
        "rgb_mask_linear": np.concatenate([rgb_features, mask_features], axis=1),
    }

    metric_rows: list[dict[str, object]] = []
    class_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    warning_rows: list[dict[str, object]] = []
    aggregate_inputs: dict[tuple[str, str], list[float]] = {}

    for direction_index, (source_site, target_site) in enumerate(DIRECTIONS):
        direction = f"{source_site}_train_to_{target_site}_test"
        source_indices = np.flatnonzero((sites == source_site) & (splits == "train"))
        target_indices = np.flatnonzero((sites == target_site) & (splits == "test"))
        if not len(source_indices) or not len(target_indices):
            raise RuntimeError(f"Empty direction: {direction}")

        for task_index, task in enumerate(TASKS):
            y_all = np.asarray([target(annotations[record_id], task, q8) for record_id in record_ids])
            y_source = y_all[source_indices]
            y_target = y_all[target_indices]
            labels = sorted(set(y_source) | set(y_target))
            source_counts = Counter(y_source)
            target_counts = Counter(y_target)
            majority = sorted(source_counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
            predictions = {"majority_prior": np.repeat(majority, len(target_indices))}

            for condition, matrix in feature_sets.items():
                if len(set(y_source)) < 2:
                    predictions[condition] = np.repeat(majority, len(target_indices))
                    warning_rows.append(
                        {
                            "direction": direction,
                            "task": task,
                            "condition": condition,
                            "warning": "Single source-training class; reused source majority label.",
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
                    model.fit(matrix[source_indices], y_source)
                for warning in captured:
                    warning_rows.append(
                        {
                            "direction": direction,
                            "task": task,
                            "condition": condition,
                            "warning": str(warning.message),
                        }
                    )
                predictions[condition] = model.predict(matrix[target_indices]).astype(str)

            test_evaluable = len(set(y_target)) >= 2 and len(set(y_source)) >= 2
            for condition_index, condition in enumerate(CONDITIONS):
                y_pred = predictions[condition]
                metrics = metric_bundle(y_target, y_pred, labels)
                cis = cluster_ci(
                    y_target,
                    y_pred,
                    videos[target_indices],
                    labels,
                    BOOTSTRAP_SEED + direction_index * 1000 + task_index * 10 + condition_index,
                )
                metric_rows.append(
                    {
                        "direction": direction,
                        "source_site": source_site,
                        "target_site": target_site,
                        "task": task,
                        "condition": condition,
                        "source_train_n": len(source_indices),
                        "source_train_videos": len(set(videos[source_indices])),
                        "target_test_n": len(target_indices),
                        "target_test_videos": len(set(videos[target_indices])),
                        "source_classes": len(set(y_source)),
                        "target_classes": len(set(y_target)),
                        "test_evaluable": int(test_evaluable),
                        "source_distribution": json.dumps(dict(sorted(source_counts.items())), sort_keys=True),
                        "target_distribution": json.dumps(dict(sorted(target_counts.items())), sort_keys=True),
                        **metrics,
                        "accuracy_ci_low": cis["accuracy"][0],
                        "accuracy_ci_high": cis["accuracy"][1],
                        "macro_f1_ci_low": cis["macro_f1"][0],
                        "macro_f1_ci_high": cis["macro_f1"][1],
                        "balanced_accuracy_ci_low": cis["balanced_accuracy"][0],
                        "balanced_accuracy_ci_high": cis["balanced_accuracy"][1],
                    }
                )
                if task in NON_DIRECT_TASKS and test_evaluable:
                    aggregate_inputs.setdefault((direction, condition), []).append(metrics["macro_f1"])

                for label in labels:
                    mask = y_target == label
                    tp = int(np.sum(mask & (y_pred == label)))
                    fp = int(np.sum((~mask) & (y_pred == label)))
                    fn = int(np.sum(mask & (y_pred != label)))
                    precision = tp / (tp + fp) if tp + fp else 0.0
                    recall = tp / (tp + fn) if tp + fn else 0.0
                    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
                    class_rows.append(
                        {
                            "direction": direction,
                            "task": task,
                            "condition": condition,
                            "label": label,
                            "support": int(np.sum(mask)),
                            "precision": precision,
                            "recall": recall,
                            "f1": f1,
                        }
                    )
                for index, prediction in zip(target_indices, y_pred):
                    prediction_rows.append(
                        {
                            "direction": direction,
                            "record_id": record_ids[index],
                            "video_key": videos[index],
                            "target_site": target_site,
                            "task": task,
                            "condition": condition,
                            "reference": y_all[index],
                            "prediction": prediction,
                        }
                    )

    aggregate_rows = []
    for direction, _ in [(f"{s}_train_to_{t}_test", None) for s, t in DIRECTIONS]:
        for condition in CONDITIONS:
            values = aggregate_inputs.get((direction, condition), [])
            aggregate_rows.append(
                {
                    "direction": direction,
                    "condition": condition,
                    "eligible_non_direct_tasks": len(values),
                    "tasks": "|".join(
                        row["task"]
                        for row in metric_rows
                        if row["direction"] == direction
                        and row["condition"] == condition
                        and row["task"] in NON_DIRECT_TASKS
                        and int(row["test_evaluable"]) == 1
                    ),
                    "mean_task_macro_f1": float(np.mean(values)) if values else "",
                }
            )

    metric_fields = [
        "direction", "source_site", "target_site", "task", "condition",
        "source_train_n", "source_train_videos", "target_test_n", "target_test_videos",
        "source_classes", "target_classes", "test_evaluable", "source_distribution", "target_distribution",
        "accuracy", "macro_f1", "balanced_accuracy", "accuracy_ci_low", "accuracy_ci_high",
        "macro_f1_ci_low", "macro_f1_ci_high", "balanced_accuracy_ci_low", "balanced_accuracy_ci_high",
    ]
    write_csv(output / "metrics.csv", metric_rows, metric_fields)
    write_csv(
        output / "per_class_metrics.csv",
        class_rows,
        ["direction", "task", "condition", "label", "support", "precision", "recall", "f1"],
    )
    write_csv(
        output / "predictions.csv",
        prediction_rows,
        ["direction", "record_id", "video_key", "target_site", "task", "condition", "reference", "prediction"],
    )
    write_csv(
        output / "aggregate_metrics.csv",
        aggregate_rows,
        ["direction", "condition", "eligible_non_direct_tasks", "tasks", "mean_task_macro_f1"],
    )
    (output / "warnings.json").write_text(
        json.dumps(warning_rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    run = {
        "design": "source-train to target-test",
        "directions": [f"{source}_train_to_{target}_test" for source, target in DIRECTIONS],
        "aggregate_tasks": NON_DIRECT_TASKS,
        "feature_sha256": sha256(features_path),
        "annotations_sha256": sha256(annotations_path),
        "manifest_sha256": sha256(manifest_path),
        "q8_sha256": sha256(q8_path),
        "prediction_rows": len(prediction_rows),
        "metric_rows": len(metric_rows),
        "warning_count": len(warning_rows),
        "aggregate": aggregate_rows,
    }
    (output / "run.json").write_text(
        json.dumps(run, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(run, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
