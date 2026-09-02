#!/usr/bin/env python3
"""Evaluate frozen taskwise predictions against the adjudicated human audit."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
PREDICTIONS = PROJECT / "results" / "taskwise" / "predictions.csv"
BASELINE_CONFIG = PREDICTIONS.parent / "run.json"
FINAL_HUMAN = PROJECT / "data" / "human_reference" / "reference.csv"
MACHINE_AUDIT = PROJECT / "data" / "human_reference" / "candidate_labels.csv"
OUT = PROJECT / "results" / "human_reference"

CONDITIONS = ["majority_prior", "rgb_linear", "mask_linear", "rgb_mask_linear"]
VISUAL_TASKS = ["q4.bleeding", "q4.occlusion", "q5", "q6", "q7"]
TASK_SETS = {
    "all_five_candidate_visual_tasks": VISUAL_TASKS,
    "without_q7": ["q4.bleeding", "q4.occlusion", "q5", "q6"],
    "without_q4_view_obstruction": ["q4.bleeding", "q5", "q6", "q7"],
    "without_q7_and_q4_view_obstruction": ["q4.bleeding", "q5", "q6"],
}
COMPARISONS = [
    ("rgb_linear", "majority_prior"),
    ("rgb_mask_linear", "rgb_linear"),
]
BOOTSTRAP_REPLICATES = 5000
BOOTSTRAP_SEED = 20260826

HUMAN_FIELDS = {
    "q4.bleeding": "q4_bleeding",
    "q4.smoke": "q4_smoke",
    "q4.occlusion": "q4_occlusion",
    "q5": "q5_value",
    "q6": "q6_glare",
    "q7": "q7_contrast_normal",
    "q9": "q9_smoke_region",
}
MACHINE_FIELDS = {
    "q4.bleeding": "q4_bleeding",
    "q4.smoke": "q4_smoke",
    "q4.occlusion": "q4_occlusion",
    "q5": "q5_value",
    "q6": "q6_value",
    "q7": "q7_value",
    "q9": "q9_value",
}
COMPARISON_LABELS = {
    "q4.bleeding": ["no", "yes"],
    "q4.smoke": ["no", "yes"],
    "q4.occlusion": ["no", "yes"],
    "q5": ["clear", "degraded"],
    "q6": ["no", "yes"],
    "q7": ["no", "yes"],
    "q9": ["none", "smoke present"],
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    fieldnames = fields or list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def metrics(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict[str, object]:
    if len(y_true) != len(y_pred) or not y_true:
        raise ValueError("Metric inputs must be non-empty and aligned")
    label_set = set(labels)
    outside_true = sorted(set(y_true) - label_set)
    outside_pred = sorted(set(y_pred) - label_set)
    if outside_true or outside_pred:
        raise ValueError(f"Labels outside fixed list: true={outside_true}, pred={outside_pred}")
    f1_values: list[float] = []
    recalls: list[float] = []
    per_class: list[dict[str, object]] = []
    for label in labels:
        tp = sum(a == label and b == label for a, b in zip(y_true, y_pred))
        fp = sum(a != label and b == label for a, b in zip(y_true, y_pred))
        fn = sum(a == label and b != label for a, b in zip(y_true, y_pred))
        support = sum(a == label for a in y_true)
        predicted = sum(b == label for b in y_pred)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        if support:
            recalls.append(recall)
        per_class.append(
            {
                "label": label,
                "support": support,
                "predicted": predicted,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return {
        "n": len(y_true),
        "accuracy": sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true),
        "macro_f1": sum(f1_values) / len(labels),
        "balanced_accuracy": (sum(recalls) / len(recalls)) if len(recalls) >= 2 else None,
        "observed_reference_classes": len(set(y_true)),
        "per_class": per_class,
    }


def map_label(task: str, value: str, source: str) -> str | None:
    value = value.strip().lower()
    if value == "uncertain":
        return None
    if task == "q5":
        if value == "clear":
            return "clear"
        if value in {"blurry", "mixed", "reflective"}:
            return "degraded"
    elif task == "q9":
        if value == "none":
            return "none"
        if source == "machine" and value == "near instrument jaws":
            return "smoke present"
        if source == "human" and value in {
            "left",
            "center",
            "right",
            "multiple",
            "near instrument jaws",
            "smoke present but unlocalizable",
        }:
            return "smoke present"
    elif value in {"yes", "no"}:
        return value
    raise ValueError(f"Unmapped {source} value for {task}: {value!r}")


def build_mapping_table() -> list[dict[str, object]]:
    rows = [
        ("q4.bleeding", "machine", "no | yes", "no | yes", "Exact binary mapping"),
        ("q4.bleeding", "human", "no | yes | uncertain", "no | yes", "Uncertain excluded"),
        ("q4.smoke", "machine", "no | yes", "no | yes", "Exact binary mapping"),
        ("q4.smoke", "human", "no | yes | uncertain", "no | yes", "Uncertain excluded"),
        ("q4.occlusion", "machine", "no | yes", "no | yes", "Exact binary mapping"),
        ("q4.occlusion", "human", "no | yes | uncertain", "no | yes", "Uncertain excluded"),
        ("q5", "machine", "clear | blurry | mixed", "clear | degraded", "Blurry and mixed collapse to degraded"),
        ("q5", "human", "clear | blurry | mixed | reflective | uncertain", "clear | degraded", "Blurry, mixed and reflective collapse to degraded; uncertain excluded"),
        ("q6", "machine", "no | yes", "no | yes", "Exact binary mapping"),
        ("q6", "human", "no | yes | uncertain", "no | yes", "Uncertain excluded"),
        ("q7", "machine", "no | yes", "no | yes", "Exact binary mapping; yes means normal contrast"),
        ("q7", "human", "no | yes | uncertain", "no | yes", "Uncertain excluded; yes means normal contrast"),
        ("q9", "machine", "none | near instrument jaws", "none | smoke present", "Near instrument jaws collapses to smoke present"),
        ("q9", "human", "none | left | center | right | multiple | near instrument jaws | smoke present but unlocalizable | uncertain", "none | smoke present", "All localised or unlocalisable smoke categories collapse to smoke present; uncertain excluded"),
    ]
    return [
        {
            "task": task,
            "source": source,
            "source_vocabulary": vocabulary,
            "comparison_vocabulary": comparison,
            "mapping_rule": rule,
            "fixed_metric_labels": " | ".join(COMPARISON_LABELS[task]),
        }
        for task, source, vocabulary, comparison, rule in rows
    ]


def macro_f1_from_confusion(
    counts: Counter[tuple[str, str]], labels: list[str]
) -> float:
    scores = []
    for label in labels:
        tp = counts[(label, label)]
        fp = sum(count for (truth, pred), count in counts.items() if truth != label and pred == label)
        fn = sum(count for (truth, pred), count in counts.items() if truth == label and pred != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
    return sum(scores) / len(labels)


def aggregate_task_scores_from_index(
    confusion_index: dict[str, dict[str, dict[str, Counter[tuple[str, str]]]]],
    fixed_labels: dict[str, list[str]],
    task_set: list[str],
    condition: str,
    sampled_keys: list[str],
) -> float:
    scores = []
    for task in task_set:
        combined: Counter[tuple[str, str]] = Counter()
        for key in sampled_keys:
            combined.update(confusion_index[task][condition][key])
        scores.append(macro_f1_from_confusion(combined, fixed_labels[task]))
    return sum(scores) / len(scores)


def build_candidate_sensitivity(prediction_rows: list[dict[str, str]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    config = json.loads(BASELINE_CONFIG.read_text(encoding="utf-8"))
    fixed_labels = {
        task: list(config["task_summary"][task]["labels"])
        for task in VISUAL_TASKS
    }
    keys = sorted({row["video_key"] for row in prediction_rows if row["task"] in VISUAL_TASKS})
    if len(keys) != 6:
        raise RuntimeError(f"Expected six held-out sequence keys, found {keys}")
    confusion_index: dict[str, dict[str, dict[str, Counter[tuple[str, str]]]]] = {
        task: {
            condition: {key: Counter() for key in keys}
            for condition in CONDITIONS
        }
        for task in VISUAL_TASKS
    }
    for row in prediction_rows:
        task = row["task"]
        condition = row["condition"]
        if task not in VISUAL_TASKS or condition not in CONDITIONS:
            continue
        confusion_index[task][condition][row["video_key"]][
            (row["reference"], row["prediction"])
        ] += 1
    summary_rows: list[dict[str, object]] = []
    delta_rows: list[dict[str, object]] = []
    for set_index, (set_name, tasks) in enumerate(TASK_SETS.items()):
        points = {
            condition: aggregate_task_scores_from_index(
                confusion_index, fixed_labels, tasks, condition, keys
            )
            for condition in CONDITIONS
        }
        for condition in CONDITIONS:
            summary_rows.append(
                {
                    "task_set": set_name,
                    "included_tasks": " | ".join(tasks),
                    "task_count": len(tasks),
                    "condition": condition,
                    "mean_fixed_label_macro_f1": points[condition],
                }
            )
        for comparison_index, (condition_a, condition_b) in enumerate(COMPARISONS):
            rng = random.Random(BOOTSTRAP_SEED + set_index * 100 + comparison_index)
            replicate_deltas = []
            for _ in range(BOOTSTRAP_REPLICATES):
                sampled = rng.choices(keys, k=len(keys))
                score_a = aggregate_task_scores_from_index(
                    confusion_index, fixed_labels, tasks, condition_a, sampled
                )
                score_b = aggregate_task_scores_from_index(
                    confusion_index, fixed_labels, tasks, condition_b, sampled
                )
                replicate_deltas.append(score_a - score_b)
            delta_rows.append(
                {
                    "task_set": set_name,
                    "included_tasks": " | ".join(tasks),
                    "comparison": f"{condition_a}_minus_{condition_b}",
                    "delta_macro_f1": points[condition_a] - points[condition_b],
                    "ci_low": percentile(replicate_deltas, 0.025),
                    "ci_high": percentile(replicate_deltas, 0.975),
                    "held_out_sequence_clusters": len(keys),
                    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                    "bootstrap_seed": BOOTSTRAP_SEED + set_index * 100 + comparison_index,
                    "bootstrap_design": "paired shared six-sequence resampling across tasks and conditions",
                }
            )
    return summary_rows, delta_rows


def audit_subset(row: dict[str, str]) -> str:
    return "representative" if row["representative_stratum"] == "1" else "challenge"


def cluster_bootstrap_candidate_human(
    records: list[dict[str, object]], labels: list[str], seed: int
) -> dict[str, float]:
    clusters: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in records:
        clusters[str(row["sequence_key"])].append(row)
    keys = sorted(clusters)
    rng = random.Random(seed)
    accuracies: list[float] = []
    macro_f1s: list[float] = []
    for _ in range(2000):
        sample = [row for key in rng.choices(keys, k=len(keys)) for row in clusters[key]]
        result = metrics(
            [str(row["human_comparison"]) for row in sample],
            [str(row["machine_comparison"]) for row in sample],
            labels,
        )
        accuracies.append(float(result["accuracy"]))
        macro_f1s.append(float(result["macro_f1"]))
    return {
        "accuracy_ci_low": percentile(accuracies, 0.025),
        "accuracy_ci_high": percentile(accuracies, 0.975),
        "macro_f1_ci_low": percentile(macro_f1s, 0.025),
        "macro_f1_ci_high": percentile(macro_f1s, 0.975),
        "sequence_clusters": len(keys),
    }


def build_harmonised_candidate_human(
    human_rows: list[dict[str, str]], machine_rows: list[dict[str, str]]
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    machine_by_id = {row["sample_id"]: row for row in machine_rows}
    frame_rows: list[dict[str, object]] = []
    for human in human_rows:
        if human["global_unusable_frame_yes_no"] == "yes":
            continue
        machine = machine_by_id[human["sample_id"]]
        for task in HUMAN_FIELDS:
            human_value = map_label(task, human[HUMAN_FIELDS[task]], "human")
            machine_value = map_label(task, machine[MACHINE_FIELDS[task]], "machine")
            if human_value is None or machine_value is None:
                continue
            frame_rows.append(
                {
                    "sample_id": human["sample_id"],
                    "record_id": human["record_id"],
                    "dataset": human["dataset"],
                    "split": human["split"],
                    "sequence_key": f"{human['site']}/{human['split']}/{human['video_key'].split('/', 1)[1]}",
                    "subset": audit_subset(human),
                    "task": task,
                    "human_original": human[HUMAN_FIELDS[task]],
                    "machine_original": machine[MACHINE_FIELDS[task]],
                    "human_comparison": human_value,
                    "machine_comparison": machine_value,
                }
            )
    metric_rows: list[dict[str, object]] = []
    confusion_rows: list[dict[str, object]] = []
    for subset in ["all", "representative", "challenge"]:
        subset_rows = frame_rows if subset == "all" else [row for row in frame_rows if row["subset"] == subset]
        for task_index, task in enumerate(HUMAN_FIELDS):
            rows = [row for row in subset_rows if row["task"] == task]
            result = metrics(
                [str(row["human_comparison"]) for row in rows],
                [str(row["machine_comparison"]) for row in rows],
                COMPARISON_LABELS[task],
            )
            interval = (
                cluster_bootstrap_candidate_human(
                    rows, COMPARISON_LABELS[task], BOOTSTRAP_SEED + 2000 + task_index
                )
                if subset == "representative"
                else None
            )
            metric_rows.append(
                {
                    "subset": subset,
                    "task": task,
                    "n": result["n"],
                    "fixed_labels": " | ".join(COMPARISON_LABELS[task]),
                    "observed_human_classes": result["observed_reference_classes"],
                    "accuracy": result["accuracy"],
                    "accuracy_ci_low": interval["accuracy_ci_low"] if interval else None,
                    "accuracy_ci_high": interval["accuracy_ci_high"] if interval else None,
                    "macro_f1": result["macro_f1"],
                    "macro_f1_ci_low": interval["macro_f1_ci_low"] if interval else None,
                    "macro_f1_ci_high": interval["macro_f1_ci_high"] if interval else None,
                    "balanced_accuracy": result["balanced_accuracy"],
                    "balanced_accuracy_status": "reported" if result["balanced_accuracy"] is not None else "not applicable: one observed human class",
                    "sequence_clusters": interval["sequence_clusters"] if interval else None,
                    "bootstrap_replicates": 2000 if interval else 0,
                }
            )
            for human_label in COMPARISON_LABELS[task]:
                for machine_label in COMPARISON_LABELS[task]:
                    confusion_rows.append(
                        {
                            "subset": subset,
                            "task": task,
                            "human_label": human_label,
                            "machine_label": machine_label,
                            "count": sum(
                                row["human_comparison"] == human_label
                                and row["machine_comparison"] == machine_label
                                for row in rows
                            ),
                        }
                    )
    return frame_rows, metric_rows, confusion_rows


def cluster_bootstrap_human(
    records: list[dict[str, str]], labels: list[str], seed: int
) -> dict[str, float]:
    clusters: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in records:
        clusters[row["sequence_key"]].append(row)
    keys = sorted(clusters)
    rng = random.Random(seed)
    accuracies: list[float] = []
    macro_f1s: list[float] = []
    for _ in range(2000):
        sample = [row for key in rng.choices(keys, k=len(keys)) for row in clusters[key]]
        result = metrics(
            [row["human"] for row in sample],
            [row["prediction"] for row in sample],
            labels,
        )
        accuracies.append(float(result["accuracy"]))
        macro_f1s.append(float(result["macro_f1"]))
    return {
        "accuracy_ci_low": percentile(accuracies, 0.025),
        "accuracy_ci_high": percentile(accuracies, 0.975),
        "macro_f1_ci_low": percentile(macro_f1s, 0.025),
        "macro_f1_ci_high": percentile(macro_f1s, 0.975),
    }


def build_harmonised_human_baselines(
    human_rows: list[dict[str, str]], prediction_rows: list[dict[str, str]]
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    prediction_map = {
        (row["record_id"], row["task"], row["condition"]): row
        for row in prediction_rows
    }
    test_rows = [
        row
        for row in human_rows
        if row["representative_stratum"] == "1"
        and row["split"] == "test"
        and row["global_unusable_frame_yes_no"] != "yes"
    ]
    tasks = ["q4.bleeding", "q5", "q6"]
    metric_rows: list[dict[str, object]] = []
    confusion_rows: list[dict[str, object]] = []
    for task_index, task in enumerate(tasks):
        labels = COMPARISON_LABELS[task]
        for condition_index, condition in enumerate(CONDITIONS):
            records: list[dict[str, str]] = []
            for human in test_rows:
                human_label = map_label(task, human[HUMAN_FIELDS[task]], "human")
                if human_label is None:
                    continue
                prediction = prediction_map[(human["record_id"], task, condition)]["prediction"]
                prediction_label = map_label(task, prediction, "machine")
                if prediction_label is None:
                    raise RuntimeError("Frozen prediction unexpectedly mapped to None")
                records.append(
                    {
                        "sample_id": human["sample_id"],
                        "record_id": human["record_id"],
                        "sequence_key": f"{human['site']}/{human['split']}/{human['video_key'].split('/', 1)[1]}",
                        "human": human_label,
                        "prediction": prediction_label,
                    }
                )
            result = metrics(
                [row["human"] for row in records],
                [row["prediction"] for row in records],
                labels,
            )
            interval = cluster_bootstrap_human(
                records, labels, BOOTSTRAP_SEED + 1000 + task_index * 10 + condition_index
            )
            metric_rows.append(
                {
                    "task": task,
                    "condition": condition,
                    "n": result["n"],
                    "sequence_clusters": len({row["sequence_key"] for row in records}),
                    "fixed_labels": " | ".join(labels),
                    "accuracy": result["accuracy"],
                    "accuracy_ci_low": interval["accuracy_ci_low"],
                    "accuracy_ci_high": interval["accuracy_ci_high"],
                    "macro_f1": result["macro_f1"],
                    "macro_f1_ci_low": interval["macro_f1_ci_low"],
                    "macro_f1_ci_high": interval["macro_f1_ci_high"],
                    "balanced_accuracy": result["balanced_accuracy"],
                    "observed_human_classes": result["observed_reference_classes"],
                }
            )
            for human_label in labels:
                for prediction_label in labels:
                    confusion_rows.append(
                        {
                            "task": task,
                            "condition": condition,
                            "human_label": human_label,
                            "prediction_label": prediction_label,
                            "count": sum(
                                row["human"] == human_label and row["prediction"] == prediction_label
                                for row in records
                            ),
                        }
                    )
    aggregate_rows: list[dict[str, object]] = []
    for condition in CONDITIONS:
        rows = [row for row in metric_rows if row["condition"] == condition]
        aggregate_rows.append(
            {
                "condition": condition,
                "included_tasks": " | ".join(tasks),
                "task_count": len(tasks),
                "mean_accuracy": sum(float(row["accuracy"]) for row in rows) / len(rows),
                "mean_macro_f1": sum(float(row["macro_f1"]) for row in rows) / len(rows),
                "mean_balanced_accuracy": sum(float(row["balanced_accuracy"]) for row in rows) / len(rows),
            }
        )
    return metric_rows, aggregate_rows, confusion_rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    prediction_rows = read_csv(PREDICTIONS)
    human_rows = read_csv(FINAL_HUMAN)
    machine_rows = read_csv(MACHINE_AUDIT)

    mapping_rows = build_mapping_table()
    write_csv(OUT / "label_mapping.csv", mapping_rows)

    sensitivity, sensitivity_deltas = build_candidate_sensitivity(prediction_rows)
    write_csv(OUT / "task_removal_metrics.csv", sensitivity)
    write_csv(OUT / "task_removal_deltas.csv", sensitivity_deltas)

    frame_rows, candidate_metrics, candidate_confusions = build_harmonised_candidate_human(
        human_rows, machine_rows
    )
    write_csv(OUT / "candidate_human_framewise.csv", frame_rows)
    write_csv(OUT / "candidate_human_metrics.csv", candidate_metrics)
    write_csv(OUT / "candidate_human_confusions.csv", candidate_confusions)

    human_metrics, human_aggregate, human_confusions = build_harmonised_human_baselines(
        human_rows, prediction_rows
    )
    write_csv(OUT / "taskwise_human_reference_metrics.csv", human_metrics)
    write_csv(OUT / "human_reference_aggregate.csv", human_aggregate)
    write_csv(OUT / "human_reference_confusions.csv", human_confusions)

    outputs = sorted(
        path
        for path in OUT.iterdir()
        if path.is_file() and path.name != "run.json"
    )
    run = {
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "inputs": {
            PREDICTIONS.relative_to(PROJECT).as_posix(): sha256(PREDICTIONS),
            BASELINE_CONFIG.relative_to(PROJECT).as_posix(): sha256(BASELINE_CONFIG),
            FINAL_HUMAN.relative_to(PROJECT).as_posix(): sha256(FINAL_HUMAN),
            MACHINE_AUDIT.relative_to(PROJECT).as_posix(): sha256(MACHINE_AUDIT),
        },
        "outputs": {path.name: sha256(path) for path in outputs},
    }
    (OUT / "run.json").write_text(
        json.dumps(run, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(run, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
