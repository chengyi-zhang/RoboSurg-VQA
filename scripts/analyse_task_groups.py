#!/usr/bin/env python3
"""Compute shared-VQA metrics for the reported task groups."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREDICTIONS = (
    PROJECT_ROOT
    / "results"
    / "shared_vqa"
    / "predictions.csv"
)
DEFAULT_SUMMARY = (
    PROJECT_ROOT
    / "data"
    / "vqa_summary.json"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "task_groups"

TASK_GROUPS = {
    "primary_seven": [
        "q2",
        "q4.bleeding",
        "q4.occlusion",
        "q5",
        "q6",
        "q7",
        "q8.mask_center_5pct_v1",
    ],
    "perceptual_five": [
        "q4.bleeding",
        "q4.occlusion",
        "q5",
        "q6",
        "q7",
    ],
    "audit_supported_five": [
        "q2",
        "q4.bleeding",
        "q5",
        "q6",
        "q8.mask_center_5pct_v1",
    ],
    "audited_perceptual_three": ["q4.bleeding", "q5", "q6"],
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
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--replicates", type=int, default=5000)
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


def f1_from_counts(counts: dict[tuple[str, str], int], labels: list[str]) -> float:
    scores = []
    for label in labels:
        tp = counts.get((label, label), 0)
        fp = sum(
            value
            for (reference, prediction), value in counts.items()
            if reference != label and prediction == label
        )
        fn = sum(
            value
            for (reference, prediction), value in counts.items()
            if reference == label and prediction != label
        )
        denominator = 2 * tp + fp + fn
        scores.append(0.0 if denominator == 0 else 2 * tp / denominator)
    return float(np.mean(scores))


def main() -> None:
    args = parse_args()
    predictions = read_csv(args.predictions.resolve())
    with args.summary.resolve().open("r", encoding="utf-8") as handle:
        data_summary = json.load(handle)
    task_labels: dict[str, list[str]] = data_summary["task_labels"]

    condition_rows: dict[tuple[str, str], list[dict[str, str]]] = {}
    for key in CONDITIONS:
        rows = [
            row
            for row in predictions
            if row["condition"] == key[0] and row["wording"] == key[1]
        ]
        if not rows:
            raise RuntimeError(f"No predictions for {key}")
        condition_rows[key] = rows

    sequence_site = {}
    for row in condition_rows[("image_question", "canonical")]:
        sequence = row["source_sequence_id"]
        existing = sequence_site.setdefault(sequence, row["site"])
        if existing != row["site"]:
            raise RuntimeError(f"Source sequence assigned to multiple sites: {sequence}")
    sequences_by_site: dict[str, list[str]] = defaultdict(list)
    for sequence, site in sequence_site.items():
        sequences_by_site[site].append(sequence)
    if sum(len(values) for values in sequences_by_site.values()) != 6:
        raise RuntimeError("Expected six held-out source sequences")

    counts: dict[
        tuple[str, str], dict[str, dict[str, dict[tuple[str, str], int]]]
    ] = {}
    for key, rows in condition_rows.items():
        by_sequence: dict[str, dict[str, dict[tuple[str, str], int]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(int))
        )
        for row in rows:
            by_sequence[row["source_sequence_id"]][row["task"]][
                (row["reference"], row["prediction"])
            ] += 1
        counts[key] = by_sequence

    def score(key: tuple[str, str], group: str, selected: list[str]) -> float:
        task_values = []
        for task in TASK_GROUPS[group]:
            merged: dict[tuple[str, str], int] = defaultdict(int)
            for sequence in selected:
                for pair, value in counts[key][sequence][task].items():
                    merged[pair] += value
            task_values.append(f1_from_counts(merged, task_labels[task]))
        return float(np.mean(task_values))

    all_sequences = sorted(sequence_site)
    point = {
        (key, group): score(key, group, all_sequences)
        for key in CONDITIONS
        for group in TASK_GROUPS
    }

    rng = np.random.default_rng(args.seed)
    replicates = {
        (key, group): np.empty(args.replicates, dtype=np.float64)
        for key in CONDITIONS
        for group in TASK_GROUPS
    }
    for replicate in range(args.replicates):
        selected = []
        for site in sorted(sequences_by_site):
            candidates = sorted(sequences_by_site[site])
            selected.extend(
                rng.choice(candidates, size=len(candidates), replace=True).tolist()
            )
        for key in CONDITIONS:
            for group in TASK_GROUPS:
                replicates[(key, group)][replicate] = score(key, group, selected)

    aggregate_rows = []
    for key in CONDITIONS:
        for group, tasks in TASK_GROUPS.items():
            low, high = np.percentile(replicates[(key, group)], [2.5, 97.5])
            aggregate_rows.append(
                {
                    "condition": key[0],
                    "wording": key[1],
                    "task_group": group,
                    "task_count": len(tasks),
                    "mean_macro_f1_fixed_labels": point[(key, group)],
                    "ci_low": float(low),
                    "ci_high": float(high),
                }
            )

    delta_rows = []
    for group, tasks in TASK_GROUPS.items():
        for key_a, key_b in COMPARISONS:
            values = replicates[(key_a, group)] - replicates[(key_b, group)]
            low, high = np.percentile(values, [2.5, 97.5])
            delta_rows.append(
                {
                    "task_group": group,
                    "task_count": len(tasks),
                    "condition_a": key_a[0],
                    "wording_a": key_a[1],
                    "condition_b": key_b[0],
                    "wording_b": key_b[1],
                    "estimate_a": point[(key_a, group)],
                    "estimate_b": point[(key_b, group)],
                    "delta_macro_f1": point[(key_a, group)] - point[(key_b, group)],
                    "ci_low": float(low),
                    "ci_high": float(high),
                }
            )

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "metrics.csv", aggregate_rows)
    write_csv(output / "bootstrap_deltas.csv", delta_rows)
    run = {
        "analysis": "shared_vqa_task_groups",
        "task_groups": TASK_GROUPS,
        "bootstrap": {
            "replicates": args.replicates,
            "seed": args.seed,
            "unit": "source_sequence",
            "design": "site-stratified paired sequence bootstrap",
        },
    }
    (output / "run.json").write_text(
        json.dumps(run, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(run, indent=2))


if __name__ == "__main__":
    main()
