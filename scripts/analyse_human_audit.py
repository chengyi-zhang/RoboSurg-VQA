#!/usr/bin/env python3
"""Compute agreement between the two independent audit annotations."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "data" / "human_reference"
OUTPUT = ROOT / "results" / "human_audit" / "agreement.csv"
FIELDS = [
    "q4_bleeding",
    "q4_smoke",
    "q4_occlusion",
    "q5_value",
    "q6_glare",
    "q7_contrast_normal",
    "q9_smoke_region",
    "global_unusable_frame_yes_no",
]
BINARY_FIELDS = {
    "q4_bleeding",
    "q4_smoke",
    "q4_occlusion",
    "q6_glare",
    "q7_contrast_normal",
    "global_unusable_frame_yes_no",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def cohen_kappa(labels_a: list[str], labels_b: list[str]) -> float | None:
    count = len(labels_a)
    observed = sum(a == b for a, b in zip(labels_a, labels_b)) / count
    counts_a = Counter(labels_a)
    counts_b = Counter(labels_b)
    expected = sum(
        counts_a[label] / count * counts_b[label] / count
        for label in set(counts_a) | set(counts_b)
    )
    if math.isclose(expected, 1.0):
        return None
    return (observed - expected) / (1.0 - expected)


def binary_agreement(labels_a: list[str], labels_b: list[str]) -> dict[str, object]:
    pairs = [
        (a, b)
        for a, b in zip(labels_a, labels_b)
        if a in {"yes", "no"} and b in {"yes", "no"}
    ]
    yy = sum(a == "yes" and b == "yes" for a, b in pairs)
    yn = sum(a == "yes" and b == "no" for a, b in pairs)
    ny = sum(a == "no" and b == "yes" for a, b in pairs)
    nn = sum(a == "no" and b == "no" for a, b in pairs)
    positive_denominator = 2 * yy + yn + ny
    negative_denominator = 2 * nn + yn + ny
    return {
        "binary_evaluable_n": len(pairs),
        "uncertain_involved_n": len(labels_a) - len(pairs),
        "positive_agreement": 2 * yy / positive_denominator if positive_denominator else None,
        "negative_agreement": 2 * nn / negative_denominator if negative_denominator else None,
        "both_yes": yy,
        "a_yes_b_no": yn,
        "a_no_b_yes": ny,
        "both_no": nn,
    }


def main() -> None:
    sample = {row["sample_id"]: row for row in read_csv(AUDIT / "sample_manifest.csv")}
    annotator_a = {row["sample_id"]: row for row in read_csv(AUDIT / "annotator_a.csv")}
    annotator_b = {row["sample_id"]: row for row in read_csv(AUDIT / "annotator_b.csv")}
    if set(sample) != set(annotator_a) or set(sample) != set(annotator_b):
        raise RuntimeError("Audit sample IDs do not match the independent annotations")

    subsets = {
        "all": sorted(sample),
        "representative": sorted(
            sample_id
            for sample_id, row in sample.items()
            if row["representative_stratum"] == "1"
        ),
        "challenge": sorted(
            sample_id
            for sample_id, row in sample.items()
            if row["challenge_additional_stratum"] == "1"
        ),
    }
    rows = []
    for subset, sample_ids in subsets.items():
        for field in FIELDS:
            labels_a = [annotator_a[sample_id][field] for sample_id in sample_ids]
            labels_b = [annotator_b[sample_id][field] for sample_id in sample_ids]
            agreement_n = sum(a == b for a, b in zip(labels_a, labels_b))
            row: dict[str, object] = {
                "subset": subset,
                "field": field,
                "n": len(sample_ids),
                "agreement_n": agreement_n,
                "disagreement_n": len(sample_ids) - agreement_n,
                "raw_agreement": agreement_n / len(sample_ids),
                "cohen_kappa": cohen_kappa(labels_a, labels_b),
                "annotator_a_distribution": json.dumps(dict(sorted(Counter(labels_a).items()))),
                "annotator_b_distribution": json.dumps(dict(sorted(Counter(labels_b).items()))),
            }
            if field in BINARY_FIELDS:
                row.update(binary_agreement(labels_a, labels_b))
            else:
                row.update(
                    {
                        "binary_evaluable_n": None,
                        "uncertain_involved_n": None,
                        "positive_agreement": None,
                        "negative_agreement": None,
                        "both_yes": None,
                        "a_yes_b_no": None,
                        "a_no_b_yes": None,
                        "both_no": None,
                    }
                )
            rows.append(row)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} agreement rows to {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
