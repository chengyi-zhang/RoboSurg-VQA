#!/usr/bin/env python3
"""Quantify coverage and distribution shift of the frozen human-audit sample."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


FIELDS = [
    "q4_bleeding",
    "q4_smoke",
    "q4_occlusion",
    "q5_value",
    "q6_glare",
    "q7_contrast_normal",
    "q9_smoke_region",
]


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


def flatten_machine(row: dict[str, Any]) -> dict[str, str]:
    q4 = row["answers"]["q4"]["value"]
    return {
        "q4_bleeding": str(q4["bleeding"]),
        "q4_smoke": str(q4["smoke"]),
        "q4_occlusion": str(q4["occlusion"]),
        "q5_value": str(row["answers"]["q5"]["value"]),
        "q6_glare": str(row["answers"]["q6"]["value"]),
        "q7_contrast_normal": str(row["answers"]["q7"]["value"]),
        "q9_smoke_region": str(row["answers"]["q9"]["value"]),
    }


def distribution_rows(
    field: str,
    full_values: list[str],
    sample_values: list[str],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    full_counts = Counter(full_values)
    sample_counts = Counter(sample_values)
    labels = sorted(set(full_counts) | set(sample_counts))
    rows = []
    absolute_differences = []
    for label in labels:
        full_prop = full_counts[label] / len(full_values)
        sample_prop = sample_counts[label] / len(sample_values)
        difference = sample_prop - full_prop
        absolute_differences.append(abs(difference))
        rows.append(
            {
                "field": field,
                "label": label,
                "full_n": full_counts[label],
                "full_proportion": full_prop,
                "representative_n": sample_counts[label],
                "representative_proportion": sample_prop,
                "difference_percentage_points": difference * 100.0,
            }
        )
    summary = {
        "field": field,
        "total_variation_distance": 0.5 * sum(absolute_differences),
        "maximum_absolute_percentage_point_difference": max(absolute_differences) * 100.0,
        "full_class_count": len(full_counts),
        "representative_class_count": len(sample_counts),
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    args = parser.parse_args()
    project = args.project.resolve()
    annotations_path = project / "data" / "frame_annotations.jsonl"
    manifest_path = project / "data" / "manifest.jsonl"
    human_path = project / "data" / "human_reference" / "reference.csv"
    output = project / "results" / "audit_sample"
    output.mkdir(parents=True, exist_ok=True)

    annotations = {row["record_id"]: row for row in load_jsonl(annotations_path)}
    manifest = {row["record_id"]: row for row in load_jsonl(manifest_path)}
    human_rows = read_csv(human_path)
    if set(annotations) != set(manifest):
        raise RuntimeError("Annotation and manifest IDs do not match")
    if len(human_rows) != 250 or len({row["record_id"] for row in human_rows}) != 250:
        raise RuntimeError("Expected 250 unique final human-reference rows")
    if not {row["record_id"] for row in human_rows}.issubset(annotations):
        raise RuntimeError("Human-reference IDs are not a subset of the frozen corpus")

    representative = [row for row in human_rows if row["representative_stratum"] == "1"]
    challenge = [row for row in human_rows if row["challenge_additional_stratum"] == "1"]
    if len(representative) != 200 or len(challenge) != 50:
        raise RuntimeError("Expected 200 representative and 50 challenge records")
    if {row["record_id"] for row in representative} & {row["record_id"] for row in challenge}:
        raise RuntimeError("Representative and challenge strata must be disjoint")

    machine = {record_id: flatten_machine(row) for record_id, row in annotations.items()}
    distribution_detail = []
    distribution_summary = []
    for field in FIELDS:
        detail, summary = distribution_rows(
            field,
            [machine[record_id][field] for record_id in sorted(machine)],
            [machine[row["record_id"]][field] for row in representative],
        )
        distribution_detail.extend(detail)
        distribution_summary.append(summary)
    write_csv(
        output / "machine_label_distribution_comparison.csv",
        distribution_detail,
        [
            "field", "label", "full_n", "full_proportion", "representative_n",
            "representative_proportion", "difference_percentage_points",
        ],
    )
    write_csv(
        output / "distribution_distance_summary.csv",
        distribution_summary,
        [
            "field", "total_variation_distance", "maximum_absolute_percentage_point_difference",
            "full_class_count", "representative_class_count",
        ],
    )

    coverage_rows = []
    strata = {
        "full_corpus": [{"record_id": record_id} for record_id in sorted(manifest)],
        "representative_200": representative,
        "challenge_50": challenge,
        "all_human_250": human_rows,
    }
    for stratum, rows in strata.items():
        ids = [row["record_id"] for row in rows]
        sites = Counter(manifest[record_id]["site"] for record_id in ids)
        splits = Counter(manifest[record_id]["split"] for record_id in ids)
        videos = Counter(manifest[record_id]["video_key"] for record_id in ids)
        for dimension, counts in [("site", sites), ("split", splits), ("video_key", videos)]:
            for value, count in sorted(counts.items()):
                coverage_rows.append(
                    {
                        "stratum": stratum,
                        "dimension": dimension,
                        "value": value,
                        "count": count,
                        "proportion": count / len(ids),
                    }
                )
    write_csv(
        output / "coverage_by_site_split_video.csv",
        coverage_rows,
        ["stratum", "dimension", "value", "count", "proportion"],
    )

    human_distribution_rows = []
    for field in FIELDS + ["global_unusable_frame_yes_no"]:
        counts = Counter(row[field] for row in representative)
        for label, count in sorted(counts.items()):
            human_distribution_rows.append(
                {
                    "stratum": "representative_200",
                    "field": field,
                    "label": label,
                    "count": count,
                    "proportion": count / len(representative),
                }
            )
    write_csv(
        output / "adjudicated_human_distribution_representative_200.csv",
        human_distribution_rows,
        ["stratum", "field", "label", "count", "proportion"],
    )

    full_videos = {manifest[record_id]["video_key"] for record_id in manifest}
    representative_videos = {manifest[row["record_id"]]["video_key"] for row in representative}
    all_human_videos = {manifest[row["record_id"]]["video_key"] for row in human_rows}
    summary = {
        "full_corpus_n": len(manifest),
        "representative_n": len(representative),
        "challenge_n": len(challenge),
        "all_human_n": len(human_rows),
        "full_video_count": len(full_videos),
        "representative_video_count": len(representative_videos),
        "all_human_video_count": len(all_human_videos),
        "representative_video_coverage_fraction": len(representative_videos) / len(full_videos),
        "uncovered_representative_videos": sorted(full_videos - representative_videos),
        "distribution_distance": distribution_summary,
        "annotations_sha256": sha256(annotations_path),
        "manifest_sha256": sha256(manifest_path),
        "human_reference_sha256": sha256(human_path),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
