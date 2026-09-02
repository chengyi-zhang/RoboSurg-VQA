#!/usr/bin/env python3
"""Recreate the 200-frame representative and 50-frame challenge audit sample."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifest.jsonl"
ANNOTATIONS = ROOT / "data" / "frame_annotations.jsonl"
MASK_TARGETS = ROOT / "data" / "mask_targets.csv"
OUTPUT = ROOT / "data" / "human_reference" / "sample_manifest.csv"

SEED = 20260821
FRAMES_PER_VIDEO = 8
CHALLENGE_SIZE = 50
TASKS = [
    "q1",
    "q2",
    "q4.bleeding",
    "q4.smoke",
    "q4.occlusion",
    "q5",
    "q6",
    "q7",
    "q8",
    "q9",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def answer(annotation: dict[str, Any], task: str) -> dict[str, Any]:
    root = task.split(".", 1)[0]
    if root in {"q1", "q2", "q3"}:
        candidates = annotation.get("candidate_answers_q1_q2_q3", {})
        item = candidates.get(root, annotation["answers"][root])
    else:
        item = annotation["answers"][root]
    if "." not in task:
        return item
    field = task.split(".", 1)[1]
    return {"value": item["value"][field], "confidence": item["confidence"]}


def temporal_sample(
    rows: list[dict[str, Any]], count: int, rng: np.random.Generator
) -> list[dict[str, Any]]:
    rows = sorted(rows, key=lambda row: (int(row["frame_index"]), row["record_id"]))
    selected = []
    for indices in np.array_split(np.arange(len(rows)), count):
        selected.append(rows[int(rng.choice(indices))])
    return selected


def challenge_score(
    annotation: dict[str, Any],
    label_counts: dict[str, Counter[str]],
    central_fraction: float,
) -> float:
    score = 0.0
    for task in TASKS:
        value = str(answer(annotation, task)["value"])
        score += math.log(len(label_counts[task]) + 1.0) / math.sqrt(
            max(1, label_counts[task][value])
        )
    confidence = min(float(answer(annotation, task)["confidence"]) for task in TASKS)
    score += 5.0 * (1.0 - confidence)
    score += 1.5 if answer(annotation, "q5")["value"] != "clear" else 0.0
    score += 2.0 if answer(annotation, "q6")["value"] == "yes" else 0.0
    score += 2.0 if answer(annotation, "q7")["value"] == "no" else 0.0
    score += 1.5 if answer(annotation, "q4.occlusion")["value"] == "yes" else 0.0
    machine_q8 = answer(annotation, "q8")["value"] == "yes"
    score += 3.0 if machine_q8 != (central_fraction >= 0.05) else 0.0
    return score


def select_challenge_records(
    manifest: list[dict[str, Any]],
    annotations: dict[str, dict[str, Any]],
    representative_ids: set[str],
    central_fractions: dict[str, float],
) -> tuple[set[str], dict[str, list[str]]]:
    label_counts = {
        task: Counter(str(answer(annotations[row["record_id"]], task)["value"]) for row in manifest)
        for task in TASKS
    }
    reasons: dict[str, list[str]] = defaultdict(list)
    selected: set[str] = set()

    for row in manifest:
        record_id = row["record_id"]
        if answer(annotations[record_id], "q4.smoke")["value"] == "yes":
            reasons[record_id].append("all_machine_smoke_positive")
            if record_id not in representative_ids:
                selected.add(record_id)

    ranked = []
    for row in manifest:
        record_id = row["record_id"]
        if record_id in representative_ids or record_id in selected:
            continue
        ranked.append(
            (
                challenge_score(
                    annotations[record_id], label_counts, central_fractions[record_id]
                ),
                record_id,
                row,
            )
        )
    ranked.sort(key=lambda item: (-item[0], item[1]))

    per_video = Counter(
        row["video_key"] for row in manifest if row["record_id"] in selected
    )
    while len(selected) < CHALLENGE_SIZE:
        candidate = next(
            (
                item
                for item in ranked
                if item[1] not in selected and per_video[item[2]["video_key"]] < 6
            ),
            None,
        )
        if candidate is None:
            candidate = next((item for item in ranked if item[1] not in selected), None)
        if candidate is None:
            raise RuntimeError("Insufficient records for the challenge sample")
        _, record_id, row = candidate
        selected.add(record_id)
        per_video[row["video_key"]] += 1
        reasons[record_id].append("ranked_rare_low_confidence_or_mask_disagreement")

    for record_id in selected:
        annotation = annotations[record_id]
        if answer(annotation, "q5")["value"] != "clear":
            reasons[record_id].append("nonclear_quality")
        if answer(annotation, "q6")["value"] == "yes":
            reasons[record_id].append("glare_positive")
        if answer(annotation, "q7")["value"] == "no":
            reasons[record_id].append("contrast_abnormal")
        if answer(annotation, "q4.occlusion")["value"] == "yes":
            reasons[record_id].append("semantic_occlusion_positive")
        machine_q8 = answer(annotation, "q8")["value"] == "yes"
        if machine_q8 != (central_fractions[record_id] >= 0.05):
            reasons[record_id].append("q8_disagrees_with_5pct_mask_rule")
        if min(float(answer(annotation, task)["confidence"]) for task in TASKS) <= 0.6:
            reasons[record_id].append("machine_min_confidence_le_0.6")
    return selected, {key: sorted(set(value)) for key, value in reasons.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    manifest = read_jsonl(MANIFEST)
    annotations = {row["record_id"]: row for row in read_jsonl(ANNOTATIONS)}
    mask_rows = read_csv(MASK_TARGETS)
    central_fractions = {
        row["record_id"]: float(row["central_instrument_fraction"]) for row in mask_rows
    }
    record_ids = {row["record_id"] for row in manifest}
    if record_ids != set(annotations) or record_ids != set(central_fractions):
        raise RuntimeError("Manifest, annotations, and mask targets are not aligned")

    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in manifest:
        by_video[row["video_key"]].append(row)
    rng = np.random.default_rng(SEED)
    representative = [
        row
        for video_key in sorted(by_video)
        for row in temporal_sample(by_video[video_key], FRAMES_PER_VIDEO, rng)
    ]
    representative_ids = {row["record_id"] for row in representative}
    challenge_ids, reasons = select_challenge_records(
        manifest, annotations, representative_ids, central_fractions
    )
    selected_ids = representative_ids | challenge_ids
    if len(representative_ids) != 200 or len(challenge_ids) != 50 or len(selected_ids) != 250:
        raise RuntimeError("Unexpected audit sample size")

    manifest_by_id = {row["record_id"]: row for row in manifest}
    rows = []
    for index, record_id in enumerate(sorted(selected_ids), start=1):
        source = manifest_by_id[record_id]
        smoke = answer(annotations[record_id], "q4.smoke")["value"] == "yes"
        rows.append(
            {
                "sample_id": f"HR{index:04d}",
                "record_id": record_id,
                "dataset": source["dataset"],
                "site": source["site"],
                "split": source["split"],
                "video_key": source["video_key"],
                "frame_index": source["frame_index"],
                "representative_stratum": int(record_id in representative_ids),
                "challenge_additional_stratum": int(record_id in challenge_ids),
                "challenge_smoke_flag": int(smoke),
                "challenge_reasons": " | ".join(reasons.get(record_id, [])),
                "central_area_instrument_fraction": central_fractions[record_id],
                "sequence_key": f"{source['site']}/{source['split']}/{source['video_id']}",
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} records to {args.output}")


if __name__ == "__main__":
    main()
