#!/usr/bin/env python3
"""Build the deterministic central-instrument target from official masks."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


WORKSPACE = Path(__file__).resolve().parents[1]
FEATURES = (
    WORKSPACE
    / "features"
    / "taskwise_resnet18_features.npz"
)
MANIFEST = WORKSPACE / "data" / "manifest.jsonl"
ANNOTATIONS = WORKSPACE / "data" / "frame_annotations.jsonl"
OUTPUT = WORKSPACE / "data" / "mask_targets.csv"
SUMMARY = WORKSPACE / "results" / "mask_target_summary.json"

TARGET_NAME = "q8_mask_center_5pct_v1"
THRESHOLD = 0.05
EXPECTED_RECORDS = 5632


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest().upper()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def candidate_q8(row: dict[str, Any]) -> str:
    return str(row["answers"]["q8"]["value"])


def main() -> None:
    for path in (FEATURES, MANIFEST, ANNOTATIONS):
        if not path.is_file():
            raise FileNotFoundError(path)

    feature_data = np.load(FEATURES)
    record_ids = feature_data["record_ids"].astype(str)
    names = feature_data["mask_feature_names"].astype(str).tolist()
    mask_features = feature_data["mask_features"]
    if "central_area_instrument_fraction" not in names:
        raise RuntimeError("Central instrument occupancy feature is missing")
    central_index = names.index("central_area_instrument_fraction")
    central_fractions = mask_features[:, central_index].astype(float)

    manifest_rows = read_jsonl(MANIFEST)
    manifest = {row["record_id"]: row for row in manifest_rows}
    annotation_rows = read_jsonl(ANNOTATIONS)
    annotations = {row["record_id"]: row for row in annotation_rows}
    expected = set(record_ids)
    if (
        len(record_ids) != EXPECTED_RECORDS
        or len(expected) != EXPECTED_RECORDS
        or expected != set(manifest)
        or expected != set(annotations)
    ):
        raise RuntimeError("Q8 source record sets are not the same frozen 5,632 frames")
    if np.any(~np.isfinite(central_fractions)) or np.any(central_fractions < 0) or np.any(central_fractions > 1):
        raise RuntimeError("Invalid central instrument occupancy values")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for record_id, fraction in zip(record_ids, central_fractions):
        source = manifest[record_id]
        label = "yes" if fraction >= THRESHOLD else "no"
        rows.append(
            {
                "record_id": record_id,
                "dataset": source["dataset"],
                "site": source["site"],
                "split": source["split"],
                "video_key": source["video_key"],
                "frame_index": source["frame_index"],
                "target_name": TARGET_NAME,
                "central_region": "x=25%-75%; y=25%-75%",
                "central_instrument_fraction": f"{fraction:.10f}",
                "positive_threshold": f">={THRESHOLD:.2f}",
                "label": label,
            }
        )
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    label_counts = Counter(row["label"] for row in rows)
    strata = {}
    for dataset in sorted({row["dataset"] for row in rows}):
        for split in sorted({row["split"] for row in rows if row["dataset"] == dataset}):
            selected = [row for row in rows if row["dataset"] == dataset and row["split"] == split]
            strata[f"{dataset}/{split}"] = dict(Counter(row["label"] for row in selected))

    candidate_pairs = Counter(
        (candidate_q8(annotations[row["record_id"]]), row["label"])
        for row in rows
    )
    valid_candidates = [
        row for row in rows if candidate_q8(annotations[row["record_id"]]) in {"yes", "no"}
    ]
    candidate_agreement = sum(
        candidate_q8(annotations[row["record_id"]]) == row["label"]
        for row in valid_candidates
    ) / len(valid_candidates)

    sensitivity = {
        ">0.00": dict(Counter(np.where(central_fractions > 0, "yes", "no").tolist()))
    }
    for threshold in (0.01, 0.05):
        labels = np.where(central_fractions >= threshold, "yes", "no")
        sensitivity[f">={threshold:.2f}"] = dict(Counter(labels.tolist()))

    report = {
        "target_name": TARGET_NAME,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "definition": {
            "question": "Does the official instrument mask occupy at least 5% of the central image region?",
            "central_region": "rectangle spanning 25%-75% in width and 25%-75% in height",
            "positive_rule": "central_area_instrument_fraction >= 0.05",
            "mask_source": "official instrument masks",
        },
        "records": len(rows),
        "label_counts": dict(label_counts),
        "stratified_label_counts": strata,
        "threshold_sensitivity": sensitivity,
        "candidate_q8_comparison": {
            "evaluable_records": len(valid_candidates),
            "raw_agreement": candidate_agreement,
            "pair_counts": {
                f"candidate={old}|deterministic={new}": count
                for (old, new), count in sorted(candidate_pairs.items())
            },
        },
        "inputs": {
            "features": FEATURES.relative_to(WORKSPACE).as_posix(),
            "features_sha256": sha256(FEATURES),
            "manifest": MANIFEST.relative_to(WORKSPACE).as_posix(),
            "manifest_sha256": sha256(MANIFEST),
            "annotations": ANNOTATIONS.relative_to(WORKSPACE).as_posix(),
            "annotations_sha256": sha256(ANNOTATIONS),
        },
        "output": OUTPUT.relative_to(WORKSPACE).as_posix(),
        "output_sha256": sha256(OUTPUT),
    }
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
