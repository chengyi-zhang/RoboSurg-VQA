#!/usr/bin/env python3
"""Run lightweight integrity checks on the public repository."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
import re
from math import isclose
from pathlib import Path
from typing import Any


EXPECTED = {
    "README.md",
    "CITATION.cff",
    "LICENSE",
    "requirements.txt",
    "configs/shared_vqa.json",
    "configs/questions.json",
    "SHA256SUMS.txt",
    "data/manifest.jsonl",
    "data/frame_annotations.jsonl",
    "data/task_applicability.csv",
    "data/mask_targets.csv",
    "data/vqa_records.jsonl",
    "data/vqa_summary.json",
    "data/human_reference/sample_manifest.csv",
    "data/human_reference/candidate_labels.csv",
    "data/human_reference/annotator_a.csv",
    "data/human_reference/annotator_b.csv",
    "data/human_reference/adjudication.csv",
    "data/human_reference/reference.csv",
    "results/taskwise/predictions.csv",
    "results/shared_vqa/predictions.csv",
    "results/shared_vqa/aggregate_metrics.csv",
    "results/human_reference/human_reference_aggregate.csv",
    "results/human_reference/taskwise_human_reference_metrics.csv",
}
INTERNAL_NAME = re.compile(
    r"(?:^|[_-])(rc\d+|draft|backup|temporary|focused[-_]?revision|source[-_]?grounded)(?:[_-]|$)",
    re.IGNORECASE,
)
LOCAL_DIRECTORIES = {
    ".git", ".venv", ".cache", "__pycache__", "external_data",
    "features", "checkpoints", "outputs",
}


def repository_files(root: Path) -> list[Path]:
    files = []
    for directory, subdirectories, names in os.walk(root):
        subdirectories[:] = [name for name in subdirectories if name not in LOCAL_DIRECTORIES]
        files.extend(Path(directory) / name for name in names)
    return files


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().lower()


def check_checksums(root: Path, errors: list[str]) -> None:
    checksum_file = root / "SHA256SUMS.txt"
    if not checksum_file.is_file():
        errors.append("Missing SHA256SUMS.txt")
        return
    for line in checksum_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        relative = relative.lstrip("* ")
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            errors.append(f"Checksum path outside repository: {relative}")
            continue
        if not path.is_file() or sha256(path) != expected.lower():
            errors.append(f"Checksum mismatch: {relative}")


def check_source_arrays(
    manifest: list[dict[str, Any]], data_root: Path, errors: list[str]
) -> None:
    import numpy as np

    for row in manifest:
        for kind in ("image", "mask"):
            relative = Path(row[f"{kind}_path"]).relative_to("external_data")
            path = data_root / relative
            if not path.is_file():
                errors.append(f"Missing source array: {path}")
                continue
            try:
                array = np.load(path, mmap_mode="r", allow_pickle=False)
                if list(array.shape) != row[f"{kind}_shape"]:
                    errors.append(f"Shape mismatch: {row['record_id']} {kind}")
                if str(array.dtype) != row[f"{kind}_dtype"]:
                    errors.append(f"Dtype mismatch: {row['record_id']} {kind}")
                if kind == "mask" and (array.min() < 0 or array.max() > 3):
                    errors.append(f"Invalid instrument-part encoding: {row['record_id']}")
            except (OSError, ValueError) as error:
                errors.append(f"Cannot read source array {path}: {error}")


def find_row(rows: list[dict[str, str]], **query: str) -> dict[str, str]:
    matches = [row for row in rows if all(row.get(key) == value for key, value in query.items())]
    if len(matches) != 1:
        raise ValueError(f"Expected one result row for {query}, found {len(matches)}")
    return matches[0]


def check_value(
    rows: list[dict[str, str]], field: str, expected: float, errors: list[str], **query: str
) -> None:
    try:
        observed = float(find_row(rows, **query)[field])
    except (KeyError, TypeError, ValueError) as error:
        errors.append(f"Cannot read {field} for {query}: {error}")
        return
    if not isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12):
        errors.append(f"Unexpected {field} for {query}: {observed} != {expected}")


def check_reported_values(root: Path, errors: list[str]) -> None:
    shared = read_csv(root / "results" / "shared_vqa" / "aggregate_metrics.csv")
    for condition, expected in {
        "answer_frequency": 0.39119604255254,
        "question_only": 0.255219792296751,
        "image_only": 0.29625343955658107,
        "image_question": 0.540134926148988,
    }.items():
        check_value(
            shared,
            "mean_macro_f1_fixed_labels",
            expected,
            errors,
            condition=condition,
            wording="canonical",
            task_group="primary_discriminative_tasks",
        )
    check_value(
        shared,
        "mean_macro_f1_fixed_labels",
        0.5276872193637524,
        errors,
        condition="image_question",
        wording="heldout_paraphrase",
        task_group="primary_discriminative_tasks",
    )

    human = read_csv(
        root / "results" / "human_reference" / "human_reference_aggregate.csv"
    )
    for condition, expected in {
        "majority_prior": 0.4499600121342489,
        "rgb_linear": 0.5273255869055733,
        "rgb_mask_linear": 0.5614704967697925,
    }.items():
        check_value(human, "mean_macro_f1", expected, errors, condition=condition)

    taskwise_human = read_csv(
        root
        / "results"
        / "human_reference"
        / "taskwise_human_reference_metrics.csv"
    )
    for condition, expected in {
        "majority_prior": 0.5,
        "rgb_mask_linear": 0.5,
    }.items():
        check_value(
            taskwise_human,
            "macro_f1_ci_high",
            expected,
            errors,
            task="q5",
            condition=condition,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--source-arrays", nargs="?", const="external_data", type=Path,
        help="Also check prepared arrays; optionally provide their external_data root.",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    errors: list[str] = []

    files = repository_files(root)
    relative_files = {path.relative_to(root).as_posix() for path in files}
    errors.extend(f"Missing file: {path}" for path in sorted(EXPECTED - relative_files))
    for path in (root / "scripts").glob("*.py"):
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as error:
            errors.append(f"Python syntax error: {path.name}: {error}")
    for relative in relative_files:
        if INTERNAL_NAME.search(relative):
            errors.append(f"Internal working name: {relative}")
    for path in files:
        if path.stat().st_size >= 100 * 1024 * 1024:
            errors.append(f"File exceeds GitHub's 100 MiB limit: {path.relative_to(root)}")

    manifest = read_jsonl(root / "data" / "manifest.jsonl")
    annotations = read_jsonl(root / "data" / "frame_annotations.jsonl")
    mask_targets = read_csv(root / "data" / "mask_targets.csv")
    vqa_records = read_jsonl(root / "data" / "vqa_records.jsonl")
    if len(manifest) != 5632:
        errors.append(f"Expected 5,632 manifest rows, found {len(manifest):,}")
    if len(annotations) != 5632:
        errors.append(f"Expected 5,632 annotation rows, found {len(annotations):,}")
    if len(mask_targets) != 5632:
        errors.append(f"Expected 5,632 mask targets, found {len(mask_targets):,}")
    if len(vqa_records) != 59552:
        errors.append(f"Expected 59,552 VQA records, found {len(vqa_records):,}")

    manifest_ids = {row["record_id"] for row in manifest}
    if manifest_ids != {row["record_id"] for row in annotations}:
        errors.append("Manifest and annotation IDs differ")
    if manifest_ids != {row["record_id"] for row in mask_targets}:
        errors.append("Manifest and mask-target IDs differ")
    if len(manifest_ids) != len(manifest):
        errors.append("Duplicate manifest record IDs")
    if len({row["qa_id"] for row in vqa_records}) != len(vqa_records):
        errors.append("Duplicate question-answer IDs")
    split_sequences = {
        split: {row["source_sequence_id"] for row in vqa_records if row["source_split"] == split}
        for split in ("train", "test")
    }
    if len(split_sequences["train"]) != 23 or len(split_sequences["test"]) != 6:
        errors.append("Expected 23 training and six test source sequences")
    if split_sequences["train"] & split_sequences["test"]:
        errors.append("Source-sequence overlap between training and test")
    for row in manifest:
        for field in ("image_path", "mask_path"):
            value = str(row[field])
            if not value.startswith("external_data/") or "\\" in value or Path(value).is_absolute():
                errors.append(f"Non-portable {field}: {value}")
                break

    sample = read_csv(root / "data" / "human_reference" / "sample_manifest.csv")
    reference = read_csv(root / "data" / "human_reference" / "reference.csv")
    if len(sample) != 250 or len(reference) != 250:
        errors.append("Human audit must contain 250 records")
    if sum(row["representative_stratum"] == "1" for row in sample) != 200:
        errors.append("Representative audit stratum must contain 200 records")
    if sum(row["challenge_additional_stratum"] == "1" for row in sample) != 50:
        errors.append("Challenge audit stratum must contain 50 records")
    if {row["record_id"] for row in sample} != {row["record_id"] for row in reference}:
        errors.append("Audit sample and adjudicated reference IDs differ")

    check_checksums(root, errors)
    check_reported_values(root, errors)
    if args.source_arrays is not None:
        data_root = args.source_arrays
        if not data_root.is_absolute():
            data_root = root / data_root
        check_source_arrays(manifest, data_root.resolve(), errors)
    result = {
        "status": "pass" if not errors else "fail",
        "files": len(files),
        "frames": len(manifest),
        "vqa_records": len(vqa_records),
        "audit_records": len(reference),
        "errors": errors,
    }
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
