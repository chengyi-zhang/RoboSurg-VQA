#!/usr/bin/env python3
"""Build the frozen multi-question records for the shared VQA baseline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


Q8_TASK = "q8.mask_center_5pct_v1"
TASK_ORDER = [
    "q1",
    "q2",
    "q3",
    "q4.bleeding",
    "q4.smoke",
    "q4.occlusion",
    "q5",
    "q6",
    "q7",
    Q8_TASK,
    "q9",
]

TASK_LABELS = {
    "q1": ["nephrectomy (robot-assisted)"],
    "q2": [
        "covered kidney (fat/fascia-covered kidney)",
        "kidney (parenchyma)",
        "other abdominal soft tissue (peritoneum/mesentery/fat)",
        "small intestine",
    ],
    "q3": ["robotic camera"],
    "q4.bleeding": ["no", "yes"],
    "q4.smoke": ["no", "yes"],
    "q4.occlusion": ["no", "yes"],
    "q5": ["blurry", "clear", "mixed"],
    "q6": ["no", "yes"],
    "q7": ["no", "yes"],
    Q8_TASK: ["no", "yes"],
    "q9": ["near instrument jaws", "none"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the frozen records for the shared VQA baseline."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root.",
    )
    parser.add_argument(
        "--protocol-dir",
        type=Path,
        default=None,
        help="Directory containing questions.json and shared_vqa.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Destination for records and the dataset summary.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest().upper()


def get_target(row: dict[str, Any], task: str, q8: dict[str, str]) -> str:
    if task == Q8_TASK:
        return q8[row["record_id"]]
    if task.startswith("q4."):
        field = task.split(".", 1)[1]
        return str(row["answers"]["q4"]["value"][field])
    return str(row["answers"][task]["value"])


def is_applicable(
    row: dict[str, Any], task: str, applicable: set[tuple[str, str]]
) -> bool:
    return (row["site"], task) in applicable


def choose_validation_videos(
    records: list[dict[str, Any]], primary_tasks: list[str], count: int
) -> tuple[list[str], dict[str, Any]]:
    train_rows = [row for row in records if row["source_split"] == "train"]
    videos = sorted({row["source_sequence_id"] for row in train_rows})
    if len(videos) != 23:
        raise RuntimeError(f"Expected 23 training videos, found {len(videos)}")
    if count < 2 or count >= len(videos):
        raise ValueError("Invalid validation video count")

    rows_by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in train_rows:
        if row["task"] in primary_tasks:
            rows_by_video[row["source_sequence_id"]].append(row)

    all_counts = Counter(
        (row["task"], row["answer"])
        for row in train_rows
        if row["task"] in primary_tasks
    )
    all_task_counts = Counter(
        row["task"] for row in train_rows if row["task"] in primary_tasks
    )
    sites_by_video = {
        video: {row["site"] for row in rows_by_video[video]} for video in videos
    }

    best: tuple[float, tuple[str, ...]] | None = None
    best_details: dict[str, Any] = {}
    for candidate in itertools.combinations(videos, count):
        candidate_set = set(candidate)
        val_sites = set().union(*(sites_by_video[video] for video in candidate))
        core_sites = set().union(
            *(sites_by_video[video] for video in videos if video not in candidate_set)
        )
        if val_sites != {"site1", "site2"} or core_sites != {"site1", "site2"}:
            continue

        val_rows = [
            row
            for video in candidate
            for row in rows_by_video[video]
            if row["task"] in primary_tasks
        ]
        val_counts = Counter((row["task"], row["answer"]) for row in val_rows)
        val_task_counts = Counter(row["task"] for row in val_rows)

        score = 0.0
        missing_penalty = 0.0
        for key, total in all_counts.items():
            task = key[0]
            overall_p = total / all_task_counts[task]
            val_p = val_counts[key] / max(val_task_counts[task], 1)
            score += abs(val_p - overall_p) * (total ** 0.5)
            core_count = total - val_counts[key]
            if total >= 5 and (val_counts[key] == 0 or core_count == 0):
                missing_penalty += 50.0

        target_fraction = count / len(videos)
        observed_fraction = len(val_rows) / max(
            sum(len(rows_by_video[video]) for video in videos), 1
        )
        score += 100.0 * abs(observed_fraction - target_fraction)
        score += missing_penalty

        key = (score, candidate)
        if best is None or key < best:
            best = key
            best_details = {
                "score": score,
                "missing_penalty": missing_penalty,
                "validation_record_fraction": observed_fraction,
                "target_sequence_fraction": target_fraction,
            }

    if best is None:
        raise RuntimeError("No valid group-wise validation split found")
    return list(best[1]), best_details


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    protocol = args.protocol_dir.resolve() if args.protocol_dir else project_root / "configs"
    output_dir = args.output_dir.resolve() if args.output_dir else project_root / "data"
    annotations_path = project_root / "data" / "frame_annotations.jsonl"
    manifest_path = project_root / "data" / "manifest.jsonl"
    applicability_path = project_root / "data" / "task_applicability.csv"
    q8_targets_path = project_root / "data" / "mask_targets.csv"
    questions_path = protocol / "questions.json"
    config_path = protocol / "shared_vqa.json"
    output_path = output_dir / "vqa_records.jsonl"
    summary_path = output_dir / "vqa_summary.json"

    for required in (
        annotations_path,
        manifest_path,
        applicability_path,
        q8_targets_path,
        questions_path,
        config_path,
    ):
        if not required.is_file():
            raise FileNotFoundError(required)

    annotations = load_jsonl(annotations_path)
    manifest_rows = load_jsonl(manifest_path)
    with applicability_path.open("r", encoding="utf-8", newline="") as handle:
        applicability_rows = list(csv.DictReader(handle))
    applicable = {
        (row["site"], row["target"])
        for row in applicability_rows
        if row["role"] != "excluded_by_source_schema"
    }
    manifest = {row["record_id"]: row for row in manifest_rows}
    if len(annotations) != 5632 or len(manifest) != 5632:
        raise RuntimeError("The frozen corpus must contain 5,632 unique records")
    if {row["record_id"] for row in annotations} != set(manifest):
        raise RuntimeError("Annotation and manifest record IDs differ")

    with q8_targets_path.open("r", encoding="utf-8", newline="") as handle:
        q8_rows = list(csv.DictReader(handle))
    q8 = {row["record_id"]: row["label"] for row in q8_rows}
    if len(q8) != 5632 or set(q8) != set(manifest):
        raise RuntimeError("Q8 deterministic targets do not match the frozen corpus")

    question_spec = json.loads(questions_path.read_text(encoding="utf-8"))["tasks"]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if list(question_spec) != TASK_ORDER:
        raise RuntimeError("Question task order does not match the frozen task order")

    global_answers: list[str] = []
    for task in TASK_ORDER:
        for answer in TASK_LABELS[task]:
            if answer not in global_answers:
                global_answers.append(answer)
    answer_to_index = {answer: index for index, answer in enumerate(global_answers)}

    base_records: list[dict[str, Any]] = []
    for annotation in annotations:
        record_id = annotation["record_id"]
        source = manifest[record_id]
        if source["split"] != annotation["split"]:
            raise RuntimeError(f"Split mismatch for {record_id}")
        for task in TASK_ORDER:
            if not is_applicable(annotation, task, applicable):
                continue
            answer = get_target(annotation, task, q8)
            if answer not in TASK_LABELS[task]:
                raise RuntimeError(f"Unexpected answer for {task}: {answer}")
            base_records.append(
                {
                    "qa_id": f"{record_id}|{task}",
                    "record_id": record_id,
                    "video_key": source["video_key"],
                    "source_sequence_id": f"{source['split']}|{source['video_key']}",
                    "source_split": source["split"],
                    "dataset": annotation["dataset"],
                    "site": annotation["site"],
                    "image_path": source["image_path"],
                    "task": task,
                    "answer": answer,
                    "answer_index": answer_to_index[answer],
                    "valid_answers": TASK_LABELS[task],
                    "question_canonical": question_spec[task]["canonical"],
                    "question_train_paraphrases": question_spec[task][
                        "train_paraphrases"
                    ],
                    "question_heldout_paraphrase": question_spec[task][
                        "heldout_paraphrase"
                    ],
                }
            )

    validation_videos, split_details = choose_validation_videos(
        base_records,
        primary_tasks=config["primary_tasks"],
        count=int(config["validation_sequence_count"]),
    )
    validation_set = set(validation_videos)
    for row in base_records:
        if row["source_split"] == "test":
            row["validation_role"] = "test"
        elif row["source_sequence_id"] in validation_set:
            row["validation_role"] = "validation"
        else:
            row["validation_role"] = "train_core"

    train_videos = {
        row["source_sequence_id"]
        for row in base_records
        if row["source_split"] == "train"
    }
    test_videos = {
        row["source_sequence_id"]
        for row in base_records
        if row["source_split"] == "test"
    }
    if train_videos & test_videos:
        raise RuntimeError("Source-sequence leakage between train and test")
    if len(train_videos) != 23 or len(test_videos) != 6:
        raise RuntimeError("Expected 23 train and 6 test source sequences")

    for row in base_records:
        train_texts = set(row["question_train_paraphrases"])
        if row["question_canonical"] in train_texts:
            raise RuntimeError("Canonical question duplicated in training paraphrases")
        if row["question_heldout_paraphrase"] in train_texts:
            raise RuntimeError("Held-out paraphrase appears in training paraphrases")
        if row["question_heldout_paraphrase"] == row["question_canonical"]:
            raise RuntimeError("Held-out paraphrase equals canonical question")

    output_dir.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in base_records:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    counts_by_role = Counter(row["validation_role"] for row in base_records)
    counts_by_task_role = Counter(
        (row["task"], row["validation_role"]) for row in base_records
    )
    summary = {
        "protocol": config["protocol"],
        "records": len(base_records),
        "unique_frames": len({row["record_id"] for row in base_records}),
        "global_answer_vocabulary": global_answers,
        "global_answer_count": len(global_answers),
        "task_labels": TASK_LABELS,
        "task_order": TASK_ORDER,
        "primary_tasks": config["primary_tasks"],
        "perceptual_tasks": config["perceptual_tasks"],
        "train_sequence_count": len(train_videos),
        "test_sequence_count": len(test_videos),
        "validation_sequences": validation_videos,
        "validation_split_details": split_details,
        "counts_by_role": dict(sorted(counts_by_role.items())),
        "counts_by_task_role": {
            f"{task}|{role}": count
            for (task, role), count in sorted(counts_by_task_role.items())
        },
        "human_labels_used": False,
        "task_specific_answer_filtering": False,
        "inputs": {
            "annotations": annotations_path.relative_to(project_root).as_posix(),
            "annotations_sha256": sha256(annotations_path),
            "manifest": manifest_path.relative_to(project_root).as_posix(),
            "manifest_sha256": sha256(manifest_path),
            "task_applicability": applicability_path.relative_to(project_root).as_posix(),
            "task_applicability_sha256": sha256(applicability_path),
            "q8_targets": q8_targets_path.relative_to(project_root).as_posix(),
            "q8_targets_sha256": sha256(q8_targets_path),
            "questions": questions_path.relative_to(project_root).as_posix(),
            "questions_sha256": sha256(questions_path),
            "config": config_path.relative_to(project_root).as_posix(),
            "config_sha256": sha256(config_path),
        },
        "output": output_path.relative_to(project_root).as_posix(),
        "output_sha256": sha256(output_path),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
