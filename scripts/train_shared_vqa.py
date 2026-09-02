#!/usr/bin/env python3
"""Train and evaluate a frozen-encoder shared question-conditioned VQA baseline."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import random
import time
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score
from torch import nn
from torch.nn import functional as F


WORKSPACE = Path(__file__).resolve().parents[1]
DATA = WORKSPACE / "data" / "vqa_records.jsonl"
DATA_SUMMARY = WORKSPACE / "data" / "vqa_summary.json"
FEATURES = WORKSPACE / "features" / "biomedclip_frozen_features.npz"
CONFIG_PATH = WORKSPACE / "configs" / "shared_vqa.json"
RESULTS = WORKSPACE / "results" / "shared_vqa"
CHECKPOINTS = RESULTS / "checkpoints"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest().upper()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class SharedVQAHead(nn.Module):
    def __init__(
        self,
        image_dim: int,
        question_dim: int,
        projection_dim: int,
        answer_count: int,
        dropout: float,
        condition: str,
    ) -> None:
        super().__init__()
        if condition not in {"question_only", "image_only", "image_question"}:
            raise ValueError(condition)
        self.condition = condition
        self.image_projection = nn.Sequential(
            nn.Linear(image_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
        )
        self.question_projection = nn.Sequential(
            nn.Linear(question_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(projection_dim * 3, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(projection_dim, answer_count)

    def forward(self, image: torch.Tensor, question: torch.Tensor) -> torch.Tensor:
        if self.condition == "question_only":
            question_hidden = self.question_projection(question)
            image_hidden = torch.zeros_like(question_hidden)
        elif self.condition == "image_only":
            image_hidden = self.image_projection(image)
            question_hidden = torch.zeros_like(image_hidden)
        else:
            image_hidden = self.image_projection(image)
            question_hidden = self.question_projection(question)
        hidden = self.fusion(
            torch.cat(
                [image_hidden, question_hidden, image_hidden * question_hidden],
                dim=-1,
            )
        )
        return self.classifier(hidden)


def examples_for_training(
    rows: list[dict[str, Any]], roles: set[str]
) -> list[tuple[dict[str, Any], str]]:
    examples: list[tuple[dict[str, Any], str]] = []
    for row in rows:
        if row["validation_role"] not in roles:
            continue
        examples.append((row, row["question_canonical"]))
        for question in row["question_train_paraphrases"]:
            examples.append((row, question))
    return examples


def examples_for_evaluation(
    rows: list[dict[str, Any]], wording: str, task_order: list[str]
) -> list[tuple[dict[str, Any], str]]:
    test_rows = [row for row in rows if row["validation_role"] == "test"]
    if wording == "canonical":
        return [(row, row["question_canonical"]) for row in test_rows]
    if wording == "heldout_paraphrase":
        return [(row, row["question_heldout_paraphrase"]) for row in test_rows]
    if wording == "wrong_question":
        canonical_by_task = {
            row["task"]: row["question_canonical"] for row in rows
        }
        wrong_task = {
            task: task_order[(index + 1) % len(task_order)]
            for index, task in enumerate(task_order)
        }
        return [
            (row, canonical_by_task[wrong_task[row["task"]]]) for row in test_rows
        ]
    raise ValueError(wording)


def make_arrays(
    examples: list[tuple[dict[str, Any], str]],
    image_map: dict[str, np.ndarray],
    question_map: dict[str, np.ndarray],
) -> dict[str, Any]:
    return {
        "image": np.asarray([image_map[row["record_id"]] for row, _ in examples], dtype=np.float32),
        "question": np.asarray([question_map[text] for _, text in examples], dtype=np.float32),
        "target": np.asarray([row["answer_index"] for row, _ in examples], dtype=np.int64),
        "answer": np.asarray([row["answer"] for row, _ in examples], dtype=str),
        "task": np.asarray([row["task"] for row, _ in examples], dtype=str),
        "video": np.asarray(
            [row["source_sequence_id"] for row, _ in examples], dtype=str
        ),
        "site": np.asarray([row["site"] for row, _ in examples], dtype=str),
        "qa_id": np.asarray([row["qa_id"] for row, _ in examples], dtype=str),
        "record_id": np.asarray([row["record_id"] for row, _ in examples], dtype=str),
        "question_text": np.asarray([text for _, text in examples], dtype=str),
    }


def task_label_weights(tasks: np.ndarray, answers: np.ndarray) -> np.ndarray:
    counts = Counter(zip(tasks.tolist(), answers.tolist()))
    task_counts = Counter(tasks.tolist())
    label_counts_by_task = Counter(task for task, _ in counts)
    weights = np.asarray(
        [
            task_counts[task]
            / (label_counts_by_task[task] * counts[(task, answer)])
            for task, answer in zip(tasks, answers)
        ],
        dtype=np.float32,
    )
    weights /= weights.mean()
    return weights


def fixed_task_metrics(
    references: np.ndarray,
    predictions: np.ndarray,
    tasks: np.ndarray,
    task_labels: dict[str, list[str]],
) -> list[dict[str, Any]]:
    rows = []
    for task in task_labels:
        indices = np.flatnonzero(tasks == task)
        if not len(indices):
            continue
        labels = task_labels[task]
        y_true = references[indices]
        y_pred = predictions[indices]
        rows.append(
            {
                "task": task,
                "n": int(len(indices)),
                "reference_class_count": int(len(set(y_true))),
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "macro_f1_fixed_labels": float(
                    f1_score(
                        y_true,
                        y_pred,
                        labels=labels,
                        average="macro",
                        zero_division=0,
                    )
                ),
                "balanced_accuracy_fixed_labels": float(
                    recall_score(
                        y_true,
                        y_pred,
                        labels=labels,
                        average="macro",
                        zero_division=0,
                    )
                ),
                "invalid_answer_rate": float(
                    np.mean([prediction not in labels for prediction in y_pred])
                ),
            }
        )
    return rows


def mean_task_macro_f1(
    references: np.ndarray,
    predictions: np.ndarray,
    tasks: np.ndarray,
    task_labels: dict[str, list[str]],
    selected_tasks: list[str],
) -> float:
    values = []
    for task in selected_tasks:
        indices = np.flatnonzero(tasks == task)
        if not len(indices):
            raise RuntimeError(f"No records for aggregate task: {task}")
        values.append(
            f1_score(
                references[indices],
                predictions[indices],
                labels=task_labels[task],
                average="macro",
                zero_division=0,
            )
        )
    return float(np.mean(values))


def predict_probabilities(
    model: nn.Module,
    arrays: dict[str, Any],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    probabilities = []
    with torch.inference_mode():
        for start in range(0, len(arrays["target"]), batch_size):
            stop = start + batch_size
            image = torch.from_numpy(arrays["image"][start:stop]).to(device)
            question = torch.from_numpy(arrays["question"][start:stop]).to(device)
            logits = model(image, question)
            probabilities.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.concatenate(probabilities, axis=0).astype(np.float32)


def train_epochs(
    model: nn.Module,
    arrays: dict[str, Any],
    weights: np.ndarray,
    epochs: int,
    config: dict[str, Any],
    seed: int,
    device: torch.device,
) -> list[float]:
    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    batch_size = int(config["batch_size"])
    rng = np.random.default_rng(seed)
    losses = []
    model.train()
    for _ in range(epochs):
        permutation = rng.permutation(len(arrays["target"]))
        epoch_loss = 0.0
        epoch_weight = 0.0
        for start in range(0, len(permutation), batch_size):
            indices = permutation[start : start + batch_size]
            image = torch.from_numpy(arrays["image"][indices]).to(device)
            question = torch.from_numpy(arrays["question"][indices]).to(device)
            target = torch.from_numpy(arrays["target"][indices]).to(device)
            sample_weight = torch.from_numpy(weights[indices]).to(device)
            optimiser.zero_grad(set_to_none=True)
            logits = model(image, question)
            loss_values = F.cross_entropy(logits, target, reduction="none")
            loss = torch.sum(loss_values * sample_weight) / torch.sum(sample_weight)
            loss.backward()
            optimiser.step()
            epoch_loss += float(torch.sum(loss_values * sample_weight).item())
            epoch_weight += float(torch.sum(sample_weight).item())
        losses.append(epoch_loss / epoch_weight)
    return losses


def select_epoch(
    condition: str,
    train_arrays: dict[str, Any],
    validation_arrays: dict[str, Any],
    answer_vocab: list[str],
    task_labels: dict[str, list[str]],
    config: dict[str, Any],
    seed: int,
    device: torch.device,
) -> tuple[int, float, list[dict[str, Any]]]:
    set_seed(seed)
    model = SharedVQAHead(
        image_dim=train_arrays["image"].shape[1],
        question_dim=train_arrays["question"].shape[1],
        projection_dim=int(config["projection_dim"]),
        answer_count=len(answer_vocab),
        dropout=float(config["dropout"]),
        condition=condition,
    ).to(device)
    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    train_weights = task_label_weights(train_arrays["task"], train_arrays["answer"])
    batch_size = int(config["batch_size"])
    rng = np.random.default_rng(seed)
    best_epoch = 0
    best_score = -math.inf
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    log_rows = []
    for epoch in range(1, int(config["maximum_epochs"]) + 1):
        model.train()
        permutation = rng.permutation(len(train_arrays["target"]))
        epoch_loss = 0.0
        epoch_weight = 0.0
        for start in range(0, len(permutation), batch_size):
            indices = permutation[start : start + batch_size]
            image = torch.from_numpy(train_arrays["image"][indices]).to(device)
            question = torch.from_numpy(train_arrays["question"][indices]).to(device)
            target = torch.from_numpy(train_arrays["target"][indices]).to(device)
            sample_weight = torch.from_numpy(train_weights[indices]).to(device)
            optimiser.zero_grad(set_to_none=True)
            logits = model(image, question)
            loss_values = F.cross_entropy(logits, target, reduction="none")
            loss = torch.sum(loss_values * sample_weight) / torch.sum(sample_weight)
            loss.backward()
            optimiser.step()
            epoch_loss += float(torch.sum(loss_values * sample_weight).item())
            epoch_weight += float(torch.sum(sample_weight).item())

        probabilities = predict_probabilities(
            model, validation_arrays, device, batch_size
        )
        predictions = np.asarray(
            [answer_vocab[index] for index in probabilities.argmax(axis=1)], dtype=str
        )
        score = mean_task_macro_f1(
            validation_arrays["answer"],
            predictions,
            validation_arrays["task"],
            task_labels,
            config["primary_tasks"],
        )
        log_rows.append(
            {
                "condition": condition,
                "seed": seed,
                "epoch": epoch,
                "training_loss": epoch_loss / epoch_weight,
                "validation_primary_mean_macro_f1": score,
            }
        )
        if score > best_score + 1e-8:
            best_score = score
            best_epoch = epoch
            best_state = deepcopy(
                {name: value.detach().cpu() for name, value in model.state_dict().items()}
            )
            stale = 0
        else:
            stale += 1
        if stale >= int(config["early_stopping_patience"]):
            break
    if best_state is None or best_epoch < 1:
        raise RuntimeError("Validation failed to select an epoch")
    return best_epoch, best_score, log_rows


def refit_model(
    condition: str,
    arrays: dict[str, Any],
    answer_vocab: list[str],
    config: dict[str, Any],
    seed: int,
    epochs: int,
    device: torch.device,
) -> tuple[SharedVQAHead, list[float]]:
    set_seed(seed)
    model = SharedVQAHead(
        image_dim=arrays["image"].shape[1],
        question_dim=arrays["question"].shape[1],
        projection_dim=int(config["projection_dim"]),
        answer_count=len(answer_vocab),
        dropout=float(config["dropout"]),
        condition=condition,
    ).to(device)
    weights = task_label_weights(arrays["task"], arrays["answer"])
    losses = train_epochs(model, arrays, weights, epochs, config, seed, device)
    return model, losses


def aggregate_rows(
    condition: str,
    wording: str,
    task_metrics: list[dict[str, Any]],
    primary_tasks: list[str],
    perceptual_tasks: list[str],
) -> list[dict[str, Any]]:
    by_task = {row["task"]: row for row in task_metrics}
    groups = {
        "all_applicable_tasks": list(by_task),
        "primary_discriminative_tasks": primary_tasks,
        "perceptual_tasks": perceptual_tasks,
    }
    rows = []
    for group, tasks in groups.items():
        values = [by_task[task]["macro_f1_fixed_labels"] for task in tasks]
        invalid = [by_task[task]["invalid_answer_rate"] for task in tasks]
        rows.append(
            {
                "condition": condition,
                "wording": wording,
                "task_group": group,
                "task_count": len(tasks),
                "mean_macro_f1_fixed_labels": float(np.mean(values)),
                "mean_invalid_answer_rate": float(np.mean(invalid)),
            }
        )
    return rows


def site_stratified_bootstrap_indices(
    videos: np.ndarray, sites: np.ndarray, replicates: int, seed: int
) -> list[np.ndarray]:
    video_site: dict[str, str] = {}
    for video, site in zip(videos, sites):
        existing = video_site.setdefault(str(video), str(site))
        if existing != site:
            raise RuntimeError(f"Video assigned to multiple sites: {video}")
    videos_by_site: dict[str, list[str]] = defaultdict(list)
    for video, site in video_site.items():
        videos_by_site[site].append(video)
    indices_by_video = {
        video: np.flatnonzero(videos == video) for video in video_site
    }
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(replicates):
        selected: list[str] = []
        for site in sorted(videos_by_site):
            candidates = sorted(videos_by_site[site])
            selected.extend(
                rng.choice(candidates, size=len(candidates), replace=True).tolist()
            )
        samples.append(np.concatenate([indices_by_video[video] for video in selected]))
    return samples


def paired_bootstrap_delta(
    references: np.ndarray,
    predictions_a: np.ndarray,
    predictions_b: np.ndarray,
    tasks: np.ndarray,
    videos: np.ndarray,
    sites: np.ndarray,
    task_labels: dict[str, list[str]],
    selected_tasks: list[str],
    replicates: int,
    seed: int,
) -> dict[str, float]:
    point_a = mean_task_macro_f1(
        references, predictions_a, tasks, task_labels, selected_tasks
    )
    point_b = mean_task_macro_f1(
        references, predictions_b, tasks, task_labels, selected_tasks
    )
    deltas = []
    for indices in site_stratified_bootstrap_indices(
        videos, sites, replicates, seed
    ):
        score_a = mean_task_macro_f1(
            references[indices],
            predictions_a[indices],
            tasks[indices],
            task_labels,
            selected_tasks,
        )
        score_b = mean_task_macro_f1(
            references[indices],
            predictions_b[indices],
            tasks[indices],
            task_labels,
            selected_tasks,
        )
        deltas.append(score_a - score_b)
    return {
        "estimate_a": point_a,
        "estimate_b": point_b,
        "delta": point_a - point_b,
        "ci_low": float(np.percentile(deltas, 2.5)),
        "ci_high": float(np.percentile(deltas, 97.5)),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    started = time.time()
    for required in (DATA, DATA_SUMMARY, FEATURES, CONFIG_PATH):
        if not required.is_file():
            raise FileNotFoundError(required)
    RESULTS.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    data_summary = json.loads(DATA_SUMMARY.read_text(encoding="utf-8"))
    rows = load_jsonl(DATA)
    answer_vocab = data_summary["global_answer_vocabulary"]
    task_labels = data_summary["task_labels"]
    task_order = data_summary["task_order"]
    if any("human" in key.lower() for row in rows for key in row):
        raise RuntimeError("Human-reference fields must not enter the VQA training data")

    feature_data = np.load(FEATURES)
    record_ids = feature_data["record_ids"].astype(str)
    image_features = feature_data["image_features"].astype(np.float32)
    question_texts = feature_data["question_texts"].astype(str)
    question_features = feature_data["question_features"].astype(np.float32)
    image_map = dict(zip(record_ids, image_features))
    question_map = dict(zip(question_texts, question_features))
    if {row["record_id"] for row in rows} != set(image_map):
        raise RuntimeError("Shared VQA data and image feature IDs differ")

    train_core = make_arrays(
        examples_for_training(rows, {"train_core"}), image_map, question_map
    )
    validation_examples = [
        (row, row["question_canonical"])
        for row in rows
        if row["validation_role"] == "validation"
    ]
    validation = make_arrays(validation_examples, image_map, question_map)
    full_train = make_arrays(
        examples_for_training(rows, {"train_core", "validation"}),
        image_map,
        question_map,
    )
    test_arrays = {
        wording: make_arrays(
            examples_for_evaluation(rows, wording, task_order), image_map, question_map
        )
        for wording in ("canonical", "heldout_paraphrase", "wrong_question")
    }
    if not np.array_equal(
        test_arrays["canonical"]["qa_id"],
        test_arrays["heldout_paraphrase"]["qa_id"],
    ):
        raise RuntimeError("Canonical and held-out paraphrase test records differ")
    if not np.array_equal(
        test_arrays["canonical"]["qa_id"], test_arrays["wrong_question"]["qa_id"]
    ):
        raise RuntimeError("Canonical and wrong-question test records differ")

    train_videos = set(full_train["video"])
    test_videos = set(test_arrays["canonical"]["video"])
    if train_videos & test_videos:
        raise RuntimeError("Sequence leakage detected before training")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Shared VQA heads are expected to train on CUDA")

    training_log: list[dict[str, Any]] = []
    seed_summary: list[dict[str, Any]] = []
    seed_probabilities: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    for condition in config["conditions"]:
        for seed in config["seeds"]:
            best_epoch, validation_score, log_rows = select_epoch(
                condition,
                train_core,
                validation,
                answer_vocab,
                task_labels,
                config,
                int(seed),
                device,
            )
            training_log.extend(log_rows)
            model, refit_losses = refit_model(
                condition,
                full_train,
                answer_vocab,
                config,
                int(seed),
                best_epoch,
                device,
            )
            checkpoint = CHECKPOINTS / f"{condition}_seed{seed}.pt"
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "condition": condition,
                    "seed": int(seed),
                    "epochs": best_epoch,
                    "answer_vocabulary": answer_vocab,
                    "config": config,
                },
                checkpoint,
            )
            seed_summary.append(
                {
                    "condition": condition,
                    "seed": int(seed),
                    "selected_epoch": best_epoch,
                    "validation_primary_mean_macro_f1": validation_score,
                    "refit_final_training_loss": refit_losses[-1],
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": sha256(checkpoint),
                }
            )
            for wording in ("canonical", "heldout_paraphrase"):
                seed_probabilities[(condition, wording)].append(
                    predict_probabilities(
                        model,
                        test_arrays[wording],
                        device,
                        int(config["batch_size"]),
                    )
                )
            if condition == "image_question":
                seed_probabilities[(condition, "wrong_question")].append(
                    predict_probabilities(
                        model,
                        test_arrays["wrong_question"],
                        device,
                        int(config["batch_size"]),
                    )
                )
            del model
            torch.cuda.empty_cache()

    canonical = test_arrays["canonical"]
    train_base_rows = [row for row in rows if row["source_split"] == "train"]
    majority_by_task = {}
    for task in task_order:
        counts = Counter(
            row["answer"] for row in train_base_rows if row["task"] == task
        )
        majority_by_task[task] = sorted(
            counts.items(), key=lambda item: (-item[1], item[0])
        )[0][0]

    predictions_by_key: dict[tuple[str, str], np.ndarray] = {}
    probabilities_by_key: dict[tuple[str, str], np.ndarray] = {}
    for key, probability_list in seed_probabilities.items():
        probabilities = np.mean(np.stack(probability_list, axis=0), axis=0)
        probabilities_by_key[key] = probabilities
        predictions_by_key[key] = np.asarray(
            [answer_vocab[index] for index in probabilities.argmax(axis=1)], dtype=str
        )

    frequency_predictions = np.asarray(
        [majority_by_task[task] for task in canonical["task"]], dtype=str
    )
    predictions_by_key[("answer_frequency", "canonical")] = frequency_predictions
    predictions_by_key[("answer_frequency", "heldout_paraphrase")] = frequency_predictions.copy()

    metric_rows: list[dict[str, Any]] = []
    aggregate_metric_rows: list[dict[str, Any]] = []
    for (condition, wording), predictions in sorted(predictions_by_key.items()):
        arrays = test_arrays[
            "canonical" if condition == "answer_frequency" else wording
        ]
        task_metric_rows = fixed_task_metrics(
            arrays["answer"], predictions, arrays["task"], task_labels
        )
        for row in task_metric_rows:
            metric_rows.append(
                {"condition": condition, "wording": wording, **row}
            )
        aggregate_metric_rows.extend(
            aggregate_rows(
                condition,
                wording,
                task_metric_rows,
                config["primary_tasks"],
                config["perceptual_tasks"],
            )
        )

    comparison_specs = [
        (
            "image_question_vs_question_only",
            ("image_question", "canonical"),
            ("question_only", "canonical"),
        ),
        (
            "image_question_vs_image_only",
            ("image_question", "canonical"),
            ("image_only", "canonical"),
        ),
        (
            "image_question_vs_answer_frequency",
            ("image_question", "canonical"),
            ("answer_frequency", "canonical"),
        ),
        (
            "heldout_paraphrase_vs_canonical",
            ("image_question", "heldout_paraphrase"),
            ("image_question", "canonical"),
        ),
        (
            "wrong_question_vs_canonical",
            ("image_question", "wrong_question"),
            ("image_question", "canonical"),
        ),
    ]
    bootstrap_rows = []
    for comparison_index, (name, key_a, key_b) in enumerate(comparison_specs):
        for group_name, selected_tasks in (
            ("primary_discriminative_tasks", config["primary_tasks"]),
            ("perceptual_tasks", config["perceptual_tasks"]),
        ):
            result = paired_bootstrap_delta(
                canonical["answer"],
                predictions_by_key[key_a],
                predictions_by_key[key_b],
                canonical["task"],
                canonical["video"],
                canonical["site"],
                task_labels,
                selected_tasks,
                int(config["bootstrap_replicates"]),
                int(config["bootstrap_seed"]) + comparison_index,
            )
            bootstrap_rows.append(
                {
                    "comparison": name,
                    "task_group": group_name,
                    "condition_a": key_a[0],
                    "wording_a": key_a[1],
                    "condition_b": key_b[0],
                    "wording_b": key_b[1],
                    "bootstrap_unit": "source_sequence_stratified_by_site",
                    "bootstrap_replicates": int(config["bootstrap_replicates"]),
                    **result,
                }
            )

    prediction_rows = []
    for key, predictions in sorted(predictions_by_key.items()):
        condition, wording = key
        arrays = test_arrays[
            "canonical" if condition == "answer_frequency" else wording
        ]
        probability_array = probabilities_by_key.get(key)
        for index, prediction in enumerate(predictions):
            prediction_rows.append(
                {
                    "condition": condition,
                    "wording": wording,
                    "qa_id": arrays["qa_id"][index],
                    "record_id": arrays["record_id"][index],
                    "source_sequence_id": arrays["video"][index],
                    "site": arrays["site"][index],
                    "task": arrays["task"][index],
                    "question": arrays["question_text"][index],
                    "reference": arrays["answer"][index],
                    "prediction": prediction,
                    "prediction_confidence": (
                        ""
                        if probability_array is None
                        else float(np.max(probability_array[index]))
                    ),
                    "valid_for_task": prediction in task_labels[arrays["task"][index]],
                }
            )

    write_csv(RESULTS / "training_log.csv", training_log)
    write_csv(RESULTS / "seed_summary.csv", seed_summary)
    write_csv(RESULTS / "task_metrics.csv", metric_rows)
    write_csv(RESULTS / "aggregate_metrics.csv", aggregate_metric_rows)
    write_csv(RESULTS / "paired_sequence_bootstrap_deltas.csv", bootstrap_rows)
    write_csv(RESULTS / "predictions.csv", prediction_rows)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "protocol": config["protocol"],
        "encoder": config["model_id"],
        "encoder_frozen": True,
        "human_labels_used_for_training_or_selection": False,
        "task_specific_answer_filtering": False,
        "global_answer_vocabulary": answer_vocab,
        "train_source_sequence_count": len(train_videos),
        "test_source_sequence_count": len(test_videos),
        "validation_sequences": data_summary["validation_sequences"],
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0),
        "packages": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "config": config,
        "inputs": {
            "runner": Path(__file__).relative_to(WORKSPACE).as_posix(),
            "runner_sha256": sha256(Path(__file__).resolve()),
            "data": DATA.relative_to(WORKSPACE).as_posix(),
            "data_sha256": sha256(DATA),
            "data_summary": DATA_SUMMARY.relative_to(WORKSPACE).as_posix(),
            "data_summary_sha256": sha256(DATA_SUMMARY),
            "features": FEATURES.relative_to(WORKSPACE).as_posix(),
            "features_sha256": sha256(FEATURES),
            "config": CONFIG_PATH.relative_to(WORKSPACE).as_posix(),
            "config_sha256": sha256(CONFIG_PATH),
        },
        "outputs": {
            path.name: sha256(path)
            for path in sorted(RESULTS.glob("*.csv"))
        },
        "elapsed_seconds": time.time() - started,
    }
    (RESULTS / "run.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
