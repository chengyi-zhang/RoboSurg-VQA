#!/usr/bin/env python3
"""Extract frozen BioMedCLIP image and question embeddings."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import open_clip
from huggingface_hub import snapshot_download
from open_clip.tokenizer import HFTokenizer


WORKSPACE = Path(__file__).resolve().parents[1]
DATA = WORKSPACE / "data" / "vqa_records.jsonl"
PROTOCOL = WORKSPACE / "configs" / "shared_vqa.json"
OUTPUT_DIR = WORKSPACE / "features"
OUTPUT = OUTPUT_DIR / "biomedclip_frozen_features.npz"
METADATA = OUTPUT_DIR / "biomedclip_feature_extraction.json"
MODEL_CACHE = WORKSPACE / ".cache" / "biomedclip"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest().upper()


class FrameDataset(Dataset):
    def __init__(
        self, record_ids: list[str], image_paths: list[str], transform: Any
    ) -> None:
        self.record_ids = record_ids
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.record_ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image_array = np.load(WORKSPACE / self.image_paths[index], allow_pickle=False)
        image = Image.fromarray(image_array.astype(np.uint8), mode="RGB")
        return self.transform(image), index


def package_version(name: str) -> str:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return "unknown"


def cache_fingerprints(cache: Path) -> list[dict[str, Any]]:
    suffixes = {".bin", ".pt", ".pth", ".safetensors", ".json", ".txt"}
    files = [
        path
        for path in cache.rglob("*")
        if path.is_file() and path.suffix.lower() in suffixes
    ]
    rows = []
    for path in sorted(files):
        rows.append(
            {
                "path": str(path.relative_to(cache)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return rows


def load_encoder(
    config: dict[str, Any], cache: Path, *, offline: bool = False
) -> tuple[Any, Any, Any]:
    model_path = Path(snapshot_download(
        config["model_id"],
        revision=config["model_revision"],
        allow_patterns=["open_clip_config.json", "open_clip_pytorch_model.bin"],
        cache_dir=str(cache),
        local_files_only=offline,
    ))
    text_path = Path(snapshot_download(
        config["text_model_id"],
        revision=config["text_model_revision"],
        allow_patterns=["config.json", "tokenizer_config.json", "vocab.txt"],
        cache_dir=str(cache / "hub"),
        local_files_only=offline,
    ))
    model_config = json.loads((model_path / "open_clip_config.json").read_text())
    text_config = dict(model_config["model_cfg"]["text_cfg"])
    # The CLIP checkpoint contains the text weights; only its config and tokenizer are external.
    text_config.update(
        hf_model_name=str(text_path),
        hf_tokenizer_name=str(text_path),
        hf_model_pretrained=False,
    )
    model, _, preprocess = open_clip.create_model_and_transforms(
        f"local-dir:{model_path}",
        text_cfg=text_config,
        cache_dir=str(cache),
    )
    tokenizer = HFTokenizer(str(text_path), context_length=text_config["context_length"])
    return model, preprocess, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-batch-size", type=int, default=16)
    parser.add_argument("--text-batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--offline", action="store_true", help="Use cached weights only.")
    parser.add_argument(
        "--refresh-text-only",
        action="store_true",
        help="Reuse the existing image embeddings and recompute question embeddings.",
    )
    args = parser.parse_args()

    for required in (DATA, PROTOCOL):
        if not required.is_file():
            raise FileNotFoundError(required)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_CACHE.mkdir(parents=True, exist_ok=True)
    if OUTPUT.exists() and not args.force and not args.refresh_text_only:
        print(f"Frozen feature cache already exists: {OUTPUT}")
        return
    if args.refresh_text_only and not OUTPUT.exists():
        raise FileNotFoundError(
            "--refresh-text-only requires an existing frozen feature cache"
        )

    config = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    model_id = str(config["model_id"])
    rows = load_jsonl(DATA)

    image_path_by_id: dict[str, str] = {}
    questions: set[str] = set()
    for row in rows:
        existing = image_path_by_id.setdefault(row["record_id"], row["image_path"])
        if existing != row["image_path"]:
            raise RuntimeError(f"Conflicting image paths for {row['record_id']}")
        questions.add(row["question_canonical"])
        questions.update(row["question_train_paraphrases"])
        questions.add(row["question_heldout_paraphrase"])

    record_ids = sorted(image_path_by_id)
    image_paths = [image_path_by_id[record_id] for record_id in record_ids]
    question_texts = sorted(questions)
    if len(record_ids) != 5632:
        raise RuntimeError(f"Expected 5,632 unique images, found {len(record_ids)}")

    os.environ.setdefault("HF_HOME", str(MODEL_CACHE / "huggingface"))
    os.environ.setdefault("TORCH_HOME", str(MODEL_CACHE / "torch"))
    torch.manual_seed(20260831)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(20260831)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("The frozen BioMedCLIP extraction is expected to use CUDA")

    started = time.time()
    model, preprocess, tokenizer = load_encoder(
        config, MODEL_CACHE / "huggingface", offline=args.offline
    )
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    if args.refresh_text_only:
        with np.load(OUTPUT) as existing:
            cached_ids = existing["record_ids"].astype(str)
            image_features = existing["image_features"].astype(np.float32)
        if not np.array_equal(cached_ids, np.asarray(record_ids, dtype=str)):
            raise RuntimeError("Cached image feature IDs differ from current records")
    else:
        frame_dataset = FrameDataset(record_ids, image_paths, preprocess)
        frame_loader = DataLoader(
            frame_dataset,
            batch_size=args.image_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
        )

        image_features: np.ndarray | None = None
        with torch.inference_mode():
            for images, indices in tqdm(frame_loader, desc="BioMedCLIP images"):
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    embeddings = model.encode_image(images.to(device, non_blocking=True))
                embeddings = torch.nn.functional.normalize(embeddings.float(), dim=-1)
                batch = embeddings.cpu().numpy().astype(np.float32)
                if image_features is None:
                    image_features = np.empty(
                        (len(record_ids), batch.shape[1]), dtype=np.float32
                    )
                image_features[indices.numpy()] = batch
        if image_features is None:
            raise RuntimeError("No image features were extracted")

    text_batches: list[np.ndarray] = []
    with torch.inference_mode():
        for start in tqdm(
            range(0, len(question_texts), args.text_batch_size),
            desc="BioMedCLIP questions",
        ):
            texts = question_texts[start : start + args.text_batch_size]
            tokens = tokenizer(texts).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                embeddings = model.encode_text(tokens)
            embeddings = torch.nn.functional.normalize(embeddings.float(), dim=-1)
            text_batches.append(embeddings.cpu().numpy().astype(np.float32))
    question_features = np.concatenate(text_batches, axis=0)

    if not np.isfinite(image_features).all() or not np.isfinite(question_features).all():
        raise RuntimeError("Non-finite BioMedCLIP features detected")
    image_norms = np.linalg.norm(image_features, axis=1)
    question_norms = np.linalg.norm(question_features, axis=1)
    if not np.allclose(image_norms, 1.0, atol=1e-4):
        raise RuntimeError("Image features are not L2 normalised")
    if not np.allclose(question_norms, 1.0, atol=1e-4):
        raise RuntimeError("Question features are not L2 normalised")

    np.savez_compressed(
        OUTPUT,
        record_ids=np.asarray(record_ids, dtype=np.str_),
        image_features=image_features,
        question_texts=np.asarray(question_texts, dtype=np.str_),
        question_features=question_features,
    )

    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_id": model_id,
        "model_revision": config["model_revision"],
        "text_model_id": config["text_model_id"],
        "text_model_revision": config["text_model_revision"],
        "encoder_frozen": True,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0),
        "image_count": len(record_ids),
        "question_count": len(question_texts),
        "embedding_dimension": int(image_features.shape[1]),
        "image_batch_size": args.image_batch_size,
        "text_batch_size": args.text_batch_size,
        "image_features_reused": args.refresh_text_only,
        "preprocessing": repr(preprocess),
        "data": DATA.relative_to(WORKSPACE).as_posix(),
        "data_sha256": sha256(DATA),
        "protocol": PROTOCOL.relative_to(WORKSPACE).as_posix(),
        "protocol_sha256": sha256(PROTOCOL),
        "output": OUTPUT.relative_to(WORKSPACE).as_posix(),
        "output_sha256": sha256(OUTPUT),
        "cache_fingerprints": cache_fingerprints(MODEL_CACHE),
        "packages": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "open_clip_torch": package_version("open_clip_torch"),
            "transformers": package_version("transformers"),
            "huggingface_hub": package_version("huggingface_hub"),
            "numpy": np.__version__,
        },
        "elapsed_seconds": time.time() - started,
    }
    METADATA.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
