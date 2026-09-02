#!/usr/bin/env python3
"""Extract one fixed frozen RGB representation and deterministic mask features."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18


WORKSPACE = Path(__file__).resolve().parents[1]
MANIFEST = WORKSPACE / "data" / "manifest.jsonl"
OUTPUT_DIR = WORKSPACE / "features"
MODEL_CACHE = WORKSPACE / ".cache" / "torch" / "hub"


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


MASK_FEATURE_NAMES = (
    [f"occupancy_8x8_r{row}_c{col}" for row in range(8) for col in range(8)]
    + [
        "instrument_area_fraction",
        "central_area_instrument_fraction",
        "centroid_x",
        "centroid_y",
        "bbox_x_min",
        "bbox_y_min",
        "bbox_x_max",
        "bbox_y_max",
        "bbox_width",
        "bbox_height",
        "has_instrument",
        "class_1_fraction",
        "class_2_fraction",
        "class_3_fraction",
        "quadrant_top_left_fraction",
        "quadrant_top_right_fraction",
        "quadrant_bottom_left_fraction",
        "quadrant_bottom_right_fraction",
    ]
)


def mask_features(mask: np.ndarray) -> np.ndarray:
    binary = np.asarray(mask) > 0
    height, width = binary.shape[:2]
    occupancy = []
    row_edges = np.linspace(0, height, 9, dtype=int)
    col_edges = np.linspace(0, width, 9, dtype=int)
    for row in range(8):
        for col in range(8):
            cell = binary[row_edges[row] : row_edges[row + 1], col_edges[col] : col_edges[col + 1]]
            occupancy.append(float(cell.mean()))

    y0, y1 = height // 4, (3 * height) // 4
    x0, x1 = width // 4, (3 * width) // 4
    central_fraction = float(binary[y0:y1, x0:x1].mean())
    area_fraction = float(binary.mean())
    ys, xs = np.nonzero(binary)
    if len(xs):
        centroid_x = float(xs.mean() / max(width - 1, 1))
        centroid_y = float(ys.mean() / max(height - 1, 1))
        bbox_x_min = float(xs.min() / max(width - 1, 1))
        bbox_y_min = float(ys.min() / max(height - 1, 1))
        bbox_x_max = float(xs.max() / max(width - 1, 1))
        bbox_y_max = float(ys.max() / max(height - 1, 1))
        bbox_width = bbox_x_max - bbox_x_min
        bbox_height = bbox_y_max - bbox_y_min
        has_instrument = 1.0
    else:
        centroid_x = centroid_y = 0.5
        bbox_x_min = bbox_y_min = bbox_x_max = bbox_y_max = 0.0
        bbox_width = bbox_height = has_instrument = 0.0

    class_fractions = [float((mask == label).mean()) for label in (1, 2, 3)]
    mid_y, mid_x = height // 2, width // 2
    quadrants = [
        binary[:mid_y, :mid_x],
        binary[:mid_y, mid_x:],
        binary[mid_y:, :mid_x],
        binary[mid_y:, mid_x:],
    ]
    quadrant_fractions = [float(quadrant.mean()) for quadrant in quadrants]
    values = occupancy + [
        area_fraction,
        central_fraction,
        centroid_x,
        centroid_y,
        bbox_x_min,
        bbox_y_min,
        bbox_x_max,
        bbox_y_max,
        bbox_width,
        bbox_height,
        has_instrument,
        *class_fractions,
        *quadrant_fractions,
    ]
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (len(MASK_FEATURE_NAMES),):
        raise RuntimeError(f"Mask feature shape mismatch: {result.shape}")
    return result


class SurgicalFrameDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], transform: Any) -> None:
        self.rows = rows
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        row = self.rows[index]
        image_array = np.load(WORKSPACE / row["image_path"], allow_pickle=False)
        mask_array = np.load(WORKSPACE / row["mask_path"], allow_pickle=False)
        image = Image.fromarray(image_array.astype(np.uint8), mode="RGB")
        rgb_tensor = self.transform(image)
        mask_tensor = torch.from_numpy(mask_features(mask_array))
        return rgb_tensor, mask_tensor, index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "taskwise_resnet18_features.npz"
    config_path = OUTPUT_DIR / "taskwise_feature_extraction.json"
    if output_path.exists() and not args.force:
        print(f"Feature cache already exists: {output_path}")
        return

    rows = load_jsonl(MANIFEST)
    if len(rows) != 5632:
        raise RuntimeError(f"Expected 5,632 frozen records, found {len(rows)}")

    torch.set_num_threads(max(1, min(20, torch.get_num_threads())))
    torch.hub.set_dir(str(MODEL_CACHE))
    torch.manual_seed(20260821)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    dataset = SurgicalFrameDataset(rows, weights.transforms())
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    rgb_features = np.empty((len(rows), 512), dtype=np.float32)
    segmentation_features = np.empty(
        (len(rows), len(MASK_FEATURE_NAMES)), dtype=np.float32
    )

    started = time.time()
    processed = 0
    with torch.inference_mode():
        for images, masks, indices in loader:
            embeddings = model(images.to(device, non_blocking=True)).cpu().numpy().astype(np.float32)
            index_array = indices.numpy()
            rgb_features[index_array] = embeddings
            segmentation_features[index_array] = masks.numpy().astype(np.float32)
            processed += len(index_array)
            if processed % 512 < len(index_array) or processed == len(rows):
                elapsed = time.time() - started
                print(f"processed={processed}/{len(rows)} elapsed_seconds={elapsed:.1f}", flush=True)

    record_ids = np.asarray([row["record_id"] for row in rows], dtype=np.str_)
    splits = np.asarray([row["split"] for row in rows], dtype=np.str_)
    datasets = np.asarray([row["dataset"] for row in rows], dtype=np.str_)
    np.savez_compressed(
        output_path,
        record_ids=record_ids,
        splits=splits,
        datasets=datasets,
        rgb_features=rgb_features,
        mask_features=segmentation_features,
        mask_feature_names=np.asarray(MASK_FEATURE_NAMES, dtype=np.str_),
    )

    checkpoint_path = MODEL_CACHE / "checkpoints" / "resnet18-f37072fd.pth"
    config = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "manifest": MANIFEST.relative_to(WORKSPACE).as_posix(),
        "manifest_sha256": sha256(MANIFEST),
        "records": len(rows),
        "encoder": "torchvision ResNet-18",
        "weights": "ResNet18_Weights.IMAGENET1K_V1",
        "weights_path": checkpoint_path.name,
        "weights_sha256": sha256(checkpoint_path),
        "encoder_frozen": True,
        "preprocessing": repr(weights.transforms()),
        "rgb_feature_dimensions": rgb_features.shape[1],
        "mask_feature_dimensions": segmentation_features.shape[1],
        "mask_feature_names": MASK_FEATURE_NAMES,
        "device": str(device),
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "elapsed_seconds": time.time() - started,
        "output": output_path.relative_to(WORKSPACE).as_posix(),
        "output_sha256": sha256(output_path),
    }
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(config, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
