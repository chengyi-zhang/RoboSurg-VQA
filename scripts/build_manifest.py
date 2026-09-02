#!/usr/bin/env python3
"""Build a portable EndoVis17/18 manifest from paired NumPy arrays."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path, PurePosixPath

import numpy as np


SITE_DATASET = {"site1": "EndoVis 2017", "site2": "EndoVis 2018"}
STANDARD_COUNTS = {
    ("site1", "train"): 1800,
    ("site1", "test"): 600,
    ("site2", "train"): 2235,
    ("site2", "test"): 997,
}


def frame_parts(identifier: str) -> tuple[str, int]:
    match = re.fullmatch(r"(video\d+)frame(\d+)", identifier)
    if not match:
        raise ValueError(f"Unsupported frame filename stem: {identifier}")
    return match.group(1), int(match.group(2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/manifest.jsonl"),
    )
    parser.add_argument(
        "--path-prefix",
        default="external_data",
        help="Portable path prefix stored in the manifest.",
    )
    parser.add_argument("--verify-arrays", action="store_true")
    parser.add_argument("--expect-standard-counts", action="store_true")
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    rows = []
    observed_counts = {}
    for site, dataset in SITE_DATASET.items():
        for split in ("train", "test"):
            image_dir = data_root / site / split / "image"
            mask_dir = data_root / site / split / "mask"
            image_paths = sorted(image_dir.glob("*.npy"))
            mask_by_stem = {path.stem: path for path in mask_dir.glob("*.npy")}
            observed_counts[(site, split)] = len(image_paths)
            if len(mask_by_stem) != len(image_paths):
                raise RuntimeError(
                    f"Pair count mismatch for {site}/{split}: "
                    f"images={len(image_paths)} masks={len(mask_by_stem)}"
                )
            for image_path in image_paths:
                mask_path = mask_by_stem.get(image_path.stem)
                if mask_path is None:
                    raise FileNotFoundError(f"Missing mask for {image_path}")
                video_id, frame_index = frame_parts(image_path.stem)
                image_shape = mask_shape = image_dtype = mask_dtype = None
                if args.verify_arrays:
                    image = np.load(image_path, mmap_mode="r")
                    mask = np.load(mask_path, mmap_mode="r")
                    if image.shape[:2] != mask.shape[:2]:
                        raise RuntimeError(
                            f"Shape mismatch for {image_path.stem}: {image.shape} vs {mask.shape}"
                        )
                    image_shape, mask_shape = list(image.shape), list(mask.shape)
                    image_dtype, mask_dtype = str(image.dtype), str(mask.dtype)
                rows.append(
                    {
                        "record_id": f"{site}/{split}/{image_path.stem}",
                        "id": image_path.stem,
                        "dataset": dataset,
                        "site": site,
                        "split": split,
                        "video_id": video_id,
                        "video_key": f"{site}/{video_id}",
                        "frame_index": frame_index,
                        "image_path": str(
                            PurePosixPath(args.path_prefix)
                            / PurePosixPath(image_path.relative_to(data_root).as_posix())
                        ),
                        "image_bytes": image_path.stat().st_size,
                        "mask_path": str(
                            PurePosixPath(args.path_prefix)
                            / PurePosixPath(mask_path.relative_to(data_root).as_posix())
                        ),
                        "mask_bytes": mask_path.stat().st_size,
                        "image_shape": image_shape,
                        "mask_shape": mask_shape,
                        "image_dtype": image_dtype,
                        "mask_dtype": mask_dtype,
                    }
                )

    if args.expect_standard_counts and observed_counts != STANDARD_COUNTS:
        raise RuntimeError(
            f"Unexpected standard-corpus counts: {observed_counts}; expected {STANDARD_COUNTS}"
        )
    record_ids = [row["record_id"] for row in rows]
    if len(record_ids) != len(set(record_ids)):
        raise RuntimeError("Duplicate record IDs")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "records": len(rows),
                "counts": {f"{site}/{split}": count for (site, split), count in observed_counts.items()},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
