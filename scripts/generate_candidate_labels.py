#!/usr/bin/env python3
"""Generate and validate the machine candidate-label layer."""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "data" / "manifest.jsonl"
DEFAULT_CONFIG = ROOT / "configs" / "candidate_generation.json"
DEFAULT_PROMPT = ROOT / "configs" / "candidate_generation_prompt.txt"
DEFAULT_OUTPUT = ROOT / "outputs" / "candidate_annotations.jsonl"

QUESTIONS = {
    "q1": "What is the primary surgical context in this frame?",
    "q2": "Which organ or anatomical region does this image belong to?",
    "q3": "What is the imaging modality/view?",
    "q4": "Is there bleeding, smoke, or occlusion visible in the image?",
    "q5": "How is the visual quality of the current scene?",
    "q6": "Is there specular reflection/glare?",
    "q7": "Is the image contrast normal?",
    "q8": "Is the centre of the frame occluded by instruments?",
    "q9": "Which regions contain smoke?",
}
Q1_LABELS = {
    "nephrectomy (robot-assisted)",
    "renal/kidney-focused procedure",
    "other abdominal procedure",
}
Q2_LABELS = {
    "small intestine",
    "kidney (parenchyma)",
    "covered kidney (fat/fascia-covered kidney)",
    "other abdominal soft tissue (peritoneum/mesentery/fat)",
}
Q5_LABELS = {"clear", "blurry", "reflective", "mixed", "Unknown"}
YES_NO = {"yes", "no", "Unknown"}

Q1_ALIASES = {
    "robot-assisted nephrectomy": "nephrectomy (robot-assisted)",
    "nephrectomy": "nephrectomy (robot-assisted)",
    "urologic surgery (unspecified)": "renal/kidney-focused procedure",
    "renal surgery": "renal/kidney-focused procedure",
    "kidney-focused procedure": "renal/kidney-focused procedure",
    "renal/kidney focused procedure": "renal/kidney-focused procedure",
}
Q2_ALIASES = {
    "kidney parenchyma": "kidney (parenchyma)",
    "kidney": "kidney (parenchyma)",
    "covered kidney": "covered kidney (fat/fascia-covered kidney)",
    "other/unspecified tissue": "other abdominal soft tissue (peritoneum/mesentery/fat)",
    "other tissue": "other abdominal soft tissue (peritoneum/mesentery/fat)",
    "peritoneum": "other abdominal soft tissue (peritoneum/mesentery/fat)",
    "mesentery": "other abdominal soft tissue (peritoneum/mesentery/fat)",
    "fat": "other abdominal soft tissue (peritoneum/mesentery/fat)",
}
PALETTE = np.asarray(
    [
        [20, 20, 20], [230, 25, 75], [60, 180, 75], [0, 130, 200],
        [245, 130, 48], [145, 30, 180], [70, 240, 240], [240, 50, 230],
        [210, 245, 60], [250, 190, 190], [0, 128, 128], [230, 190, 255],
        [170, 110, 40], [255, 250, 200], [128, 0, 0], [170, 255, 195],
        [128, 128, 0], [255, 215, 180], [0, 0, 128], [128, 128, 128],
    ],
    dtype=np.uint8,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_rgb(path: Path) -> Image.Image:
    array = np.load(path)
    if array.ndim == 4 or (array.ndim == 3 and array.shape[-1] not in (1, 3, 4)):
        array = array[len(array) // 2]
    if array.ndim == 2:
        array = array[..., None]
    if array.ndim != 3 or array.shape[-1] not in (1, 3, 4):
        raise ValueError(f"Unsupported RGB array shape: {array.shape}")
    if np.issubdtype(array.dtype, np.floating):
        finite = array[np.isfinite(array)]
        if not finite.size or finite.max() == finite.min():
            array = np.zeros_like(array, dtype=np.uint8)
        else:
            array = ((array - finite.min()) / (finite.max() - finite.min()) * 255).clip(0, 255)
    array = array.astype(np.uint8)
    if array.shape[-1] == 4:
        array = array[..., :3]
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    return Image.fromarray(array)


def load_mask(path: Path, size: tuple[int, int]) -> Image.Image:
    array = np.load(path)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim != 2:
        raise ValueError(f"Unsupported mask array shape: {array.shape}")
    array = np.maximum(np.rint(array).astype(np.int64), 0)
    if array.shape[::-1] != size:
        mask = Image.fromarray(array.astype(np.int32), mode="I")
        array = np.asarray(mask.resize(size, resample=Image.Resampling.NEAREST))
    return Image.fromarray(PALETTE[array % len(PALETTE)])


def data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def confidence(value: Any) -> float:
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def normalise_label(
    value: Any, labels: set[str], aliases: dict[str, str] | None = None
) -> str | None:
    text = str(value).strip()
    canonical = {label.lower(): label for label in labels}
    if text.lower() in canonical:
        return canonical[text.lower()]
    if aliases and text.lower() in aliases:
        return aliases[text.lower()]
    return None


def normalise_yes_no(value: Any) -> str:
    text = str(value).strip()
    if text.lower() in {"yes", "no"}:
        return text.lower()
    if text.lower() == "unknown":
        return "Unknown"
    return "Unknown"


def validate_answers(response: dict[str, Any]) -> dict[str, Any]:
    source = response.get("answers", response)
    if not isinstance(source, dict) or any(key not in source for key in QUESTIONS):
        raise ValueError("Response does not contain all q1-q9 answers")

    output: dict[str, Any] = {}
    q2 = source["q2"]
    q2_confidence = confidence(q2.get("confidence"))
    q2_value = normalise_label(q2.get("value"), Q2_LABELS, Q2_ALIASES)
    if q2_value is None:
        q2_value = "other abdominal soft tissue (peritoneum/mesentery/fat)"
        q2_confidence = min(q2_confidence, 0.6)
    output["q2"] = {"value": q2_value, "confidence": q2_confidence}

    q1 = source["q1"]
    q1_confidence = confidence(q1.get("confidence"))
    q1_value = normalise_label(q1.get("value"), Q1_LABELS, Q1_ALIASES)
    if q1_value is None:
        q1_value = (
            "renal/kidney-focused procedure"
            if q2_value
            in {"kidney (parenchyma)", "covered kidney (fat/fascia-covered kidney)"}
            else "other abdominal procedure"
        )
        q1_confidence = min(q1_confidence, 0.6)
    if q2_value in {"kidney (parenchyma)", "covered kidney (fat/fascia-covered kidney)"}:
        if q1_value == "other abdominal procedure":
            q1_value = "renal/kidney-focused procedure"
            q1_confidence = max(q1_confidence, 0.7)
    elif q1_value == "nephrectomy (robot-assisted)":
        q1_value = "other abdominal procedure"
        q1_confidence = min(q1_confidence, 0.6)
    output["q1"] = {"value": q1_value, "confidence": q1_confidence}

    q3 = source["q3"]
    q3_value = str(q3.get("value", "Unknown")).strip() or "Unknown"
    output["q3"] = {"value": q3_value, "confidence": confidence(q3.get("confidence"))}

    q4 = source["q4"]
    q4_value = q4.get("value")
    if not isinstance(q4_value, dict):
        raise ValueError("q4.value must contain bleeding, smoke, and occlusion")
    output["q4"] = {
        "value": {name: normalise_yes_no(q4_value.get(name)) for name in ("bleeding", "smoke", "occlusion")},
        "confidence": confidence(q4.get("confidence")),
    }

    q5 = source["q5"]
    output["q5"] = {
        "value": normalise_label(q5.get("value"), Q5_LABELS) or "Unknown",
        "confidence": confidence(q5.get("confidence")),
    }
    for key in ("q6", "q7", "q8"):
        item = source[key]
        output[key] = {
            "value": normalise_yes_no(item.get("value")),
            "confidence": confidence(item.get("confidence")),
        }

    q9 = source["q9"]
    q9_value = str(q9.get("value", "Unknown")).strip() or "Unknown"
    if output["q4"]["value"]["smoke"] == "no":
        q9_value = "none"
    output["q9"] = {"value": q9_value, "confidence": confidence(q9.get("confidence"))}
    return {key: output[key] for key in QUESTIONS}


def build_user_content(
    rgb: Image.Image, mask: Image.Image, detail: str, instruction: str
) -> list[dict[str, Any]]:
    question_text = "\n".join(f"{key}: {text}" for key, text in QUESTIONS.items())
    return [
        {"type": "text", "text": "Original image"},
        {"type": "image_url", "image_url": {"url": data_url(rgb), "detail": detail}},
        {"type": "text", "text": "GT segmentation mask (pseudocolour)"},
        {"type": "image_url", "image_url": {"url": data_url(mask), "detail": detail}},
        {"type": "text", "text": instruction},
        {"type": "text", "text": f"Here are the enabled questions:\n{question_text}"},
    ]


def request_annotation(client: Any, payload: dict[str, Any], retries: int) -> str:
    delay = 0.5
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(**payload)
            content = response.choices[0].message.content
            if not isinstance(content, str):
                raise RuntimeError("Model returned no text content")
            return content
        except Exception:
            if attempt + 1 == retries:
                raise
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("Candidate-label request failed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--retries", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY before running candidate generation")
    try:
        from openai import OpenAI
    except ImportError as error:
        raise RuntimeError("Install the optional 'openai' package") from error

    config = json.loads(args.config.read_text(encoding="utf-8"))
    prompt = args.prompt.read_text(encoding="utf-8").strip()
    rows = read_jsonl(args.manifest)
    if args.limit is not None:
        rows = rows[: args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    if args.output.exists():
        completed = {row["record_id"] for row in read_jsonl(args.output)}

    client = OpenAI()
    with args.output.open("a", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            if row["record_id"] in completed:
                continue
            rgb = load_rgb(ROOT / row["image_path"])
            mask = load_mask(ROOT / row["mask_path"], rgb.size)
            payload = {
                "model": config["model"],
                "messages": [
                    {"role": "system", "content": prompt},
                    {
                        "role": "user",
                        "content": build_user_content(
                            rgb,
                            mask,
                            config["image_detail"],
                            config["user_text"][0],
                        ),
                    },
                ],
                "temperature": config["temperature"],
                "response_format": config["response_format"],
            }
            raw = request_annotation(client, payload, max(1, args.retries))
            record = {
                "record_id": row["record_id"],
                "model": config["model"],
                "answers": validate_answers(json.loads(raw)),
            }
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            print(row["record_id"])


if __name__ == "__main__":
    main()
