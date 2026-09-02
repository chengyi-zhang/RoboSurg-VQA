"""Render the README schematic from the checked-in dataset manifest.

Run with matplotlib 3.10: python docs/assets/render_overview.py
The figure contains no source images or new experimental results.
"""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[2]
ASSETS = Path(__file__).resolve().parent
INK = "#203239"
MUTED = "#62737A"
TEAL = "#137C82"
BLUE = "#43698C"
ROSE = "#94526D"
LINE = "#D6E1E4"


def read_counts():
    summary = json.loads((ROOT / "data/vqa_summary.json").read_text())
    with (ROOT / "data/manifest.jsonl").open() as handle:
        manifest = [json.loads(line) for line in handle if line.strip()]
    with (ROOT / "data/human_reference/sample_manifest.csv").open(newline="") as handle:
        audit_count = sum(1 for _ in csv.DictReader(handle))
    sources = Counter(row["dataset"] for row in manifest)
    if sum(sources.values()) != summary["unique_frames"]:
        raise ValueError("Manifest and dataset summary disagree")
    return sources, [summary["unique_frames"], summary["records"], audit_count,
                     summary["train_sequence_count"] + summary["test_sequence_count"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, help="Optional PDF export for local review")
    args = parser.parse_args()
    sources, counts = read_counts()
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 12, "text.color": INK, "svg.fonttype": "none",
        "pdf.fonttype": 42, "svg.hashsalt": "robosurg-overview",
    })
    fig, ax = plt.subplots(figsize=(12, 6.4), dpi=300)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set(xlim=(0, 1200), ylim=(640, 0))
    ax.axis("off")
    fig.patch.set_facecolor("white")

    def text(x, y, value, size=12, colour=INK, weight="normal", **kwargs):
        return ax.text(x, y, value, fontsize=size, color=colour, fontweight=weight,
                       va="center", **kwargs)

    def box(x, y, w, h, fill="white", edge=LINE, radius=5):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                     boxstyle=f"round,pad=0,rounding_size={radius}",
                     facecolor=fill, edgecolor=edge, linewidth=0.8))

    def arrow(start, end, colour=MUTED):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>",
                     mutation_scale=13, linewidth=1.2, color=colour))

    text(32, 36, "FROM SEGMENTATION TO SURGICAL QUESTIONS", 11, TEAL, "bold")
    text(32, 80, "One benchmark. Every answer has an origin.", 24, INK, "bold")
    ax.plot([32, 1168], [116, 116], color=LINE, linewidth=0.9)

    for x, number, title, colour in [
        (32, "01", "Reuse source data", TEAL),
        (442, "02", "Record answer provenance", BLUE),
        (850, "03", "Evaluate image + question", ROSE),
    ]:
        text(x, 150, number, 12, colour, "bold")
        text(x + 34, 150, title, 13, INK, "bold")

    # Sources stay schematic: the repository does not redistribute surgical media.
    for y, source in [(185, "EndoVis 2017"), (258, "EndoVis 2018")]:
        box(32, y, 340, 58, "#F1F7F7")
        text(51, y + 29, source, 14, INK, "bold")
        text(354, y + 29, f"{sources[source]:,} frames", 12, TEAL, ha="right")
    box(32, 341, 151, 70)
    box(199, 341, 173, 70)
    text(107, 365, "RGB", 16, TEAL, "bold", ha="center")
    text(107, 392, "surgical frames", 11, MUTED, ha="center")
    text(285, 365, "MASKS", 16, TEAL, "bold", ha="center")
    text(285, 392, "official annotations", 11, MUTED, ha="center")
    text(32, 446, "Original sequence splits retained", 12, MUTED)
    arrow((385, 305), (427, 305))

    for y, heading, detail, colour, fill in [
        (185, "Source context", "Dataset metadata", TEAL, "#F1F7F7"),
        (266, "Anatomy + spatial targets", "Official masks", BLUE, "#F1F5F9"),
        (347, "Visual attributes", "Generated candidate labels", ROSE, "#F9F2F5"),
    ]:
        box(442, y, 337, 67, fill)
        ax.add_patch(Rectangle((442, y + 10), 3, 47, color=colour, linewidth=0))
        text(458, y + 23, heading, 13, colour, "bold")
        text(458, y + 48, detail, 11, MUTED)
    text(442, 446, "Separate human audit: 250 frames", 12, MUTED)
    arrow((792, 305), (835, 305))

    box(850, 185, 137, 55, "#F1F7F7")
    box(1002, 185, 166, 55, "#F9F2F5")
    text(918, 212, "RGB frame", 12, TEAL, "bold", ha="center")
    text(1085, 212, "Question text", 12, ROSE, "bold", ha="center")
    arrow((918, 243), (918, 273), TEAL)
    arrow((1085, 243), (1085, 273), ROSE)
    box(850, 278, 318, 63)
    text(1009, 300, "Frozen BiomedCLIP encoders", 12, INK, "bold", ha="center")
    text(1009, 325, "Image and text features", 11, MUTED, ha="center")
    arrow((1009, 344), (1009, 368))
    box(850, 373, 318, 44, "#F9F2F5", ROSE)
    text(1009, 395, "Shared closed-set answer head", 12, ROSE, "bold", ha="center")
    text(850, 446, "Unimodal controls + unseen phrasing", 11, MUTED)

    ax.axhspan(490, 640, facecolor="#F1F6F7", zorder=-1)
    for i, (value, label) in enumerate(zip(counts, [
        "SURGICAL FRAMES", "QUESTION-ANSWER RECORDS", "AUDITED FRAMES", "SOURCE SEQUENCES"
    ])):
        centre = 150 + i * 300
        if i:
            ax.plot([i * 300, i * 300], [521, 599], color=LINE, linewidth=0.9)
        text(centre, 544, f"{value:,}", 28, TEAL if i < 2 else INK, "bold", ha="center")
        text(centre, 588, label, 10, MUTED, "bold", ha="center")

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = fig.bbox
    for label in ax.texts:
        extent = label.get_window_extent(renderer)
        if extent.x0 < bounds.x0 or extent.x1 > bounds.x1 or extent.y0 < bounds.y0 or extent.y1 > bounds.y1:
            raise ValueError(f"Text outside figure: {label.get_text()}")
    fig.savefig(ASSETS / "overview.png", dpi=300, facecolor="white")
    fig.savefig(ASSETS / "overview.svg", facecolor="white", metadata={"Date": None})
    if args.pdf:
        fig.savefig(args.pdf, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
