"""Render expert-usage and adaptive-capacity panels from validated analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from .analysis import validate_analysis
from .protocol import NUM_EXPERTS, NUM_LAYERS, OUTPUT_ROOT


EXPORT_WIDTH_INCHES = 7.19
EXPORT_HEIGHT_INCHES = 2.87
TITLE_FONTSIZE = 12.5
AXIS_FONTSIZE = 11.0
DETAIL_FONTSIZE = 8.5


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=OUTPUT_ROOT / "analysis.json")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args(argv)


def load_plot_data(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing layer-wise analysis: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    validate_analysis(payload, path)
    return {
        "usage": np.asarray(payload["proportions_layer_by_expert"], dtype=float),
        "capacities": np.asarray(payload["capacities_layer_by_expert"], dtype=float),
        "activated": np.asarray(payload["activated_capacity_by_layer"], dtype=float),
        "target": float(payload["target_ratio"]),
    }


def configure_matplotlib() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def export_bbox(figure: Any) -> Any:
    from matplotlib.transforms import Bbox

    figure.canvas.draw()
    tight = figure.get_tightbbox(figure.canvas.get_renderer())
    center_x = (tight.x0 + tight.x1) / 2
    center_y = (tight.y0 + tight.y1) / 2
    return Bbox.from_bounds(
        center_x - EXPORT_WIDTH_INCHES / 2,
        center_y - EXPORT_HEIGHT_INCHES / 2,
        EXPORT_WIDTH_INCHES,
        EXPORT_HEIGHT_INCHES,
    )


def draw_expert_usage(usage: np.ndarray, output_dir: Path) -> tuple[Path, Path]:
    import matplotlib.pyplot as plt

    configure_matplotlib()
    figure, axis = plt.subplots(figsize=(7.4, 3.80), dpi=300)
    image = axis.imshow(
        usage.T,
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        aspect="auto",
    )
    axis.set_title(
        "Layer-wise Expert Usage of Patch Tokens",
        fontsize=TITLE_FONTSIZE,
        fontweight="bold",
        pad=8,
    )
    axis.set_xlabel("Layers", fontsize=AXIS_FONTSIZE)
    axis.set_ylabel("Experts", fontsize=AXIS_FONTSIZE)
    axis.set_xticks([0, 4, 8, 12, 16, 20, 23])
    axis.set_xticklabels(
        ["0", "4", "8", "12", "16", "20", "23"],
        fontsize=DETAIL_FONTSIZE,
    )
    axis.set_yticks(np.arange(NUM_EXPERTS))
    axis.set_yticklabels(
        [str(expert) for expert in range(1, NUM_EXPERTS + 1)],
        fontsize=DETAIL_FONTSIZE,
    )
    axis.tick_params(length=2.2, width=0.7)
    for spine in axis.spines.values():
        spine.set_linewidth(0.75)
    colorbar = figure.colorbar(image, ax=axis, fraction=0.026, pad=0.01)
    colorbar.ax.tick_params(labelsize=DETAIL_FONTSIZE, length=2.2, width=0.7)
    colorbar.set_label("Token fraction", fontsize=AXIS_FONTSIZE)
    figure.subplots_adjust(left=0.078, right=0.94, top=0.81, bottom=0.25)
    bounding_box = export_bbox(figure)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "expert_usage_distribution.pdf"
    png_path = output_dir / "expert_usage_distribution.png"
    figure.savefig(pdf_path, bbox_inches=bounding_box, pad_inches=0)
    figure.savefig(png_path, dpi=600, bbox_inches=bounding_box, pad_inches=0)
    plt.close(figure)
    return pdf_path, png_path


def draw_capacity_allocation(
    capacities: np.ndarray,
    activated: np.ndarray,
    target: float,
    output_dir: Path,
) -> tuple[Path, Path]:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    configure_matplotlib()
    figure, axis = plt.subplots(figsize=(7.4, 3.80), dpi=300)
    layers = np.arange(NUM_LAYERS)
    e1, e2, e3, e4 = capacities.T
    colors = {
        "envelope": "#E4F2F8",
        "saving": "#A9D8ED",
        "internal": "#56B4E9",
        "lower_edge": "#8CCCE9",
        "budget": "#0072B2",
        "activated": "#004B73",
        "baseline": "#77848C",
        "grid": "#E7EBED",
    }
    axis.fill_between(
        layers,
        e1,
        e4,
        color=colors["envelope"],
        alpha=0.95,
        linewidth=0,
        zorder=1,
    )
    axis.fill_between(
        layers,
        activated,
        e4,
        color=colors["saving"],
        alpha=0.48,
        linewidth=0,
        zorder=2,
    )
    axis.plot(
        layers,
        e2,
        color=colors["internal"],
        linewidth=0.8,
        alpha=0.75,
        zorder=3,
    )
    axis.plot(
        layers,
        e3,
        color=colors["internal"],
        linewidth=0.8,
        alpha=0.75,
        zorder=3,
    )
    axis.plot(layers, e1, color=colors["lower_edge"], linewidth=0.9, zorder=3)
    axis.plot(
        layers,
        e4,
        color=colors["budget"],
        linewidth=1.9,
        marker="o",
        markersize=4.1,
        markerfacecolor="white",
        markeredgecolor=colors["budget"],
        markeredgewidth=1.05,
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=5,
    )
    axis.plot(
        layers,
        activated,
        color=colors["activated"],
        linewidth=1.75,
        marker="o",
        markersize=2.7,
        markerfacecolor=colors["activated"],
        markeredgewidth=0,
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=6,
    )
    axis.axhline(
        target,
        color=colors["baseline"],
        linewidth=1.05,
        linestyle=(0, (5, 3)),
        zorder=4,
    )
    axis.text(
        0.35,
        target + 0.025,
        rf"target $p={target:.1f}$",
        color=colors["baseline"],
        fontsize=DETAIL_FONTSIZE,
        ha="left",
        va="bottom",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.7},
    )
    axis.set_title(
        "Layer-wise Adaptive FFN Capacity Allocation",
        fontsize=TITLE_FONTSIZE,
        fontweight="bold",
        pad=8,
    )
    axis.set_xlabel("Layers", fontsize=AXIS_FONTSIZE)
    axis.set_ylabel("FFN retention ratio", fontsize=AXIS_FONTSIZE)
    axis.set_xlim(-0.5, NUM_LAYERS - 0.5)
    axis.set_ylim(0.0, 1.0)
    axis.set_xticks([0, 4, 8, 12, 16, 20, 23])
    axis.set_xticklabels(
        ["0", "4", "8", "12", "16", "20", "23"],
        fontsize=DETAIL_FONTSIZE,
    )
    y_ticks = np.arange(0.0, 1.01, 0.2)
    axis.set_yticks(y_ticks)
    axis.set_yticklabels(
        [f"{value:.1f}" for value in y_ticks],
        fontsize=DETAIL_FONTSIZE,
    )
    axis.grid(axis="y", color=colors["grid"], linewidth=0.65, zorder=0)
    axis.tick_params(length=2.2, width=0.7)
    for spine in axis.spines.values():
        spine.set_linewidth(0.75)
    handles = [
        Line2D(
            [0],
            [0],
            color=colors["budget"],
            linewidth=1.9,
            marker="o",
            markersize=3.8,
            markerfacecolor="white",
            markeredgecolor=colors["budget"],
            label="Layer budget",
        ),
        Patch(
            facecolor=colors["envelope"],
            edgecolor=colors["lower_edge"],
            linewidth=0.7,
            label="Available capacity span",
        ),
        Line2D(
            [0],
            [0],
            color=colors["activated"],
            linewidth=1.75,
            marker="o",
            markersize=2.7,
            label="Activated capacity",
        ),
    ]
    axis.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.008, 0.025),
        ncol=3,
        frameon=False,
        fontsize=DETAIL_FONTSIZE,
        columnspacing=1.45,
        handlelength=1.9,
        handletextpad=0.5,
    )
    figure.subplots_adjust(left=0.092, right=0.985, top=0.81, bottom=0.25)
    bounding_box = export_bbox(figure)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "adaptive_capacity_allocation.pdf"
    png_path = output_dir / "adaptive_capacity_allocation.png"
    figure.savefig(pdf_path, bbox_inches=bounding_box, pad_inches=0)
    figure.savefig(png_path, dpi=600, bbox_inches=bounding_box, pad_inches=0)
    plt.close(figure)
    return pdf_path, png_path


def run(args: argparse.Namespace) -> dict[str, Any]:
    data = load_plot_data(args.input)
    result: dict[str, Any] = {
        "input": str(args.input.resolve()),
        "usage_shape": list(data["usage"].shape),
        "capacity_shape": list(data["capacities"].shape),
    }
    if args.check_only:
        print(json.dumps(result, indent=2))
        return result
    usage_pdf, usage_png = draw_expert_usage(data["usage"], args.output_dir)
    capacity_pdf, capacity_png = draw_capacity_allocation(
        data["capacities"],
        data["activated"],
        data["target"],
        args.output_dir,
    )
    result["generated"] = [
        str(usage_pdf),
        str(usage_png),
        str(capacity_pdf),
        str(capacity_png),
    ]
    print(json.dumps(result, indent=2))
    return result


def main(argv: Optional[Sequence[str]] = None) -> None:
    run(parse_args(argv))
