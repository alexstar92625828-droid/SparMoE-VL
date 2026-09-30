"""Render Figure 3 from a validated Dense-only analysis file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from .merge import validate_merged
from .protocol import (
    CAPACITY_LEVELS,
    NUM_LAYERS,
    OUTPUT_ROOT,
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=OUTPUT_ROOT / "analysis.json")
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=OUTPUT_ROOT / "dense_ffn_capacity_requirement",
    )
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args(argv)


def load_plot_data(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"missing merged Figure-3 analysis: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_merged(payload, path)
    layers = payload.get("layers")
    if not isinstance(layers, Mapping):
        raise ValueError(f"{path}: missing layer results")
    if sorted(int(layer) for layer in layers) != list(range(1, NUM_LAYERS + 1)):
        raise ValueError(f"{path}: expected layers 1 through {NUM_LAYERS}")
    levels = np.asarray(CAPACITY_LEVELS, dtype=float)
    proportions = np.asarray(
        [
            layers[str(layer)]["required_capacity_proportions"]
            for layer in range(1, NUM_LAYERS + 1)
        ],
        dtype=float,
    )
    means = np.asarray(
        [layers[str(layer)]["layer_mean"] for layer in range(1, NUM_LAYERS + 1)],
        dtype=float,
    )
    if proportions.shape != (NUM_LAYERS, len(CAPACITY_LEVELS)):
        raise ValueError(f"{path}: invalid heatmap shape {proportions.shape}")
    return levels, proportions, means


def journal_blue_colormap() -> Any:
    import matplotlib.colors as colors

    return colors.LinearSegmentedColormap.from_list(
        "journal_capacity_blue",
        (
            "#F7FAFD",
            "#E4EFF8",
            "#C5DCEC",
            "#8DB9D8",
            "#4E8DBD",
            "#1F5F94",
            "#083B66",
        ),
    )


def draw(
    output_stem: Path,
    levels: np.ndarray,
    proportions: np.ndarray,
    means: np.ndarray,
) -> tuple[Path, Path]:
    try:
        import matplotlib.patheffects as path_effects
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("install the analysis dependencies to render Figure 3") from error
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure = plt.figure(figsize=(4.75, 2.48))
    layout = figure.add_gridspec(
        1,
        2,
        width_ratios=(1.0, 0.035),
        left=0.115,
        right=0.925,
        bottom=0.22,
        top=0.94,
        wspace=0.045,
    )
    axis = figure.add_subplot(layout[0, 0])
    colorbar_axis = figure.add_subplot(layout[0, 1])
    image = axis.imshow(
        proportions.T,
        origin="lower",
        interpolation="nearest",
        aspect="auto",
        extent=(0.5, 24.5, 0.25, 1.05),
        cmap=journal_blue_colormap(),
        vmin=0.0,
        vmax=1.0,
        rasterized=True,
    )
    layers = np.arange(1, NUM_LAYERS + 1)
    line = axis.plot(
        layers,
        means,
        color="#163F5C",
        linewidth=1.35,
        marker="o",
        markersize=2.45,
        markerfacecolor="#F8FBFD",
        markeredgecolor="#163F5C",
        markeredgewidth=0.65,
        zorder=4,
        label="Layer mean",
    )[0]
    line.set_path_effects(
        [
            path_effects.Stroke(linewidth=2.5, foreground="white", alpha=0.88),
            path_effects.Normal(),
        ]
    )
    axis.set_xlim(0.5, 24.5)
    axis.set_ylim(0.25, 1.05)
    axis.set_xticks([1, 4, 8, 12, 16, 20, 24])
    axis.set_yticks(levels)
    axis.set_yticklabels([f"{value:.1f}" for value in levels])
    axis.set_xlabel("Transformer layer", fontsize=8.6, labelpad=3)
    axis.set_ylabel("Required FFN capacity", fontsize=8.6, labelpad=4)
    axis.tick_params(axis="both", labelsize=7.2, length=2.2, width=0.65, pad=2)
    axis.set_xticks(np.arange(0.5, 25.0, 1.0), minor=True)
    axis.set_yticks(np.arange(0.25, 1.06, 0.1), minor=True)
    axis.grid(which="minor", color="white", linewidth=0.28, alpha=0.55)
    axis.tick_params(which="minor", bottom=False, left=False)
    for spine in axis.spines.values():
        spine.set_color("#52616F")
        spine.set_linewidth(0.7)
    legend = axis.legend(
        loc="upper left",
        bbox_to_anchor=(0.012, 0.988),
        borderaxespad=0,
        frameon=False,
        fontsize=7.0,
        handlelength=1.7,
        handletextpad=0.45,
    )
    for handle in legend.legend_handles:
        handle.set_path_effects(
            [
                path_effects.Stroke(linewidth=2.5, foreground="white", alpha=0.9),
                path_effects.Normal(),
            ]
        )
    colorbar = figure.colorbar(image, cax=colorbar_axis)
    colorbar.set_ticks(np.linspace(0.0, 1.0, 6))
    colorbar.ax.tick_params(labelsize=6.8, length=2.0, width=0.6, pad=2)
    colorbar.set_label("Token fraction", fontsize=7.8, labelpad=4)
    colorbar.outline.set_linewidth(0.65)
    colorbar.outline.set_edgecolor("#52616F")
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = output_stem.with_suffix(".pdf")
    png_path = output_stem.with_suffix(".png")
    figure.savefig(pdf_path, bbox_inches="tight", pad_inches=0.025)
    figure.savefig(png_path, dpi=600, bbox_inches="tight", pad_inches=0.025)
    plt.close(figure)
    return pdf_path, png_path


def run(args: argparse.Namespace) -> dict[str, Any]:
    levels, proportions, means = load_plot_data(args.input)
    result = {
        "input": str(args.input.resolve()),
        "layers": NUM_LAYERS,
        "capacity_levels": len(levels),
        "heatmap_shape": list(proportions.shape),
    }
    if args.check_only:
        print(json.dumps(result, indent=2))
        return result
    pdf_path, png_path = draw(args.output_stem, levels, proportions, means)
    result.update(pdf=str(pdf_path), png=str(png_path))
    print(json.dumps(result, indent=2))
    return result


def main(argv: Optional[Sequence[str]] = None) -> None:
    run(parse_args(argv))
