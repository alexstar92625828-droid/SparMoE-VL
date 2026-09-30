"""Render the validated layer-wise local/global representation analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from .analysis import expected_bytes, validate_analysis
from .protocol import (
    HISTOGRAM_BINS,
    HISTOGRAM_RANGE,
    NUM_LAYERS,
    OUTPUT_ROOT,
    PATCHES_PER_IMAGE,
    STUDY_NAME,
)


RASTER_DPI = 600
BACKGROUND = "#FFFFFF"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=OUTPUT_ROOT / "analysis.json")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--allow-partial-smoke", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args(argv)


def load_plot_data(path: Path, *, allow_partial_smoke: bool = False) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing representation analysis: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    validate_analysis(payload, path, allow_partial_smoke=allow_partial_smoke)
    return payload


def load_patch_cache(payload: Mapping[str, Any], analysis_path: Path) -> np.memmap:
    spec = payload["cache"]["patch_cosine"]
    path = analysis_path.parent / str(spec["file"])
    shape = tuple(int(value) for value in spec["shape"])
    if shape[1:] != (NUM_LAYERS, PATCHES_PER_IMAGE):
        raise ValueError(f"{analysis_path}: invalid patch cache geometry")
    if not path.is_file() or path.stat().st_size != expected_bytes(shape):
        raise ValueError(f"{path}: patch cache is missing or has an unexpected size")
    return np.memmap(path, dtype=np.float32, mode="r", shape=shape)


def layer_histograms(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if values.ndim != 3 or values.shape[1:] != (NUM_LAYERS, PATCHES_PER_IMAGE):
        raise ValueError("patch values must have shape [images, 24, 256]")
    if not np.isfinite(values).all() or np.any(values < -1.0) or np.any(values > 1.0):
        raise ValueError("patch cosine cache contains invalid values")
    lower, upper = HISTOGRAM_RANGE
    if np.any(values < lower) or np.any(values > upper):
        raise ValueError("patch cosine values fall outside the registered plot range")
    edges = np.linspace(lower, upper, HISTOGRAM_BINS + 1)
    fractions = np.zeros((HISTOGRAM_BINS, NUM_LAYERS), dtype=np.float64)
    samples_per_layer = values.shape[0] * values.shape[2]
    for layer in range(NUM_LAYERS):
        layer_values = np.asarray(values[:, layer, :]).reshape(-1)
        counts, _ = np.histogram(layer_values, bins=edges)
        fractions[:, layer] = 100.0 * counts / samples_per_layer
    if not np.allclose(fractions.sum(axis=0), 100.0, atol=1e-10):
        raise RuntimeError("histogram bins do not conserve patch tokens")
    return edges, fractions


def validate_cache_statistics(payload: Mapping[str, Any], values: np.ndarray) -> None:
    distribution = payload["patch_token_cosine"]
    for layer in range(NUM_LAYERS):
        layer_values = np.asarray(values[:, layer, :]).reshape(-1)
        quantiles = np.quantile(layer_values, (0.10, 0.25, 0.50, 0.75, 0.90))
        expected = np.asarray(
            [
                distribution["q10"][layer],
                distribution["q25"][layer],
                distribution["median"][layer],
                distribution["q75"][layer],
                distribution["q90"][layer],
            ]
        )
        if not np.allclose(quantiles, expected, rtol=0.0, atol=1e-7):
            raise ValueError(f"patch cache disagrees with layer {layer + 1} quantiles")


def similarity_colormap() -> Any:
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        "patch_density_blue",
        ["#F7FAFC", "#E2EFF7", "#B8D8EA", "#75ADD0", "#2F7FAF", "#0A416F"],
        N=256,
    )


def configure_matplotlib() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
            "font.size": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.edgecolor": "#27333D",
            "axes.labelcolor": "#202A33",
            "xtick.color": "#27333D",
            "ytick.color": "#27333D",
            "figure.facecolor": BACKGROUND,
            "axes.facecolor": BACKGROUND,
            "savefig.facecolor": BACKGROUND,
        }
    )


def draw(payload: Mapping[str, Any], edges: np.ndarray, fractions: np.ndarray) -> Any:
    import matplotlib.patheffects as path_effects
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    configure_matplotlib()
    layers = np.asarray(payload["layers_one_based"], dtype=float)
    cka = np.asarray(payload["cls_linear_cka"], dtype=float)
    cls_cosine = np.asarray(payload["cls_token_cosine"], dtype=float)
    patch_median = np.asarray(payload["patch_token_cosine"]["median"], dtype=float)
    nonzero = fractions[fractions > 0]
    if nonzero.size == 0:
        raise ValueError("patch histogram contains no nonzero bins")
    color_min = max(1e-3, float(np.quantile(nonzero, 0.08)))
    color_max = float(fractions.max())
    if color_max <= color_min:
        color_min = max(1e-6, color_max / 10.0)

    figure = plt.figure(figsize=(8.2, 4.1), dpi=300, facecolor=BACKGROUND)
    grid = figure.add_gridspec(
        1,
        2,
        width_ratios=(1.0, 0.022),
        left=0.088,
        right=0.955,
        bottom=0.20,
        top=0.975,
        wspace=0.035,
    )
    axis = figure.add_subplot(grid[0, 0])
    colorbar_axis = figure.add_subplot(grid[0, 1])
    x_edges = np.arange(0.5, NUM_LAYERS + 1.5)
    mesh = axis.pcolormesh(
        x_edges,
        edges,
        np.ma.masked_less_equal(fractions, 0.0),
        cmap=similarity_colormap(),
        norm=LogNorm(vmin=color_min, vmax=color_max),
        shading="flat",
        rasterized=True,
        zorder=1,
    )
    (median_line,) = axis.plot(
        layers,
        patch_median,
        color="white",
        linewidth=1.75,
        marker="s",
        markersize=3.2,
        markerfacecolor="white",
        markeredgecolor="#185C89",
        markeredgewidth=0.85,
        label="Patch-token median",
        zorder=5,
    )
    median_line.set_path_effects(
        [path_effects.Stroke(linewidth=3.0, foreground="#185C89"), path_effects.Normal()]
    )
    (cka_line,) = axis.plot(
        layers,
        cka,
        color="#073F68",
        linewidth=2.0,
        marker="o",
        markersize=4.0,
        markerfacecolor="white",
        markeredgecolor="#073F68",
        markeredgewidth=1.0,
        label="CLS linear CKA",
        zorder=7,
    )
    (cls_line,) = axis.plot(
        layers,
        cls_cosine,
        color="#2C91C7",
        linewidth=1.7,
        linestyle=(0, (2.2, 1.8)),
        marker="^",
        markersize=3.5,
        markerfacecolor="white",
        markeredgecolor="#2C91C7",
        markeredgewidth=0.9,
        label="CLS-token cosine",
        zorder=6,
    )
    axis.set_xlim(0.5, NUM_LAYERS + 0.5)
    axis.set_ylim(HISTOGRAM_RANGE[0], 1.005)
    axis.set_xlabel("Transformer layer", fontsize=11.0, labelpad=5)
    axis.set_ylabel("Similarity to Dense CLIP", fontsize=11.0, labelpad=6)
    axis.set_xticks([1, 4, 8, 12, 16, 20, 24])
    axis.set_yticks([-0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axis.tick_params(labelsize=8.5, length=2.4, width=0.7, pad=2.5)
    axis.grid(axis="x", color="white", linewidth=0.42, alpha=0.62, zorder=3)
    axis.legend(
        handles=[cka_line, cls_line, median_line],
        loc="lower left",
        bbox_to_anchor=(0.012, 0.022),
        frameon=False,
        fontsize=8.5,
        ncol=3,
        handlelength=2.25,
        columnspacing=1.0,
    )
    colorbar = figure.colorbar(mesh, cax=colorbar_axis)
    colorbar.set_label("Patch-token fraction per bin (%)", fontsize=11.0, labelpad=5)
    ticks = [
        value
        for value in (0.001, 0.01, 0.1, 1.0, 10.0, 50.0)
        if color_min <= value <= color_max
    ]
    colorbar.set_ticks(ticks)
    colorbar.set_ticklabels([f"{value:g}" for value in ticks])
    colorbar.ax.tick_params(labelsize=8.5, length=2.1, width=0.6, pad=2)
    return figure


def save_figure(figure: Any, output_stem: Path) -> tuple[Path, Path]:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = output_stem.with_suffix(".pdf")
    png_path = output_stem.with_suffix(".png")
    figure.savefig(pdf_path, format="pdf", dpi=RASTER_DPI, facecolor=BACKGROUND)
    figure.savefig(png_path, format="png", dpi=RASTER_DPI, facecolor=BACKGROUND)
    return pdf_path, png_path


def run(args: argparse.Namespace) -> tuple[Path, Path] | None:
    payload = load_plot_data(
        args.input,
        allow_partial_smoke=args.allow_partial_smoke,
    )
    values = load_patch_cache(payload, args.input)
    validate_cache_statistics(payload, values)
    edges, fractions = layer_histograms(values)
    if args.check_only:
        print(
            json.dumps(
                {
                    "study": STUDY_NAME,
                    "analysis": str(args.input.resolve()),
                    "patch_cache": str(
                        (args.input.parent / payload["cache"]["patch_cosine"]["file"]).resolve()
                    ),
                    "patch_tokens_per_layer": payload["patch_tokens_per_layer"],
                    "histogram_bins": HISTOGRAM_BINS,
                    "histogram_range": list(HISTOGRAM_RANGE),
                },
                indent=2,
            )
        )
        return None
    figure = draw(payload, edges, fractions)
    paths = save_figure(figure, args.output_dir / STUDY_NAME)
    import matplotlib.pyplot as plt

    plt.close(figure)
    for path in paths:
        print(f"saved={path}")
    return paths


def main(argv: Optional[Sequence[str]] = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
