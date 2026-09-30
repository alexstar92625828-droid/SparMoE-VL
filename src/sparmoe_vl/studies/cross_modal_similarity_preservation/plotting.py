"""Render Dense and SparMoE-VL cross-modal similarity structure panels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from .analysis import (
    load_paper_coco,
    subset_coco,
    validate_analysis,
    validate_visualization,
)
from .protocol import (
    COCO_ANNOTATIONS,
    COCO_IMAGES,
    DISPLAY_GROUPS,
    OUTPUT_ROOT,
    STUDY_NAME,
)


RASTER_DPI = 600
BACKGROUND = "#FFFFFF"
TEXT_COLOR = "#202A33"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=OUTPUT_ROOT / "analysis.json")
    parser.add_argument("--coco-annotations", type=Path, default=COCO_ANNOTATIONS)
    parser.add_argument("--coco-images", type=Path, default=COCO_IMAGES)
    parser.add_argument("--display-groups", type=int, default=DISPLAY_GROUPS)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--allow-partial-smoke", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args(argv)


def load_plot_data(
    analysis_path: Path,
    coco_annotations: Path,
    coco_images: Path,
    *,
    display_groups: int,
    allow_partial_smoke: bool = False,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    if not analysis_path.is_file():
        raise FileNotFoundError(f"missing cross-modal analysis: {analysis_path}")
    payload = json.loads(analysis_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{analysis_path}: expected a JSON object")
    validate_analysis(
        payload,
        analysis_path,
        allow_partial_smoke=allow_partial_smoke,
    )
    sample_count = int(payload["visualization"]["sample_count"])
    if display_groups <= 0 or sample_count % display_groups != 0:
        raise ValueError("display groups must divide the visualization sample count")
    if not allow_partial_smoke and display_groups != DISPLAY_GROUPS:
        raise ValueError(f"paper visualization requires {DISPLAY_GROUPS} display groups")
    full_corpus = load_paper_coco(coco_annotations, coco_images)
    corpus = subset_coco(full_corpus, int(payload["evaluation"]["images"]))
    sample_path = analysis_path.parent / str(payload["visualization"]["sample_file"])
    manifest_path = analysis_path.parent / str(payload["visualization"]["manifest_file"])
    if not sample_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("visualization matrix or manifest is missing")
    with np.load(sample_path, allow_pickle=False) as stored:
        sample = {key: np.array(stored[key], copy=True) for key in stored.files}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("visualization manifest must contain a JSON object")
    validate_visualization(
        sample,
        manifest,
        corpus,
        sample_count,
        allow_partial_smoke=allow_partial_smoke,
    )
    return payload, sample, manifest


def aggregate_square_matrix(matrix: np.ndarray, groups: int) -> np.ndarray:
    size = int(matrix.shape[0])
    if matrix.shape != (size, size):
        raise ValueError(f"expected a square matrix, found {matrix.shape}")
    if groups <= 0 or size % groups != 0:
        raise ValueError(f"matrix size {size} is not divisible by {groups} groups")
    group_size = size // groups
    return matrix.reshape(groups, group_size, groups, group_size).mean(axis=(1, 3))


def similarity_colormap() -> Any:
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        "sparmoe_similarity",
        ("#F7FAFD", "#DCEAF6", "#9EC5E5", "#4F8FC5", "#174A7E", "#082C52"),
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
            "axes.edgecolor": "#52616F",
            "axes.labelcolor": TEXT_COLOR,
            "xtick.color": "#27333D",
            "ytick.color": "#27333D",
            "figure.facecolor": BACKGROUND,
            "axes.facecolor": BACKGROUND,
            "savefig.facecolor": BACKGROUND,
        }
    )


def style_axis(axis: Any, title: str, groups: int, show_y: bool) -> None:
    if groups == DISPLAY_GROUPS:
        ticks = [0, 15, 31, 47, 63]
    else:
        ticks = np.unique(np.linspace(0, groups - 1, min(groups, 5), dtype=int)).tolist()
    labels = [str(value + 1) for value in ticks]
    axis.set_xticks(ticks, labels)
    axis.set_yticks(ticks, labels if show_y else [])
    axis.set_xlabel("Text group", fontsize=11.0, labelpad=3.5)
    if show_y:
        axis.set_ylabel("Image group", fontsize=11.0, labelpad=4.5)
    axis.tick_params(length=2.4, width=0.7, labelsize=8.5, pad=2.2)
    if groups >= 8:
        macro = max(1, groups // 8)
        boundaries = np.arange(macro - 0.5, groups, macro)
        axis.set_xticks(boundaries, minor=True)
        axis.set_yticks(boundaries, minor=True)
        axis.grid(which="minor", color="white", linewidth=0.42, alpha=0.90)
        axis.tick_params(which="minor", bottom=False, left=False)
    axis.set_title(title, fontsize=12.5, fontweight="semibold", color=TEXT_COLOR, pad=4.5)
    for spine in axis.spines.values():
        spine.set_linewidth(0.65)
        spine.set_color("#52616F")


def draw(
    payload: Mapping[str, Any],
    sample: Mapping[str, np.ndarray],
    display_groups: int,
) -> Any:
    import matplotlib.pyplot as plt

    configure_matplotlib()
    dense = aggregate_square_matrix(np.asarray(sample["dense"]), display_groups)
    sparse = aggregate_square_matrix(np.asarray(sample["sparse"]), display_groups)
    shared = np.concatenate([dense.reshape(-1), sparse.reshape(-1)])
    lower = float(np.quantile(shared, 0.005))
    upper = float(np.quantile(shared, 0.997))
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        padding = max(1e-4, abs(lower) * 1e-3)
        lower, upper = lower - padding, upper + padding

    figure = plt.figure(figsize=(8.2, 4.1), facecolor=BACKGROUND)
    grid = figure.add_gridspec(
        1,
        3,
        width_ratios=(1.0, 1.0, 0.045),
        left=0.075,
        right=0.905,
        bottom=0.14,
        top=0.82,
        wspace=0.12,
    )
    dense_axis = figure.add_subplot(grid[0, 0])
    sparse_axis = figure.add_subplot(grid[0, 1])
    colorbar_axis = figure.add_subplot(grid[0, 2])
    dense_image = dense_axis.imshow(
        dense,
        cmap=similarity_colormap(),
        vmin=lower,
        vmax=upper,
        origin="upper",
        interpolation="nearest",
        aspect="equal",
        rasterized=True,
    )
    sparse_axis.imshow(
        sparse,
        cmap=similarity_colormap(),
        vmin=lower,
        vmax=upper,
        origin="upper",
        interpolation="nearest",
        aspect="equal",
        rasterized=True,
    )
    style_axis(dense_axis, "(a)  Dense CLIP", display_groups, show_y=True)
    style_axis(sparse_axis, "(b)  SparMoE-VL", display_groups, show_y=False)
    statistics = payload["statistics"]
    evaluation = payload["evaluation"]
    header = (
        f"Full COCO val2017: {evaluation['images']:,} images, "
        f"{evaluation['captions']:,} captions"
        f"   |   Pearson $r$ = {statistics['pearson']:.4f}"
        f"   |   Matrix cosine = {statistics['matrix_cosine']:.4f}"
        f"   |   MAE = {statistics['mae']:.4f}"
    )
    figure.text(
        0.5,
        0.955,
        header,
        ha="center",
        va="top",
        color=TEXT_COLOR,
        fontsize=8.5,
        fontweight="semibold",
    )
    colorbar = figure.colorbar(dense_image, cax=colorbar_axis)
    colorbar.set_label("Cosine similarity", fontsize=11.0, labelpad=4.5)
    colorbar.ax.tick_params(labelsize=8.5, length=2.3, width=0.65, pad=2.2)
    colorbar.outline.set_linewidth(0.65)
    colorbar.outline.set_edgecolor("#52616F")
    return figure


def save_figure(figure: Any, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / STUDY_NAME
    pdf_path = stem.with_suffix(".pdf")
    png_path = stem.with_suffix(".png")
    figure.savefig(pdf_path, format="pdf", dpi=RASTER_DPI, facecolor=BACKGROUND)
    figure.savefig(png_path, format="png", dpi=RASTER_DPI, facecolor=BACKGROUND)
    return pdf_path, png_path


def run(args: argparse.Namespace) -> tuple[Path, Path] | None:
    payload, sample, _ = load_plot_data(
        args.input,
        args.coco_annotations,
        args.coco_images,
        display_groups=args.display_groups,
        allow_partial_smoke=args.allow_partial_smoke,
    )
    if args.check_only:
        print(
            json.dumps(
                {
                    "study": STUDY_NAME,
                    "analysis": str(args.input.resolve()),
                    "sample_count": payload["visualization"]["sample_count"],
                    "display_groups": args.display_groups,
                    "shared_color_range": True,
                    "output_dir": str(args.output_dir.resolve()),
                },
                indent=2,
            )
        )
        return None
    figure = draw(payload, sample, args.display_groups)
    paths = save_figure(figure, args.output_dir)
    import matplotlib.pyplot as plt

    plt.close(figure)
    for path in paths:
        print(f"saved={path}")
    return paths


def main(argv: Optional[Sequence[str]] = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
