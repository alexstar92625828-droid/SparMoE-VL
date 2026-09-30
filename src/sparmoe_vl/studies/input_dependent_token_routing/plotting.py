"""Render the vision and text routing panels from validated route analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from .analysis import validate_analysis, validate_fixed_inputs
from .protocol import (
    IMAGE_ROOT,
    NUM_EXPERTS,
    OUTPUT_ROOT,
    OVERLAY_ALPHA,
    PATCH_GRID_SIZE,
    TEXT_LAYER,
    VISION_LAYERS,
)


PANEL_WIDTH = 7.16
PANEL_HEIGHT = 6.98
SINGLE_COLUMN_WIDTH = 3.45
CANVAS_SCALE = PANEL_WIDTH / SINGLE_COLUMN_WIDTH
RASTER_DPI = 600
TEXT_COLOR = "#20242A"
DENSE_COLOR = "#343A40"
GRID_COLOR = "#DCE6F0"
BACKGROUND = "#FFFFFF"
EXPERT_COLORS = ("#DCEAF6", "#9EC5E5", "#4F8FC5", "#174A7E")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=OUTPUT_ROOT / "analysis.json")
    parser.add_argument("--image-root", type=Path, default=IMAGE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args(argv)


def load_plot_data(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing routing analysis: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    validate_analysis(payload, path)
    return payload


def configure_matplotlib() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": RASTER_DPI,
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Nimbus Sans",
                "Arial",
                "Helvetica",
                "Liberation Sans",
                "DejaVu Sans",
            ],
            "font.size": 8.0,
            "axes.linewidth": 0.8,
            "axes.edgecolor": DENSE_COLOR,
            "axes.labelcolor": TEXT_COLOR,
            "grid.color": GRID_COLOR,
            "figure.facecolor": BACKGROUND,
            "axes.facecolor": BACKGROUND,
            "savefig.facecolor": BACKGROUND,
            "savefig.transparent": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(figure: Any, output_stem: Path) -> tuple[Path, Path]:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = output_stem.with_suffix(".pdf")
    png_path = output_stem.with_suffix(".png")
    figure.savefig(pdf_path, format="pdf", facecolor=BACKGROUND)
    figure.savefig(png_path, format="png", dpi=RASTER_DPI, facecolor=BACKGROUND)
    return pdf_path, png_path


def display_image(path: Path) -> np.ndarray:
    """Apply the Dense CLIP resize and center crop without loading its weights."""

    from PIL import Image
    from torchvision.transforms import CenterCrop, InterpolationMode, Resize

    transform = Resize(224, interpolation=InterpolationMode.BICUBIC)
    crop = CenterCrop(224)
    with Image.open(path) as image:
        prepared = crop(transform(image.convert("RGB")))
        return np.asarray(prepared, dtype=np.float32) / 255.0


def desaturate_image(image: np.ndarray, amount: float = 0.78) -> np.ndarray:
    luminance = 0.2126 * image[..., 0] + 0.7152 * image[..., 1] + 0.0722 * image[..., 2]
    grayscale = np.repeat(luminance[..., None], 3, axis=-1)
    return np.clip((1.0 - amount) * image + amount * grayscale, 0.0, 1.0)


def draw_capacity_key(axis: Any) -> None:
    import matplotlib.patches as patches

    axis.set_axis_off()
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    center = 0.5
    swatch_width = 0.052
    gap = 0.010
    total_width = NUM_EXPERTS * swatch_width + (NUM_EXPERTS - 1) * gap
    start = center - total_width / 2
    axis.text(
        start - 0.018,
        0.51,
        "Lower capacity",
        ha="right",
        va="center",
        color=TEXT_COLOR,
        fontsize=6.15 * CANVAS_SCALE,
    )
    for index, color in enumerate(EXPERT_COLORS):
        x = start + index * (swatch_width + gap)
        axis.add_patch(
            patches.FancyBboxPatch(
                (x, 0.24),
                swatch_width,
                0.52,
                boxstyle="round,pad=0.004,rounding_size=0.012",
                linewidth=0.55,
                edgecolor=BACKGROUND,
                facecolor=color,
            )
        )
        axis.text(
            x + swatch_width / 2,
            0.50,
            f"E{index + 1}",
            ha="center",
            va="center",
            color="white" if index >= 2 else TEXT_COLOR,
            fontsize=6.3 * CANVAS_SCALE,
            fontweight="semibold",
        )
    axis.text(
        start + total_width + 0.018,
        0.51,
        "Higher capacity",
        ha="left",
        va="center",
        color=TEXT_COLOR,
        fontsize=6.15 * CANVAS_SCALE,
    )


def style_image_axis(axis: Any) -> None:
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.65)
        spine.set_edgecolor(GRID_COLOR)


def draw_vision_panel(
    base_images: Sequence[np.ndarray],
    assignments: Mapping[str, Any],
) -> Any:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    configure_matplotlib()
    figure = plt.figure(figsize=(PANEL_WIDTH, PANEL_HEIGHT), facecolor=BACKGROUND)
    grid = figure.add_gridspec(
        len(base_images) + 2,
        4,
        height_ratios=(0.16, 0.12) + (1.0,) * len(base_images),
        left=0.018,
        right=0.992,
        bottom=0.012,
        top=0.988,
        wspace=0.035,
        hspace=0.035,
    )
    draw_capacity_key(figure.add_subplot(grid[0, :]))
    for column, title in enumerate(("Input", *(f"Layer {value}" for value in VISION_LAYERS))):
        header = figure.add_subplot(grid[1, column])
        header.set_axis_off()
        header.text(
            0.5,
            0.48,
            title,
            ha="center",
            va="center",
            color=TEXT_COLOR,
            fontsize=6.6 * CANVAS_SCALE,
            fontweight="semibold",
            transform=header.transAxes,
        )

    colormap = ListedColormap(EXPERT_COLORS)
    for image_index, base_image in enumerate(base_images):
        axes = [figure.add_subplot(grid[image_index + 2, column]) for column in range(4)]
        height, width = base_image.shape[:2]
        axes[0].imshow(base_image, interpolation="lanczos")
        style_image_axis(axes[0])
        routing_base = desaturate_image(base_image)
        for axis, layer_number in zip(axes[1:], VISION_LAYERS):
            ids = np.asarray(assignments[str(layer_number)][image_index], dtype=int)
            axis.imshow(routing_base, interpolation="lanczos")
            axis.imshow(
                ids,
                cmap=colormap,
                vmin=-0.5,
                vmax=3.5,
                alpha=OVERLAY_ALPHA,
                interpolation="nearest",
                extent=(-0.5, width - 0.5, height - 0.5, -0.5),
            )
            patch_width = width / PATCH_GRID_SIZE
            patch_height = height / PATCH_GRID_SIZE
            axis.set_xticks(
                np.arange(PATCH_GRID_SIZE + 1) * patch_width - 0.5,
                minor=True,
            )
            axis.set_yticks(
                np.arange(PATCH_GRID_SIZE + 1) * patch_height - 0.5,
                minor=True,
            )
            axis.grid(which="minor", color="white", linewidth=0.22, alpha=0.34)
            axis.tick_params(which="minor", bottom=False, left=False)
            style_image_axis(axis)
    return figure


def measure_words(axis: Any, words: list[dict[str, Any]]) -> tuple[Any, float]:
    from matplotlib.font_manager import FontProperties

    figure = axis.figure
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    axis_bbox = axis.get_window_extent(renderer=renderer)
    font = FontProperties(family="Nimbus Sans", size=6.15 * CANVAS_SCALE, weight="normal")
    pad_px = 2.2 * CANVAS_SCALE * figure.dpi / 72.0
    gap_px = 1.8 * CANVAS_SCALE * figure.dpi / 72.0
    for word in words:
        text_width, _, _ = renderer.get_text_width_height_descent(
            word["label"], font, ismath=False
        )
        word["draw_width"] = (text_width + 2 * pad_px) / axis_bbox.width
    return font, gap_px / axis_bbox.width


def wrap_words(
    words: Sequence[dict[str, Any]],
    left: float,
    right: float,
    gap: float,
) -> list[list[dict[str, Any]]]:
    rows: list[list[dict[str, Any]]] = []
    row: list[dict[str, Any]] = []
    cursor = left
    for word in words:
        required = word["draw_width"] if not row else gap + word["draw_width"]
        if row and cursor + required > right:
            rows.append(row)
            row = []
            cursor = left
            required = word["draw_width"]
        row.append(word)
        cursor += required
    if row:
        rows.append(row)
    return rows


def draw_text_capacity_legend(axis: Any, divider_x: float) -> None:
    import matplotlib.patches as patches

    axis.text(
        0.040,
        0.940,
        "Expert\ncapacity",
        ha="left",
        va="top",
        fontsize=6.6 * CANVAS_SCALE,
        fontweight="semibold",
        color=TEXT_COLOR,
        linespacing=1.05,
        transform=axis.transAxes,
    )
    for expert, (color, y) in enumerate(
        zip(EXPERT_COLORS, (0.665, 0.535, 0.405, 0.275)),
        start=1,
    ):
        axis.add_patch(
            patches.Rectangle(
                (0.041, y - 0.043 / 2),
                0.050,
                0.043,
                linewidth=0,
                facecolor=color,
                transform=axis.transAxes,
            )
        )
        axis.text(
            0.105,
            y,
            f"E{expert}",
            ha="left",
            va="center",
            fontsize=6.3 * CANVAS_SCALE,
            color=TEXT_COLOR,
            transform=axis.transAxes,
        )
    axis.text(
        0.041,
        0.060,
        f"Layer {TEXT_LAYER}",
        ha="left",
        va="center",
        fontsize=6.2 * CANVAS_SCALE,
        fontweight="semibold",
        color="#174A7E",
        transform=axis.transAxes,
    )
    axis.plot(
        [divider_x, divider_x],
        [0.025, 0.965],
        color="#AEBBC8",
        linewidth=0.55 * CANVAS_SCALE,
        transform=axis.transAxes,
        clip_on=False,
    )


def draw_word_box(
    axis: Any,
    x: float,
    y: float,
    width: float,
    height: float,
    word: Mapping[str, Any],
    font: Any,
) -> None:
    import matplotlib.patches as patches

    weights = np.asarray(
        [max(1, len(piece)) for piece in word["piece_labels"]],
        dtype=float,
    )
    weights /= weights.sum()
    clip_shape = patches.FancyBboxPatch(
        (x, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0,rounding_size=0.004",
        linewidth=0,
        facecolor="none",
        transform=axis.transAxes,
    )
    axis.add_patch(clip_shape)
    segments = []
    piece_x = x
    for route, fraction in zip(word["piece_routes"], weights):
        piece_width = width * float(fraction)
        segment = patches.Rectangle(
            (piece_x, y - height / 2),
            piece_width,
            height,
            linewidth=0,
            facecolor=EXPERT_COLORS[route],
            transform=axis.transAxes,
            zorder=2,
        )
        segment.set_clip_path(clip_shape)
        axis.add_patch(segment)
        segments.append((route, segment))
        piece_x += piece_width
    axis.add_patch(
        patches.FancyBboxPatch(
            (x, y - height / 2),
            width,
            height,
            boxstyle="round,pad=0,rounding_size=0.004",
            linewidth=0.34 * CANVAS_SCALE,
            edgecolor=BACKGROUND,
            facecolor="none",
            transform=axis.transAxes,
            zorder=3,
        )
    )
    for route, segment in segments:
        label = axis.text(
            x + width / 2,
            y,
            word["label"],
            ha="center",
            va="center",
            color="white" if route >= 2 else TEXT_COLOR,
            fontproperties=font,
            transform=axis.transAxes,
            clip_on=True,
            zorder=4,
        )
        label.set_clip_path(segment)


def draw_text_panel(source_words: Sequence[Mapping[str, Any]]) -> Any:
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    configure_matplotlib()
    words = [dict(word) for word in source_words]
    figure, axis = plt.subplots(figsize=(PANEL_WIDTH, PANEL_HEIGHT), facecolor=BACKGROUND)
    figure.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    axis.set_axis_off()
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.add_patch(
        patches.Rectangle(
            (0.02918, 0.01250),
            0.95165,
            0.96545,
            linewidth=0.62 * CANVAS_SCALE,
            edgecolor="#8F9EAC",
            facecolor=BACKGROUND,
            transform=axis.transAxes,
        )
    )
    divider_x = 0.205
    draw_text_capacity_legend(axis, divider_x)
    left, right = 0.230, 0.970
    font, gap = measure_words(axis, words)
    rows = wrap_words(words, left, right, gap)
    top, bottom = 0.945, 0.055
    centers = [0.5] if len(rows) == 1 else np.linspace(top, bottom, len(rows))
    spacing = (top - bottom) / max(1, len(rows) - 1)
    box_height = min(0.066, spacing * 0.62)
    for center, row in zip(centers, rows):
        cursor = left
        for index, word in enumerate(row):
            if index:
                cursor += gap
            draw_word_box(
                axis,
                cursor,
                float(center),
                word["draw_width"],
                box_height,
                word,
                font,
            )
            cursor += word["draw_width"]
    return figure


def run(args: argparse.Namespace) -> dict[str, Any]:
    payload = load_plot_data(args.input)
    image_paths = validate_fixed_inputs(args.image_root)
    check = {
        "input": str(args.input.resolve()),
        "image_root": str(args.image_root.resolve()),
        "image_count": len(image_paths),
        "vision_layers_one_based": list(VISION_LAYERS),
        "text_layer_one_based": TEXT_LAYER,
        "output_dir": str(args.output_dir.resolve()),
    }
    if args.check_only:
        print(json.dumps(check, indent=2))
        return check

    import matplotlib.pyplot as plt

    base_images = [display_image(path) for path in image_paths]
    vision_figure = draw_vision_panel(
        base_images,
        payload["vision"]["assignments_by_layer"],
    )
    vision_paths = save_figure(
        vision_figure,
        args.output_dir / "vision_token_routing",
    )
    plt.close(vision_figure)
    text_figure = draw_text_panel(payload["text"]["words"])
    text_paths = save_figure(
        text_figure,
        args.output_dir / "text_token_routing",
    )
    plt.close(text_figure)
    result = {
        **check,
        "generated": [str(path) for path in (*vision_paths, *text_paths)],
    }
    print(json.dumps(result, indent=2))
    return result


def main(argv: Optional[Sequence[str]] = None) -> None:
    run(parse_args(argv))
