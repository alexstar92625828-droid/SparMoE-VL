"""Render the routing-granularity geometry-preservation analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from .protocol import (
    COCO_ANNOTATIONS_SHA256,
    COCO_CAPTION_COUNT,
    COCO_CAPTION_IMAGE_INDEX_SHA256,
    COCO_CAPTION_ORDER_SHA256,
    COCO_IMAGE_COUNT,
    COCO_IMAGE_ORDER_SHA256,
    DENSE_FFN_MACS_G,
    EXPERT_COUNTS,
    IMAGE_BATCH_SIZE,
    MODEL_KEY,
    MODEL_NAME,
    OUTPUT_ROOT,
    PROTOCOL,
    RUN_SEED,
    STUDY_NAME,
    TRAINING_POOL_SHA256,
    VISION_LAYERS,
    capacity_factors,
    training_protocol,
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=OUTPUT_ROOT / "analysis.json")
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=OUTPUT_ROOT / "routing_geometry_preservation",
    )
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args(argv)


def _finite_number(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be numeric") from error
    if not np.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def validate_analysis(payload: Mapping[str, Any], source: str | Path) -> None:
    """Reject partial, mismatched, or non-capacity-matched analysis files."""

    expected = {
        "study": STUDY_NAME,
        "protocol": PROTOCOL,
        "model_name": MODEL_NAME,
        "model_key": MODEL_KEY,
        "run_seed": RUN_SEED,
    }
    for key, wanted in expected.items():
        if payload.get(key) != wanted:
            raise ValueError(f"{source}: {key}={payload.get(key)!r}; expected {wanted!r}")
    control = payload.get("control")
    if not isinstance(control, Mapping) or control.get("name") != "capacity_matched_shuffle":
        raise ValueError(f"{source}: missing capacity-matched control identity")
    evaluation = payload.get("evaluation")
    expected_evaluation = {
        "dataset": "COCO-val2017",
        "images": COCO_IMAGE_COUNT,
        "captions": COCO_CAPTION_COUNT,
        "annotation_sha256": COCO_ANNOTATIONS_SHA256,
        "image_order_sha256": COCO_IMAGE_ORDER_SHA256,
        "caption_order_sha256": COCO_CAPTION_ORDER_SHA256,
        "caption_image_index_sha256": COCO_CAPTION_IMAGE_INDEX_SHA256,
    }
    if not isinstance(evaluation, Mapping):
        raise ValueError(f"{source}: missing COCO evaluation identity")
    for key, wanted in expected_evaluation.items():
        if evaluation.get(key) != wanted:
            raise ValueError(
                f"{source}: evaluation.{key}={evaluation.get(key)!r}; expected {wanted!r}"
            )
    dense = payload.get("dense")
    if not isinstance(dense, Mapping) or not np.isclose(
        _finite_number(dense.get("vision_ffn_macs_g"), "Dense FFN MACs"),
        DENSE_FFN_MACS_G,
        atol=1e-12,
    ):
        raise ValueError(f"{source}: invalid Dense FFN MAC convention")
    experiments = payload.get("experiments")
    if not isinstance(experiments, Mapping):
        raise ValueError(f"{source}: missing experiments")
    if set(experiments) != {str(value) for value in EXPERT_COUNTS}:
        raise ValueError(f"{source}: expected complete N={EXPERT_COUNTS} results")

    for count in EXPERT_COUNTS:
        experiment = experiments[str(count)]
        if not isinstance(experiment, Mapping):
            raise ValueError(f"{source}: invalid N={count} experiment")
        checkpoint = experiment.get("checkpoint")
        if not isinstance(checkpoint, Mapping):
            raise ValueError(f"{source}: N={count} is missing checkpoint identity")
        if checkpoint.get("expert_count") != count:
            raise ValueError(f"{source}: N={count} checkpoint identity disagrees")
        if checkpoint.get("dataset_sha256") != TRAINING_POOL_SHA256:
            raise ValueError(f"{source}: N={count} did not use the main training pool")
        if checkpoint.get("training_protocol") != training_protocol(count):
            raise ValueError(f"{source}: N={count} used the wrong training protocol")
        if tuple(experiment.get("capacity_factors", ())) != capacity_factors(count):
            raise ValueError(f"{source}: N={count} has the wrong capacity factors")
        verification = experiment.get("capacity_match_verification")
        if not isinstance(verification, Mapping):
            raise ValueError(f"{source}: N={count} has no capacity-match verification")
        checks = verification.get("per_layer_per_batch_multiset_checks")
        matched = verification.get("matched_checks")
        expected_checks = VISION_LAYERS * (
            (COCO_IMAGE_COUNT + IMAGE_BATCH_SIZE - 1) // IMAGE_BATCH_SIZE
        )
        if checks != expected_checks or matched != checks:
            raise ValueError(f"{source}: N={count} did not pass every multiset check")
        difference = _finite_number(
            verification.get("ffn_macs_absolute_difference_g"),
            f"N={count} FFN MAC difference",
        )
        if difference > 1e-10:
            raise ValueError(f"{source}: N={count} routes are not compute matched")
        routes = []
        for route in ("learned", "capacity_matched_shuffle"):
            metrics = experiment.get(route)
            if not isinstance(metrics, Mapping):
                raise ValueError(f"{source}: N={count} is missing {route}")
            values = {
                key: _finite_number(metrics.get(key), f"N={count} {route} {key}")
                for key in (
                    "dense_cosine",
                    "mean_r1_retention",
                    "vision_ffn_macs_g",
                )
            }
            if not -1.0 <= values["dense_cosine"] <= 1.0:
                raise ValueError(f"{source}: N={count} has invalid cosine")
            routes.append(values)
        if not np.isclose(
            routes[0]["vision_ffn_macs_g"],
            routes[1]["vision_ffn_macs_g"],
            atol=1e-10,
        ):
            raise ValueError(f"{source}: N={count} learned and shuffled MACs differ")


def load_plot_data(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"missing routing-granularity analysis: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path}: expected a JSON object")
    validate_analysis(payload, path)
    experiments = payload["experiments"]
    experts = np.asarray(EXPERT_COUNTS, dtype=float)
    ffn_macs = np.asarray(
        [experiments[str(count)]["learned"]["vision_ffn_macs_g"] for count in EXPERT_COUNTS],
        dtype=float,
    )
    return {
        "experts": experts,
        "ffn_reduction": 100.0 * (1.0 - ffn_macs / DENSE_FFN_MACS_G),
        "learned_retention": np.asarray(
            [
                experiments[str(count)]["learned"]["mean_r1_retention"]
                for count in EXPERT_COUNTS
            ],
            dtype=float,
        ),
        "shuffled_retention": np.asarray(
            [
                experiments[str(count)]["capacity_matched_shuffle"]["mean_r1_retention"]
                for count in EXPERT_COUNTS
            ],
            dtype=float,
        ),
        "learned_cosine": np.asarray(
            [experiments[str(count)]["learned"]["dense_cosine"] for count in EXPERT_COUNTS],
            dtype=float,
        ),
        "shuffled_cosine": np.asarray(
            [
                experiments[str(count)]["capacity_matched_shuffle"]["dense_cosine"]
                for count in EXPERT_COUNTS
            ],
            dtype=float,
        ),
    }


def _padded_limits(
    values: np.ndarray, fraction: float, minimum_pad: float
) -> tuple[float, float]:
    low = float(values.min())
    high = float(values.max())
    pad = max((high - low) * fraction, minimum_pad)
    return low - pad, high + pad


def draw(output_stem: Path, values: Mapping[str, np.ndarray]) -> tuple[Path, Path]:
    try:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap, Normalize
        from matplotlib.lines import Line2D
    except ImportError as error:
        raise RuntimeError("install the analysis dependencies to render this panel") from error

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    experts = values["experts"]
    reductions = values["ffn_reduction"]
    learned_retention = values["learned_retention"]
    shuffled_retention = values["shuffled_retention"]
    learned_cosine = values["learned_cosine"]
    shuffled_cosine = values["shuffled_cosine"]
    all_retention = np.concatenate((learned_retention, shuffled_retention, [100.0]))
    all_cosine = np.concatenate((learned_cosine, shuffled_cosine))
    cosine_low, cosine_high = _padded_limits(all_cosine, 0.08, 0.002)
    normalization = Normalize(vmin=max(-1.0, cosine_low), vmax=min(1.0, cosine_high))
    blue_map = LinearSegmentedColormap.from_list(
        "routing_geometry_blues",
        ("#E4F0F7", "#B7D7EA", "#79B3D5", "#3786B8", "#0B4F7D"),
    )

    background = "#FFFFFF"
    text = "#263640"
    dense = "#394B59"
    grid = "#D9E3EA"
    learned_line = "#0B4F7D"
    shuffled_line = "#7D96A5"
    gain_color = "#63A9D0"
    figure = plt.figure(figsize=(7.15, 4.05), dpi=300)
    axis = figure.add_subplot(111, projection="3d")
    figure.patch.set_facecolor(background)
    axis.set_facecolor(background)

    x_limits = _padded_limits(experts, 0.05, 0.35)
    y_limits = _padded_limits(reductions, 0.14, 0.5)
    z_limits = _padded_limits(all_retention, 0.08, 1.0)
    z_limits = (z_limits[0], max(z_limits[1], 100.5))
    plane_x, plane_y = np.meshgrid(x_limits, y_limits)
    axis.plot_surface(
        plane_x,
        plane_y,
        np.full_like(plane_x, 100.0),
        color="#C9D7E0",
        alpha=0.075,
        linewidth=0,
        shade=False,
        zorder=0,
    )
    for x, y, low, high in zip(experts, reductions, shuffled_retention, learned_retention):
        axis.quiver(
            x,
            y,
            low,
            0,
            0,
            high - low,
            color=gain_color,
            linewidth=1.45,
            arrow_length_ratio=0.09,
            alpha=0.78,
            zorder=2,
        )
    axis.plot(
        experts,
        reductions,
        learned_retention,
        color=learned_line,
        linewidth=2.1,
        solid_capstyle="round",
        zorder=4,
    )
    axis.plot(
        experts,
        reductions,
        shuffled_retention,
        color=shuffled_line,
        linewidth=1.65,
        linestyle=(0, (3.2, 2.2)),
        zorder=3,
    )
    axis.scatter(
        experts,
        reductions,
        learned_retention,
        s=78,
        c=blue_map(normalization(learned_cosine)),
        marker="o",
        edgecolor="white",
        linewidth=1.05,
        depthshade=False,
        zorder=7,
    )
    axis.scatter(
        experts,
        reductions,
        shuffled_retention,
        s=62,
        c=blue_map(normalization(shuffled_cosine)),
        marker="s",
        edgecolor="#667E8D",
        linewidth=0.95,
        depthshade=False,
        zorder=6,
    )

    axis.set_xlabel("Number of experts", labelpad=5.0, fontsize=11.0)
    axis.set_ylabel(r"FFN reduction (%) $\uparrow$", labelpad=4.0, fontsize=11.0)
    axis.set_zlabel("")
    axis.set_xlim(*x_limits)
    axis.set_ylim(*y_limits)
    axis.set_zlim(*z_limits)
    axis.set_xticks(experts)
    axis.set_xticklabels([str(int(value)) for value in experts])
    axis.view_init(elev=25.5, azim=-56.0)
    axis.set_box_aspect((1.30, 1.0, 0.82), zoom=1.12)
    for pane_axis in (axis.xaxis, axis.yaxis, axis.zaxis):
        pane_axis.pane.set_facecolor((0.975, 0.986, 0.996, 1.0))
        pane_axis.pane.set_edgecolor((0.68, 0.75, 0.82, 0.82))
        pane_axis._axinfo["grid"].update({"color": grid, "linewidth": 0.55, "linestyle": "-"})
        pane_axis._axinfo["axisline"].update({"color": dense, "linewidth": 0.7})

    figure.legend(
        handles=(
            Line2D(
                [0],
                [0],
                marker="o",
                color=learned_line,
                linewidth=1.8,
                markersize=5.8,
                markerfacecolor="#3786B8",
                markeredgecolor="white",
                label="Learned routing",
            ),
            Line2D(
                [0],
                [0],
                marker="s",
                color=shuffled_line,
                linewidth=1.5,
                linestyle=(0, (3.2, 2.2)),
                markersize=5.2,
                markerfacecolor="#B7D7EA",
                markeredgecolor="#667E8D",
                label="Capacity-matched shuffle",
            ),
        ),
        loc="upper right",
        bbox_to_anchor=(0.91, 0.985),
        frameon=False,
        fontsize=8.5,
        handlelength=2.2,
    )
    scalar = mpl.cm.ScalarMappable(norm=normalization, cmap=blue_map)
    scalar.set_array([])
    colorbar_axis = figure.add_axes([0.78, 0.17, 0.018, 0.67])
    colorbar = figure.colorbar(scalar, cax=colorbar_axis)
    colorbar.set_label("Cosine to Dense", fontsize=11.0, color=text, labelpad=5)
    colorbar.ax.tick_params(labelsize=8.5, width=0.65, length=2.5, colors=dense)
    colorbar.outline.set_linewidth(0.7)
    colorbar.outline.set_edgecolor("#647383")
    figure.text(
        0.71,
        0.52,
        r"Mean R@1 retention (%) $\uparrow$",
        rotation=90,
        ha="center",
        va="center",
        fontsize=11.0,
        color=text,
    )
    axis.set_position([-0.10, 0.075, 0.90, 0.96])
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = output_stem.with_suffix(".pdf")
    png_path = output_stem.with_suffix(".png")
    figure.savefig(pdf_path, format="pdf", bbox_inches="tight", pad_inches=0.025)
    figure.savefig(png_path, format="png", dpi=600, bbox_inches="tight", pad_inches=0.025)
    plt.close(figure)
    return pdf_path, png_path


def run(args: argparse.Namespace) -> dict[str, Any]:
    values = load_plot_data(args.input)
    result: dict[str, Any] = {
        "input": str(args.input.resolve()),
        "expert_counts": [int(value) for value in values["experts"]],
        "routing_conditions": ["learned", "capacity_matched_shuffle"],
    }
    if args.check_only:
        print(json.dumps(result, indent=2))
        return result
    pdf_path, png_path = draw(args.output_stem, values)
    result.update(pdf=str(pdf_path), png=str(png_path))
    print(json.dumps(result, indent=2))
    return result


def main(argv: Optional[Sequence[str]] = None) -> None:
    run(parse_args(argv))
