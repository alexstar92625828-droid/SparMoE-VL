#!/usr/bin/env python3
"""Aggregate three complete-pool visual FLAP evaluations."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from sparmoe_vl.baselines.vision.common import (
    EXPECTED_IMAGE_POOL_SHA256,
    POOL_SIZE,
    save_json,
)


FIELDS = {
    "active_visual_parameters_m": lambda item: item["active_visual_parameters_m"],
    "macs_vision_g": lambda item: item["macs_vision_g"],
    "ffn_macs_vision_g": lambda item: item["ffn_macs_vision_g"],
    "ffn_macs_reduction_percent": lambda item: item["ffn_macs_reduction_percent"],
}
for dataset in ("coco", "flickr30k"):
    for metric in ("i2t_r1", "i2t_r5", "i2t_r10", "t2i_r1", "t2i_r5", "t2i_r10"):
        FIELDS[f"{dataset}_{metric}"] = lambda item, dataset=dataset, metric=metric: item[
            "metrics"
        ][dataset][metric]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=(42, 123, 2026))
    args = parser.parse_args()

    runs = []
    for seed in args.seeds:
        path = args.root / f"seed_{seed}" / "result.json"
        with path.open(encoding="utf-8") as handle:
            run = json.load(handle)
        expected = {
            "seed": seed,
            "data_seed": 42,
            "unique_samples": POOL_SIZE,
            "calibration_exposures": POOL_SIZE,
            "dataset_sha256": EXPECTED_IMAGE_POOL_SHA256,
            "weight_updates": 0,
        }
        mismatches = {
            key: (run.get(key), value)
            for key, value in expected.items()
            if run.get(key) != value
        }
        if mismatches:
            raise ValueError(f"seed {seed} FLAP result violates protocol: {mismatches}")
        runs.append(run)

    aggregate = {
        "method": "FLAP-CLIP-FFN",
        "seeds": list(args.seeds),
        "data_seed": 42,
        "unique_samples_per_seed": POOL_SIZE,
        "calibration_exposures_per_seed": POOL_SIZE,
        "dataset_sha256": EXPECTED_IMAGE_POOL_SHA256,
        "n": len(runs),
        "ddof": 1,
        "statistics": {},
    }
    for name, getter in FIELDS.items():
        values = [float(getter(run)) for run in runs]
        aggregate["statistics"][name] = {
            "values": values,
            "mean": statistics.mean(values),
            "sample_std": statistics.stdev(values),
        }
    save_json(aggregate, args.output)
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
