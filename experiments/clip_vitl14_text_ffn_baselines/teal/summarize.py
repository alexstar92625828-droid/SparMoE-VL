#!/usr/bin/env python3
"""Aggregate three complete-pool TEAL runs with sample standard deviation."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from sparmoe_vl.baselines.text.common import (
    EXPECTED_TEXT_POOL_SHA256,
    POOL_SIZE,
    save_json,
)


FIELDS = {
    "active_text_parameters_m": lambda item: item["active_text_parameters_m"],
    "macs_text_g": lambda item: item["macs_text_g"],
    "ffn_macs_text_g": lambda item: item["ffn_macs_text_g"],
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
        if run.get("processing_seed") != seed:
            raise ValueError(f"processing seed mismatch in {path}")
        if run.get("samples") != POOL_SIZE:
            raise ValueError(f"incomplete calibration pool in {path}")
        if run.get("dataset_sha256") != EXPECTED_TEXT_POOL_SHA256:
            raise ValueError(f"main-data fingerprint mismatch in {path}")
        runs.append(run)

    summary = {
        "seeds": args.seeds,
        "samples_per_seed": POOL_SIZE,
        "dataset_sha256": EXPECTED_TEXT_POOL_SHA256,
        "n": len(runs),
        "ddof": 1,
        "statistics": {},
    }
    for name, getter in FIELDS.items():
        values = [float(getter(run)) for run in runs]
        summary["statistics"][name] = {
            "values": values,
            "mean": statistics.mean(values),
            "sample_std": statistics.stdev(values),
        }
    save_json(summary, args.output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
