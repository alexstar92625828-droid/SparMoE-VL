#!/usr/bin/env python3
"""Aggregate the three complete-pool OPTIN text runs."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from sparmoe_vl.baselines.text.common import (
    EXPECTED_PAIRED_PATH_POOL_SHA256,
    EXPECTED_TEXT_POOL_SHA256,
    POOL_SIZE,
    save_json,
)


FIELDS = {
    "active_text_parameters_m": lambda item: item["active_text_parameters_m"],
    "text_total_macs_g": lambda item: item["text_total_macs_g"],
    "text_ffn_macs_g": lambda item: item["text_ffn_macs_g"],
    "ffn_reduction_percent": lambda item: item["ffn_reduction_percent"],
}
for metric in ("i2t_r1", "i2t_r5", "i2t_r10", "t2i_r1", "t2i_r5", "t2i_r10"):
    FIELDS[f"coco_{metric}"] = lambda item, metric=metric: item["metrics"]["coco"][metric]


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
            "samples": POOL_SIZE,
            "text_pool_sha256": EXPECTED_TEXT_POOL_SHA256,
            "paired_path_pool_sha256": EXPECTED_PAIRED_PATH_POOL_SHA256,
        }
        if any(run.get(key) != value for key, value in expected.items()):
            raise ValueError(f"incomplete or mismatched OPTIN run: {path}")
        runs.append(run)

    summary = {
        "seeds": list(args.seeds),
        "samples_per_seed": POOL_SIZE,
        "text_pool_sha256": EXPECTED_TEXT_POOL_SHA256,
        "paired_path_pool_sha256": EXPECTED_PAIRED_PATH_POOL_SHA256,
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
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
