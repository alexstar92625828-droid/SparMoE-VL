"""Verify every downstream benchmark against the result-generating data."""

import json

from sparmoe_vl.downstream.llava.data import validate_all_benchmarks


if __name__ == "__main__":
    print(json.dumps(validate_all_benchmarks(), indent=2, ensure_ascii=True))
