#!/usr/bin/env python3
"""Freeze the Stage-1 ablation structure and train its token router."""

from sparmoe_vl.studies.component_ablation.training import main


if __name__ == "__main__":
    main(stage=2)
