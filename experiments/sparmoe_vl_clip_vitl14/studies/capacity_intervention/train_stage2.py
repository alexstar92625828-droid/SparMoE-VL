#!/usr/bin/env python3
"""Freeze the N=8 Stage-1 structure and train its token router."""

from sparmoe_vl.studies.capacity_intervention.training import main


if __name__ == "__main__":
    main(stage=2)
