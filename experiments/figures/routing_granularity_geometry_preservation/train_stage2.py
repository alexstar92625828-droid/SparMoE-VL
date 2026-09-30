#!/usr/bin/env python3
"""Freeze the Stage-1 structure and train token routing for one N."""

from sparmoe_vl.studies.routing_granularity_geometry_preservation.training import main


if __name__ == "__main__":
    main(phase="stage2")
