"""Freeze the CLIP-336 Stage-1 structure and train token routers only."""

from sparmoe_vl.downstream.llava.training import main


if __name__ == "__main__":
    main(stage=2)
