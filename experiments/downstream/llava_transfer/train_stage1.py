"""Learn CLIP ViT-L/14-336 SPG structure under the global budget."""

from sparmoe_vl.downstream.llava.training import main


if __name__ == "__main__":
    main(stage=1)
