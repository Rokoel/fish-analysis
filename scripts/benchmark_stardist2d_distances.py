"""Evaluate fine-tuned StarDist 2D using an explicit held-out audit."""
from benchmark_all_heldout import main
if __name__=='__main__':
    raise SystemExit(main(pipelines=('stardist2d',)))
