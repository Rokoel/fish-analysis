# Training the retained networks

[Documentation](README.md) · [Data preparation](training-data.md)

Analysis exposes fixed inference pipelines. Developer training remains separate:
StarDist 3D proposals, center/box detection and the contour U-Net support consensus;
fine-tuned StarDist 2D supports the other pipeline; the signal-center U-Net is shared.
Training uses explicit new data and output paths. Archived metadata and annotations
are no longer bundled.

The Viewer **Neural training…** panel handles sparse StarDist training and saved-run
review, with process-isolated TensorFlow, loss/progress graphs and cancellation.
Its current initial policy uses visible boundary evidence; saved model parameters
remain available for restoring/checking a specific trained detector. Collection
audits retain acquisition splits, duplicate identities and source fingerprints.
Unpainted labels are ignored; only reviewed background is negative.

For a new collection, use the neural audit worker with `--audit-only` and an explicit
input folder/output directory. The resulting calibrated cache is supplied to later
workers. Run each retained command with `--help` for its required data/index files:

```sh
python -m fish_viewer.neural /data/scans --audit-only --output /data/cell-audit --evidence-only
python scripts/finetune_stardist2d.py --help
python -m fish_viewer.proposals --help
python -m fish_viewer.refinement --help
python scripts/build_signal_centers.py --help
python scripts/train_signal_centers.py --help
python scripts/evaluate_signal_centers.py --help
```

The StarDist 2D fine-tuner recreates a reproducible official pretrained cache when
needed, preserves the source license, and writes preprocessing/acceptance/linking
recipes. The removed preprocessed 3D checkpoint is not required.

Signal data-building accepts either a fresh supervision audit or a completed cell
prediction report plus its collection audit. Physical resampling, ambiguous-channel
exclusions and same-cell ownership checks remain shared with evaluation. Train the
U-Net in the PyTorch environment using an explicit `--index` and new `--output`.
Checkpoint/threshold selection uses validation data; held-out test data are for
reporting. Exported measurements must not be used as training truth without review.

New weights are not installed over bundled models automatically. Keep them external
until their data identity, retained-cell quality, volume and distance errors have
been reviewed. [Evaluation](evaluation.md) documents metrics and limitations.
