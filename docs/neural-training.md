# Training the retained networks

[Documentation](README.md) · [Data preparation](training-data.md)

Analysis exposes the fixed both-detector consensus pipeline. Developer training
remains separate: StarDist 3D proposals, center/box detection, the selectable
contour U-Nets and the signal-center U-Net support that workflow.
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
python -m fish_viewer.proposals --help
python -m fish_viewer.refinement --help
python scripts/build_signal_centers.py --help
python scripts/train_signal_centers.py --help
python scripts/evaluate_signal_centers.py --help
```

The newly bundled contour checkpoints are documented in the
[training/comparison report](../reports/contour-training-20261104/README.md).
Each completed run added 20 epochs, with two seeds per plain crop size. Shared
training/inference preprocessing and initial-checkpoint validation are consistent;
both selection modes retain epoch zero when no qualifying epoch improves it.
The initial checkpoint is actually loaded before scoring and training. New 96³
and 64³ model manifests retain the exact recipe and checkpoint identity.

Signal data-building accepts either a fresh supervision audit or a completed cell
prediction report plus its collection audit. Physical resampling, ambiguous-channel
exclusions and same-cell ownership checks remain shared with evaluation. Train the
U-Net in the PyTorch environment using an explicit `--index` and new `--output`.
Checkpoint/threshold selection uses validation data; held-out test data are for
reporting. Exported measurements must not be used as training truth without review.

New weights are not installed over bundled models automatically. Keep them external
until their data identity, retained-cell quality, volume and distance errors have
been reviewed. [Evaluation](evaluation.md) documents metrics and limitations.
