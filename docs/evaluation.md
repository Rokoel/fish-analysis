# Evaluation and reports

[Documentation](README.md)

The current [contour training/comparison report](../reports/contour-training-20261104/README.md)
contains the new 96³ and longer-trained 64³ checkpoints, five complete training
histories, selection rules, test-cell CSVs, training curves and acquisition-bootstrap
intervals. Its archive is attached to the v0.10.0 release.

The [historical consensus roundness report](../reports/consensus-roundness-20261028/README.md)
uses the retired contour model. It preserves the original cohorts and parameters;
its full-pipeline measurements are not results for the newly bundled models.
Older StarDist 2D evidence remains in the v0.9.2 release history. The local combined
archive is ignored by Git and excluded from releases.

## Interpretation

The new comparison scores all 508 complete manual test cells using manual boxes
and interior seeds, mapped to a common calibrated grid. Empty predictions and
IoU <0.5 failures stay in scores. Surface error is mean symmetric voxel-boundary
distance; volume error is MAPE relative to manual volume. Acquisition-group
bootstrap intervals preserve acquisition dependence. Reused historical folds and
the limited number of seeds make this exploratory evidence.

Contour-only accuracy does not establish detector recall, field precision,
full-consensus yield or signal-distance accuracy. For whole-pipeline evaluation,
matches are one-to-one IoU ≥0.5; primary volume errors exclude cut manual cells.
Signal distances use actual manual A–B lines, with same-cell/channel matching within
0.6 µm. Unknown/unannotated objects remain unscored. Conditional errors must be
read alongside retained-cell counts and matched-subset overlap.

## Tools

```sh
python scripts/benchmark_consensus_roundness.py --audit /data/audit/audit.json --work /scratch/roundness --output /results/roundness
```

This evaluates the currently bundled default model at research roundness gates
50%, 75%, 80%, 82% and 85%, omitting only image export. It requires explicit inputs
and never reads a deleted historical training cache. Optional `--channel-audit`
supplies reviewed channels. Focused `evaluate_cell_centers.py`,
`evaluate_cell_contours.py` and `evaluate_signal_centers.py` accept explicit
predictions/data. `audit_files.py --pixels` checks original scan decoding read-only.
