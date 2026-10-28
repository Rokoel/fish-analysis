# Evaluation and reports

[Documentation](README.md)

The local checkout contains exactly three report bundles:

1. The combined historical report (`reports/combined/`), consolidating fourteen old
   experiments and unique evidence. It is deliberately ignored by Git and excluded
   from release source bundles, so it is available only in local checkouts that
   generated it.
2. [Latest consensus roundness evaluation](../reports/consensus-roundness-20261028/README.md).
3. [Fine-tuned StarDist evaluation](../reports/stardist2d/README.md), including its
   signal-to-border distance evaluation in the same narrative.

Historical models, channels, thresholds and cohorts retain their original labels.
Do not pool different experiments or treat them as new current-default results.
In particular, repeated exploratory use of a test fold is not prospective validation.

## Metrics

Cell matches are one-to-one whole-object IoU ≥0.5. Unknown/unannotated objects stay
unscored, so unmatched detections do not establish field precision. Primary volume
errors exclude truncated manual cells. Distances compare actual manual A–B lines
with same-cell, same-channel detected signals matched within 0.6 µm. Ambiguous manual
assignments and conflicting A–B lines are excluded and counted. Missing measurements
are not invented. Conditional error metrics must be interpreted alongside retained
cell counts, manual-match counts and matched-subset overlap.

Reports include volume MAPE, distance MAE and other error summaries. Acquisition-group
bootstrap intervals preserve acquisition dependence. Completeness checks refuse to
publish a partial scan/pipeline or scan/roundness comparison. Model/input identities
and hashes are recorded. Changing a rejection threshold does not improve surviving
contours; common-subset comparisons separate rejection from geometry changes.

## Reusable commands

Supply a completed calibrated audit and explicit output locations:

```sh
python scripts/benchmark_all_heldout.py --audit /data/audit/audit.json --output /results/two-pipelines --device auto
python scripts/benchmark_stardist2d_distances.py --audit /data/audit/audit.json --output /results/stardist2d
python scripts/benchmark_consensus_roundness.py --audit /data/audit/audit.json --work /scratch/roundness --output /results/roundness
```

The roundness benchmark compares 50%, 75%, 80%, 82% and 85%; these research thresholds
can be lower than the UI's 75% minimum. An optional `--channel-audit` supplies reviewed
channel metadata. No command requires deleted historical training directories.
The held-out benchmark runs both supported pipelines at 82% by default. Only image
export is omitted; normal inference and measurement workers run unchanged.

`evaluate_cell_centers.py`, `evaluate_cell_contours.py` and `evaluate_signal_centers.py`
provide focused checks on explicit predictions/data. `audit_files.py --pixels`
validates scan decoding at all stored resolutions and times without changing sources.
