# Signal centers

[Documentation](README.md) · [Analysis](fish-analysis.md)

Both pipelines use `signals-center-unet`, a PyTorch 3D U-Net with signal channel,
membrane channel and target-cell mask inputs. Each cell/channel crop is resampled
at the checkpoint's physical spacing. Learned heatmap peaks use subvoxel quadratic
localization and calibrated physical suppression within the same cell/channel.
Signals can lie near a cell boundary. The app does not require an eroded interior.

Only selected signal channels are processed. A cell is retained if any selected
channel has an accepted center inside it. Border distances, mesh volumes and
cross-channel pairs are recalculated after edits. Pairing uses closest-first
one-to-one matches separately for each channel pair, within a cell/time point and
the configurable physical cutoff (initially 1 µm).

Training treats unpainted/unknown regions as unknown, not background. Only eligible
marked-cell regions contribute to evaluation; channels with ambiguous manual A
assignments are excluded. Validation selects the score threshold and checkpoint.
The old candidate classifier and Spotiflow implementations/models have been removed.
Saved annotations from those versions remain readable.

Use explicit inputs for [building new supervision and training](neural-training.md).
The signal worker dispatches through the configured separate PyTorch environment;
`FISH_SPOTS_PYTHON` can override it in source installations. Standalone builds use
their own bundled runtime.
