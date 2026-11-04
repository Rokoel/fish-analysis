# FISH Analysis v0.11.0

## Signal model choice

**Analysis settings → Signal model** now offers the original model and the newly
trained experimental U-Net with stain/channel-specific cutoffs. Both checkpoints
are bundled; the original remains the default, and your selection is remembered.
The new model found more test signals with more false positives: recall 93.0%
versus 90.1%, precision 81.4% versus 83.8%. It is an alternative for review,
not a replacement default.

## Review cells without detected signals

**Show removed cells** displays cells excluded for having no signals, in grey in
2D and 3D. They remain absent from Imaris and Excel exports. Adding a signal inside
a grey cell automatically includes it in exports and restores its normal colour.
Deleting or moving its last signal outside the cell excludes it again; undo and
redo restore the corresponding state. Selected review cells take precedence over
overlapping imported annotations when adding signals.

Saved analyses can restore these cells while their internal contour files remain
available. Cells rejected earlier by consensus or roundness are not restored.

## Downloads

Windows x64 EXE and Linux x64 AppImage include all six model bundles and are
published after native build and installed-application checks pass. Windows uses
CPU; Linux supports CPU/NVIDIA GPU inference with its verified neural runtime.
Original scans, training crops and experimental reports are not distributed.
