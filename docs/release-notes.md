# FISH Analysis v0.10.0

Both-detector consensus is now the app's only analysis pipeline. Its name appears
as fixed information, with no pipeline chooser. The **Contour model** dropdown
selects **96³ (default)** or **64³ — longer-trained**; the same selected U-Net draws
contours for both detector branches. The choice persists across launches.

## Newly bundled contour models

Both new checkpoints are seed 42, epoch 19 of an additional 20-epoch fine-tuning
run. Two seeds completed for each plain crop size, plus one CLAHE control. Remaining
runs were cancelled at the user's request. Checkpoint and seed selection used
validation data; the 96³ default was frozen before the final test comparison.

| Contour checkpoint | Test surface error ↓ | Test volume MAPE ↓ | Mean IoU ↑ |
| --- | ---: | ---: | ---: |
| Previous selected 64³ | 0.1130 µm | 4.52% | 0.8619 |
| New 64³ | **0.1080 µm** | **4.18%** | **0.8665** |
| New 96³ (default) | 0.1096 µm | 4.26% | 0.8652 |

Each new model matched all 508 complete manual test cells at IoU ≥0.5. The 64³
point estimates are slightly better; direct 96³–64³ acquisition-bootstrap intervals
include zero. The selected default is not a claim of proven superiority. The test
uses manual boxes/interior seeds and historically reused folds. It isolates contour
drawing and does not establish end-to-end consensus recall or signal-distance
accuracy. Old full-consensus results retain their original model identity.

The release includes a **training-comparison archive** with the report, all five
training histories, per-epoch and per-cell CSVs, curves, checkpoint identities,
original protocol, cancellation amendment and bootstrap intervals. See the
[full report](https://github.com/Rokoel/fish-analysis/blob/v0.10.0/reports/contour-training-20261104/README.md).

## Reliability and migration

- Fine-tuning loads initial checkpoint weights before validation/training.
- Initial and per-epoch validation share preprocessing, threshold and scored identities.
- Both checkpoint-selection modes retain epoch zero if training does not improve it.
- Saved StarDist 2D or other retired pipeline choices migrate to consensus.
- Analysis provenance records the contour crop size and exact checkpoint hash.
- StarDist 2D, the previous contour bundle, obsolete scripts/tests and its redundant
  reports are removed. Shared readers, geometry, training and evaluation remain.

Roundness remains 82% by default (75–100% selectable). Signal detection, editing,
signal pairing and standalone annotated IMS/Excel export retain their workflows.

## Downloads

Windows x64 EXE and Linux x64 AppImage are published after native build and installed
application checks pass. Windows uses CPU; Linux supports CPU/NVIDIA GPU inference
with its verified neural runtime. The release includes model hashes, build metadata,
checksums and the training-comparison archive. Original microscopy data are excluded.
