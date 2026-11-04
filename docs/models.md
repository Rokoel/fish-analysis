# Bundled models

[Documentation](README.md) · [Analysis](fish-analysis.md)

**Both-detector consensus** is the only pipeline. The **Contour model** dropdown
chooses the U-Net used by both detectors; its preference is remembered.

| Bundle | Role |
| --- | --- |
| [cells-sensitive](../models/cells-sensitive/) | Consensus StarDist 3D proposals |
| [proposals](../models/proposals/) | Independent center/box detector for consensus |
| [contours-96](../models/contours-96/) | Default shared 96³ contour U-Net |
| [contours-64](../models/contours-64/) | Shared longer-trained 64³ contour U-Net |
| [signals-center-unet](../models/signals-center-unet/) | Signal-center 3D U-Net |
| [signals-center-unet-improved](../models/signals-center-unet-improved/) | Optional experimental signal U-Net with stain/channel cutoffs |

Both contour checkpoints select seed 42 and epoch 19 of an additional 20-epoch run.
They use the same architecture, threshold 0.55, raw normalized membrane plus Gaussian
seed inputs, and no CLAHE or experimental microscopy augmentation. 96³ samples
the same physical crop more finely; 64³ uses less computation.

[origins.json](../models/origins.json) records every bundled file's hash. Each new
manifest records the parent checkpoint, configuration and validation metrics.
Inference requires no training caches. Analysis results record the chosen contour
crop size and checkpoint hash. The [training report](../reports/contour-training-20261104/README.md)
contains all completed seeds, selection rules, curves and test results.

No additional old model bundles, candidate-CNN runtime, Spotiflow runtime, classical
segmentation UI, recovery, signal scouting or membrane snapping is required.
The pipeline applies an 82% roundness default and common signal/measurement stages.
Saved retired pipeline preferences migrate to consensus; valid 96³/64³ contour
preferences survive later launches. See [consensus](cell-consensus.md).
