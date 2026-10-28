# Retained models

[Documentation](README.md) · [Analysis](fish-analysis.md)

Only **Both-detector consensus** (default) and **Fine-tuned StarDist 2D** are
selectable. Five complete component bundles support these two pipelines:

| Bundle | Role |
| --- | --- |
| [cells-sensitive](../models/cells-sensitive/) | Consensus StarDist 3D proposals |
| [proposals](../models/proposals/) | Independent center/box detector for consensus |
| [refinement-bias-candidate](../models/refinement-bias-candidate/) | Shared contour U-Net for both consensus detectors |
| [cells-stardist2d](../models/cells-stardist2d/) | Fine-tuned XY StarDist polygons and adjacent-section linking |
| [signals-center-unet](../models/signals-center-unet/) | Signal-center 3D U-Net used by both pipelines |

The checkpoints and manifests are preserved byte-for-byte. [origins.json](../models/origins.json)
records their hashes and historical source locations. Training paths inside a
manifest describe provenance; inference does not read the deleted training archive.
The StarDist 2D bundle retains the required pretrained-model license.

No additional old model bundles, candidate-CNN runtime, Spotiflow runtime, classical
segmentation UI, recovery, signal scouting or membrane snapping is required.
Both pipelines apply the same 82% roundness default and signal/measurement stages.
See [consensus](cell-consensus.md) and [StarDist 2D](stardist2d.md).
