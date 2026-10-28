# FISH Analysis

Desktop analysis of Imaris microscopy scans: detect cells and fluorescent signals,
review and edit their annotations, and measure physical 3D cell volumes,
signal-to-border distances and distances between nearby signals of different colors.
Original scans and annotations are opened read-only.

**[Download v0.9.0](https://github.com/Rokoel/fish-analysis/releases/tag/v0.9.0)** ·
**[Analysis guide](docs/fish-analysis.md)** · **[Documentation](docs/README.md)** ·
**[Models and evaluation](docs/models.md)**

## Get started

Download the Windows x64 **EXE** or Linux x64 **AppImage** from the release.
Windows uses CPU inference. Linux supports CPU or NVIDIA GPU inference and installs
its verified neural runtime on first launch. See [installation and runtime details](docs/releasing.md).

1. Add an IMS scan, an IMX scene with its image, or a folder of scans.
2. Check the cell channel and select the signal channels.
3. Choose an output folder and run **Both-detector consensus**, the default pipeline.
4. Review cells, signals and measurements. Export creates exactly two files:
   a standalone annotated **IMS** image and an **Excel workbook**.

The other supported pipeline is **Fine-tuned StarDist 2D**. Both pipelines use the
same signal-center U-Net. Minimum roundness defaults to **82%**, with a selectable
range of **75–100%**. A higher value rejects irregular cells; it leaves surviving
contours unchanged.

## Consensus results at 82% roundness

The [roundness evaluation](reports/consensus-roundness-20261028/README.md) processed
**71 historical test scans** with frozen checkpoints and reviewed cell channels:

| Result | At 82% roundness |
| --- | ---: |
| Cells passing roundness, before signal filtering | 3,101 |
| Retained cells containing signals | 1,031 |
| Detected signals in retained cells | 1,063 |
| Complete manual cells matched after signal filtering | 182 / 508 |
| Mean absolute volume error, relative to manual volume | 5.56% across 182 matched cells |
| Mean absolute signal-to-border distance error | 0.115 µm across 166 matched signals |
| Median cell overlap with manual contours (IoU) | 0.880 |

Volume and distance errors exclude cut manual cells and are conditional on matches;
unmatched detections have incomplete manual labels and do not establish overall
precision. The test fold was previously used in exploratory comparisons, so this
is not independent validation of the chosen threshold. See the full report for
all five roundness values, error tails and acquisition-bootstrap intervals, or the
[separate StarDist 2D report](reports/stardist2d/README.md).

## Review, edit and measure

- Imported annotations and annotations created in the current run have independent
  visibility controls. UUIDs and run provenance support repeated import/export and
  reanalysis without relying on object names.
- Select a cell or signal to reveal its nearest visible 2D section. Display
  interpolation starts enabled, contrast adapts to sparse fluorescence, and zoom
  stays anchored at high magnification.
- Precise and proportional editing, layered contours, selection movement,
  simplification with per-cell skip reporting, and undo support contour review.
  Edits recalculate volumes and affected distances. Deleting a cell also removes
  its associated signals.
- Different-channel signals are paired closest-first within each cell and time
  point. A configurable physical cutoff, initially **1 µm**, excludes distant pairs.
  Pair distances are exported in the IMS and the workbook's **Signal pairs** sheet.

The editing sidebar starts hidden. Export preserves pre-existing names and dates;
new annotations receive creation timestamps. Excel includes cell volumes and
signal-to-border distances in **Measurements**, plus signal pairs when present.
See [editing and measurements](docs/fish-analysis.md) and
[Imaris compatibility](docs/imaris-export.md).

## Source installation and development

Use Python 3.12. On Linux, run `./setup-analysis.sh` then `./run-analysis.sh`;
on Windows, run `setup-analysis.bat` then `run-analysis.bat`.
Setup installs the separate PyTorch signal runtime. The repository includes the
five component model bundles required by the two analysis pipelines.

`./run-viewer.sh` opens the general Viewer and developer training tools.
`./run-annotator.sh --audit /absolute/path/audit.json --output /absolute/path/annotations`
opens the training annotator. Future training and evaluation require explicit data
paths; historical training metadata and painted annotations are no longer bundled.
The combined historical report stays local and is excluded from Git and releases;
the latest consensus roundness and StarDist reports remain public.

Run core tests with `python -m pytest -p no:cacheprovider -m "not neural and not gui and not integration"`.
GUI tests require Qt/VTK or `QT_QPA_PLATFORM=offscreen`; neural tests require the
TensorFlow/StarDist and configured PyTorch runtimes. See [architecture](docs/architecture.md),
[training](docs/neural-training.md) and [release checks](docs/releasing.md).
