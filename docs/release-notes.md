# FISH Analysis v0.9.1

This release focuses on more selective consensus detection, reliable imported
annotations, contour editing, cross-channel measurements and standalone IMS/Excel
export. **Both-detector consensus** remains the default, with **82% roundness**
and a minimum selectable roundness of **75%**. **Fine-tuned StarDist 2D** is the
other supported pipeline; both use the signal-center 3D U-Net.

## Consensus benchmark at 82% roundness

Across **71 historical test scans**, with frozen checkpoints and reviewed cell
channels:

| Result | At 82% roundness |
| --- | ---: |
| Cells passing roundness, before signal filtering | 3,101 |
| Retained cells containing signals | 1,031 |
| Detected signals in retained cells | 1,063 |
| Complete manual cells matched after signal filtering | 182 / 508 |
| Volume mean absolute percentage error | 5.56% over 182 matched cells |
| Signal-to-border distance mean absolute error | 0.115 µm over 166 matched signals |
| Median matched-cell IoU | 0.880 |
| 95th-percentile absolute volume percentage error | 14.90% |
| 95th-percentile absolute distance error | 0.359 µm |

The acquisition-bootstrap 95% interval is **3.58–7.91%** for mean absolute volume
percentage error and **0.095–0.137 µm** for mean absolute distance error.
Compared with the historical 50% gate, the 82% gate retains fewer cells
(1,031 versus 1,558); the matched-subset volume error is 5.56% versus 8.67%, and
distance error is 0.115 versus 0.131 µm. These comparisons involve different
surviving subsets. Raising roundness rejects cells without improving the contours
of cells that remain.

Matches require one-to-one cell IoU ≥0.5; signal centers must match within 0.6 µm
in the same cell and channel. Volume and distance errors exclude cut manual cells,
retain outliers, and use actual manual A–B lines for distance references. They are
conditional on matches. Sparse annotations cannot establish the overall precision
of unmatched detections. This historically reused test fold is exploratory, not
independent validation of the default threshold.

[Full roundness report and raw results](https://github.com/Rokoel/fish-analysis/blob/v0.9.1/reports/consensus-roundness-20261028/README.md) ·
[Fine-tuned StarDist report](https://github.com/Rokoel/fish-analysis/blob/v0.9.1/reports/stardist2d/README.md)

## Annotation and export fixes

- Correct native imported IMS mesh coordinates, normals and triangle connectivity,
  and preserve independent measurement-line connections.
- Use UUIDs and current-run provenance for annotations. Imported results from any
  previous run are pre-existing, regardless of their names. Reanalysis creates
  new identities instead of reusing local detector IDs.
- Open source images and scenes read-only. Editing changes in-memory annotations;
  export writes a new image copy and preserves source voxels.
- Export exactly a standalone annotated **IMS** and an **Excel workbook**. Native
  scene geometry, identities and assignments are embedded in the IMS. New
  annotations receive UTC creation timestamps; imported names/dates are preserved.

## Editing and display

- Default display interpolation smooths 2D and 3D image transitions. Auto display
  better handles sparse puncta and dim fluorescence without changing raw pixels.
- Selecting a cell or signal reveals its nearest visible slice. Selected objects
  and list text have clearer contrast. Extreme zoom stays anchored.
- The annotation/editing sidebar starts hidden. Contextual tools support selection,
  precise editing, proportional editing, new layered contours and simplification.
- Simplification processes eligible selected cells and reports which were skipped.
  Contour changes update volumes and affected distances. Deleting a cell removes
  its associated signals; undo restores them together. Stale selections are handled
  safely after creation or deletion.

## Cross-channel signal pairs

Choose multiple signal channels and measure closest-first pairs within each cell
and time point. Each signal is used once per channel pair. A configurable physical
cutoff, initially **1 µm**, omits distant pairs. Pair lines and distances appear in
the IMS and the workbook's **Signal pairs** sheet. Moving/deleting signals, editing
cells, changing the cutoff and undo update these derived measurements.

## Smaller supported codebase

Only the two supported analysis pipelines and their five required component model
bundles remain. Retired models and implementation branches, temporary experiments,
obsolete documentation images and archived training annotations/metadata are
removed. Reusable training and evaluation tools accept explicit inputs.

Historical reports are consolidated into one local report, excluded from Git and
release bundles. The latest consensus roundness report and fine-tuned StarDist
report remain public, preserving their original cohorts and parameters.

## Windows report portability

Generated roundness reports and their CSV tables now use explicit UTF-8 encoding,
including on Windows systems whose default encoding cannot represent scientific
symbols. A regression test reproduces that Windows encoding before checking the fix.

## Verification and downloads

Local verification passed **348 tests**, with three skipped in the main environment,
plus **nine signal-network tests** in the separate PyTorch runtime. All five model
bundles loaded and ran CPU inference. Both supported pipelines completed real-scan
IMS/XLSX exports and reopening checks; original scan checksums and retained model
checkpoint hashes were preserved.

Download the Windows x64 **EXE** or Linux x64 **AppImage** below. Windows uses CPU
inference. Linux supports CPU/NVIDIA GPU inference and installs its verified neural
runtime on first launch. Release automation builds and checks each platform's
installed application before publishing its binaries and checksums.

The native IMS export profile is based on the supplied annotated Imaris files;
exports were checked by reopening them in this application. Opening these new
exports in Imaris itself has not been verified in this environment.
