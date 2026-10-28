# FISH Analysis

[Documentation](README.md) · [Models](models.md) · [Export compatibility](imaris-export.md)

FISH Analysis detects cells and signals, measures physical 3D distances and volumes,
and exports a standalone annotated Imaris image and an Excel workbook.

## Installation

Use Python 3.12 for source installations. Run `./setup-analysis.sh`, then
`./run-analysis.sh` on Linux; run `setup-analysis.bat`, then `run-analysis.bat`
on Windows. Setup installs the separate PyTorch signal runtime. The five required
component model bundles are included in the repository. Native Windows runs the
TensorFlow cell models on CPU; Linux supports CPU and a configured NVIDIA GPU.
Both platforms read the same image pixels and use the same physical calibration.

Standalone Windows EXE and Linux AppImage release builds include the application
and models. Linux installs a verified neural runtime into its local cache on first
launch; Windows bundles its CPU runtime. See [releasing](releasing.md) for build,
runtime and platform details.

## Processing

1. Add scans or a folder. Select scans to process.
2. Preview each scan. Open **Channels…** to choose its cell channel and one or more
   signal channels. Choices persist per file. **Check channels** prompts review
   when metadata does not unambiguously identify the membrane stain.
3. Choose an output folder, pipeline and device under **Pipeline and settings**.
4. Click **Analyze selected**. Scans run sequentially. Cancel preserves finished
   results; **Select unfinished scans** prepares pending, failed and cancelled scans
   for retry. A failed scan retains its error and the queue continues.

The processing summary shows the selected pipeline, device and channel roles.
There are exactly two choices:

| Pipeline | Behavior |
| --- | --- |
| **Both-detector consensus (default)** | Sensitive StarDist 3D proposals are redrawn by the contour U-Net. A separate center/box detector produces contours with the same U-Net. One-to-one overlap at IoU ≥0.5 retains whole redrawn StarDist contours detected by both approaches. |
| **Fine-tuned StarDist 2D** | Membrane-preprocessed XY images produce polygons which are linked between adjacent sections into 3D cells. |

Both use the signal-center 3D U-Net. Saved retired pipeline selections migrate to
consensus. Component checkpoints are fixed; Analysis has no training/model-picker
controls. Training remains in the separate developer tools.

**Minimum roundness** defaults to **82%**, with a UI range of **75–100%**.
It is calibrated 3D sphericity `π^(1/3) (6V)^(2/3) / A` on the voxel mask.
Raising the gate rejects irregular cells without changing contours that survive.
Saved values below 75% reset to 82%. The gate runs before signal detection.

## Measurements and exports

Signals are detected separately in the selected signal channels. A cell survives
when at least one accepted signal center lies inside it. Volumes use closed meshes.
For signal A, B is the closest point anywhere on its cell mesh in physical XYZ;
the A–B length is the signal-to-border distance.

For each pair of different signal channels within the same cell and time point,
the closest available signals are paired first, with each signal used at most once
for that channel pair. A signal can pair again with a third channel.
**Signal-pair limit** sets the maximum physical separation, initially **1 µm**;
longer pairs are omitted. Pair lines and their measured distances are included in
both exported formats.

Export produces exactly two files: annotated **.ims** and **.xlsx**. Open the IMS
directly; it contains pixels and native annotations. Excel's **Measurements** sheet
lists cell volumes, signal channels and border distances. **Signal pairs** is added
when pairs exist. A cell's repeated volume rows must not be summed. Internal worker
reports and prediction archives support reopening runs but are not companion
export files. See [Imaris compatibility](imaris-export.md).

## Imported annotations and editing

Open `.ims` files with or without embedded annotations, or an `.imx` scene with
its referenced image beside it. Imported surfaces, drawn section contours,
signal centers and A–B lines appear alongside new detections. Opening an IMS
with embedded objects uses its own annotations; explicitly opening an IMX uses
that saved scene, including an intentionally empty scene.
The annotation list groups objects as **Pre-existing** and **This run**; each
group has a visibility checkbox. All imported objects are pre-existing, including
results from previous runs. Opening a saved analysis result activates only that
result's run. Object names never determine these groups. Editing an imported
object keeps it in Pre-existing. Hiding a group
does not delete it or exclude it from export.

Select an object in the annotation list or click its border/marker in 2D.
Selecting a cell in the annotation list or **Cell counts** moves the 2D slice
to the nearest visible cross-section if the cell is outside the current slice.
The current slice stays in place when the cell is already visible. This works
in XY, XZ and YZ without changing zoom; clicking the same selected cell again
also brings it back into view after scrolling away.
Selecting a **Measurement** similarly reveals its signal center (A), keeping
the current slice and zoom when A is already visible. A visible B point or line
alone does not replace locating A. This also works for single-point signals
and when clicking an already selected Measurement after scrolling away.

The image toolbar provides these contextual tools:

- **Select** starts active. Click a visible border/marker to select an object,
  **Shift+click** to add one, drag a rectangle to select an area, or
  **Shift+drag** to add an area. Only geometry drawn on the current slice can be
  picked. Selecting any part of a cell selects that entire cell across slices.
  Click a dot on an already selected contour to select that point; Shift adds
  points, and a rectangle around that contour's dots selects several points.
  Select never moves geometry.
- **Edit selection** moves selected objects together by dragging or by calibrated
  XYZ offsets. Signals assigned to moved cells follow once, even when selected
  separately. For a single cell contour, this tool shows precise point controls.
- **Precise editing** moves selected white dots by dragging or by calibrated XYZ
  coordinates. Dots show only actual intersections with the current slice.
  **Delete selected points** removes those points; **Delete selected objects**
  removes whole objects. **Delete** acts on selected points when present,
  otherwise on selected objects. Unsafe mesh-point deletion is rejected.
- **Add new** offers **Signal**, **Contour point**, and **Layered cell contour**.
  A signal click places A in the current slice. To add a point, select one cell
  and click its visible border. To draw a new cell, click a polygon and press
  **Enter** or **Close layer** (double-click also closes it). Change slice and
  draw more layers in the same orientation, then choose **Create cell from
  layers**. Signed-distance interpolation fills the space between drawn layers
  and caps the ends to make a closed mesh. One layer is allowed; more layers
  give control over the cell's 3D shape. **Esc** discards the unfinished drawing.
- **Simplify contours** reduces selected polygon/mesh point counts. For meshes,
  it preserves closed topology and checks the **Max volume change** limit
  (default 5%), reducing the requested simplification if necessary. Review the
  resulting border: the volume limit alone does not bound local shape changes.
  Cells that cannot meet the limits are skipped while eligible cells continue.
  The sidebar lists the simplified and skipped cells, their point counts and
  reasons for skipping. Successful changes share one undo step; if every cell
  is skipped, no geometry or undo history changes.
- **Proportional editing** (also called soft selection) moves nearby mesh points
  within a physical 3D sphere, including points on neighboring slices.
  **Influence radius (µm)** controls its size; **Falloff** chooses Smooth or
  Linear tapering to zero at its edge. A dashed circle shows that sphere's
  intersection with the current slice. Switch to **Precise editing** to move
  only selected points.

Use middle-drag or **Space+drag** to pan and **Ctrl+wheel** to zoom.
**Ctrl+Z** undoes an operation; the **Redo** button restores it. Group operations
are a single undo step. B points are calculated from the signal center and its
assigned surface and are not independently moved.

Dragging a slice border makes its intersections explicit mesh points, keeping
the other points on that slice fixed, and moves the selected point as one
undoable edit. Connected border lines follow the dot while dragging.
Adding/removing a surface point maintains the connected triangle
mesh; an unsafe deletion is rejected. Cell volumes and affected signal distances
update immediately. Original manual A–B measurements remain unchanged until the
corresponding signal or cell is edited. Signals with no assigned cell retain
their centers but have no fabricated border distance. Deleting a cell also
deletes its assigned signals and any unassigned signals enclosed by it, including
their B points and lines. Signals assigned to another overlapping cell keep
their own association.
Undo restores the cell, signals and assignments together.

**Export edited Imaris…** saves exactly two files: a standalone annotated `.ims`
and an Excel `.xlsx` workbook. Native scene geometry, UUIDs, exact cell assignments
and run provenance are embedded in the IMS; no companion scene, CSV or JSON file
is needed. New annotations receive a UTC creation date and time
in their exported names; imported names and existing timestamps are preserved.
Reanalysis creates new identities even when the detector reuses local cell or
signal numbers. Manually moved centers have no stale neural detection score. Open the **`.ims`**
to load both the image and annotations. Originals are never overwritten;
embedded stale scenes are replaced in the exported copy, while all image
channels, time points and pyramid data are preserved. Unsupported annotation
schemas are reported and prevent export rather than silently losing objects.
Import opens the source image and scene read-only and loads image arrays and
editable annotations into memory. It does not create a working disk copy at
import time. Display adjustments and annotation edits affect that in-memory
state; export creates the new image/scene files. Existing export filenames are
also protected against overwriting.
Edits and undo histories survive switching scans in the same app session;
export them before closing the app.

**Smooth**, beside **Channels…**, is enabled by default and remembers its setting.
It uses bilinear image interpolation in 2D and trilinear voxel interpolation in
3D; turn it off for nearest-neighbor display. This changes display only, leaving
scan pixels and measurements untouched. Surfaces use the actual mesh with smooth
shading. Selected objects use thick yellow outlines/markers and contrasting list
text. The **Cell / signal channels…** button is next to the analysis
controls; it opens the existing role and display settings.

**Auto display** sets each channel's black/white range and gamma from a bounded
sample of the volume. It estimates background separately from bright foreground,
so sparse puncta are not lost in a global percentile and isolated hot pixels
do not dominate the white level. Gamma gently lifts dim fluorescence. The same
adjustment runs when loading a scan and through each channel's **Auto** button.
Manual brightness, contrast, gamma and range controls remain available under
**Channels…**. These adjustments affect display only; raw intensities used by
the pipeline and exports stay unchanged.

The Analysis 2D view loads resolution level 0 and keeps one image pixel per
source voxel in the current plane; fitting the scan to the window changes only
its screen size. The screen composite is 8-bit RGB, while source intensities
remain in their original arrays. The 3D view defaults to a preview with at most
256 samples per axis. Enable **Channels… → Full volume detail in 3D** to supply
all voxels to the volume renderer. These spatial-resolution rules are the same
on Windows and Linux; rendering speed and GPU limits can differ.

Ctrl+wheel zoom stays anchored at the cursor and stops at extreme magnification
(256 screen pixels per voxel), before the view's integer scrollbars can overflow.
Increasing screen magnification cannot recover detail finer than the source
voxels. The detectors separately resample images onto their trained physical
grids (0.3 µm for cell proposals, 0.15 µm for contour refinement and signal crops,
retaining coarser native axial spacing where required); their working resolution
does not depend on the current display zoom or 3D preview setting.

On Windows, run `setup-analysis.bat` when installing or updating this release.
It automatically downloads the official Microsoft Visual C++ x64 runtime,
checks its Microsoft signature, and installs it before setting up TensorFlow.
Internet access and Windows administrator approval for the Microsoft installer
may be required. If setup reports a required restart, restart before analysis.
The release includes this setup integration; it does not redistribute Microsoft's
installer binary. See [Microsoft's runtime documentation](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170).
