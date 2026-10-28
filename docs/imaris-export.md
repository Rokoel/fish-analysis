# Imaris export compatibility

[Documentation](README.md) · [Models](models.md) · [Analysis](fish-analysis.md)

**Export edited Imaris…** and the pipeline's final export produce exactly two
files: a standalone annotated `.ims` image and an Excel `.xlsx` workbook.
Open the IMS directly. No external IMX, JSON or CSV companion is required.
The workbook contains cell volumes and signal-to-border distances in
**Measurements**, plus **Signal pairs** when cross-channel pairs are present.

## Standalone scene

The IMS contains native `Scene/Content/SurfacesN` objects and `Scene/Data` XML,
following the layout observed in the supplied Imaris 7.4.2 annotated scans.
Vertices contain XYZ and normal XYZ; triangle indices are local to each surface.
`TimeNVerticesNTriangles`, `TimeInfos`, material attributes and unique surface IDs
link binary geometry to its XML viewer settings. Measurement points, independent
distance lines and unmeshed section contours are stored in the native XML.
Application/viewer defaults come from the packaged FISH Imaris 6.3.1 template;
imported native object settings are preserved where available.

Each signal-to-border measurement uses A and B points. Each accepted cross-channel
pair has its own two-point measurement line. Unrelated signals are never joined.
Channel-pair distances use calibrated physical XYZ coordinates and the configured
maximum separation within a cell/time point. Their Excel rows identify both
signals, both channels, the distance and the cutoff.

Stable annotation identities, cell assignments, channel associations, scores,
run provenance and pairing settings are embedded under `FISHAnalysis/Annotations`.
This metadata is bound to a checksum of the exact scene XML. Imported annotations
are pre-existing input to a new run, regardless of their names or old provenance.
New annotations receive a UTC creation timestamp in their exported names;
pre-existing names and timestamps remain intact.

## Image preservation

For HDF5 IMS input, the exporter copies the image and replaces scene data only in
the copy. All image datasets, channels, time points, resolutions, metadata,
compression and calibration remain unchanged. The whole-file checksum changes
because annotations are embedded. Legacy TIFF-based Imaris images are converted
to an HDF5 IMS containing the same channel/time voxel arrays and acquisition
metadata. Original images and scene companions are opened read-only and never
overwritten. Existing export files are protected against overwriting.

Unknown annotation schemas are reported and block export to avoid silently
losing objects. Before publishing, the exporter reopens the IMS without a
companion and checks coordinates, topology, times, labels and line connections.
Regression tests also cover relocation, repeated analysis/import/export, pairing,
workbook values and unchanged source voxels across all times and resolutions.
This example-derived compatibility profile is not an official vendor schema;
Imaris itself is unavailable here, so opening new exports in Imaris has not been
verified in this environment.

## Annotation-free test copy

The diagnostic `unannotated` command produces a standalone IMS without annotations:

```bash
.venv/bin/python -m fish_viewer.workflow unannotated \
  --image 'FISH/3abx3ad_101212/141212_3ab3ad_Lam633m_dsRed543_CG488b.ims' \
  --output 'artifacts/my-test/unannotated.ims'
```

This diagnostic does not create a measurement workbook. Source-file and scene
regression checks live in
[test_imaris_compatibility.py](../tests/test_imaris_compatibility.py) and
[test_signal_pairs.py](../tests/test_signal_pairs.py).
