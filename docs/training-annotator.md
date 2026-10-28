# Training annotator

[Documentation](README.md) · [Training data](training-data.md)

The separate annotator paints training-only crops from an explicit collection audit:

```sh
./run-annotator.sh --audit /data/audit/audit.json --output /data/new-annotations
```

Both paths are required. `--fish-root` relocates the original scans when the audit
was created on another machine. The annotator samples the training split and saves
painted labels separately from source scans. Zero/unpainted means unknown, never
background. Undo, filling and crop persistence remain available. Keep acquisition
splits and original physical calibration when building later training sets.

Historical painted annotations and metadata are no longer bundled. Choose a fresh
external output directory for each annotation collection. Opening and painting do
not modify the original IMS/IMX files. For editing analysis results, use the
[analysis contour tools](fish-analysis.md#imported-annotations-and-editing) instead.
