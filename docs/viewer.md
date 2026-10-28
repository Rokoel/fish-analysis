# Viewer

[Documentation](README.md) · [Analysis](fish-analysis.md)

Use `./run-viewer.sh` or `run-viewer.bat` for general image/scene inspection and
separate developer training tools. Auto-select contours opens Analysis, which
provides only consensus and fine-tuned StarDist 2D. The retired classical contour
pipeline is removed. Saved prediction/report overlays remain readable.

The Viewer supports calibrated XY/XZ/YZ sections, 3D surfaces and volumes, channel
visibility, display contrast and annotation visibility. Scan loading and neural
workers keep the Qt interface responsive. The source image/scene is read-only.
Analysis provides the full [selection/editing/export workflow](fish-analysis.md).

Display interpolation is enabled by default: bilinear in 2D and linear volume
interpolation in 3D. It smooths display transitions without changing measured
pixels. The reader's full-resolution level is used by Analysis; selecting a lower
stored resolution in the general Viewer is an explicit preview choice. CPU/GPU
execution and rendering can differ in speed between platforms, not source pixel
resolution. Zooming preserves the chosen location within Qt scrollbar limits.
