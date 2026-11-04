# Both-detector consensus

[Documentation](README.md) · [Models](models.md)

Consensus is the only analysis pipeline. Its name is fixed information; the
**Contour model** dropdown selects **96³ (default)** or **64³ — longer-trained**.
The sensitive StarDist 3D detector
keeps its saved proposal thresholds. Its proposed cells are redrawn and checked
by the selected contour U-Net. Independently, the center/box detector supplies
padded crops to the same contour U-Net. One-to-one whole-object IoU matching at
0.5 retains the redrawn StarDist contour when both detectors agree. Matching does
not average or intersect the two meshes.

The accepted cells then pass the physical roundness gate (82% default), signal
U-Net, any-signal retention and common measurement/export stages. Contours that
survive a stricter roundness gate retain their geometry. Consensus and sphericity
can discard valid cells; sparse manual annotations do not determine the precision
of unmatched detections.

See the [new contour training comparison](../reports/contour-training-20261104/README.md).
The [historical roundness evaluation](../reports/consensus-roundness-20261028/README.md)
used the retired contour checkpoint and is not a new-model measurement. Four
networks execute for either contour choice; five bundles distribute both choices,
as listed in [models](models.md).
