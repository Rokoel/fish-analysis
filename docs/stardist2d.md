# Fine-tuned StarDist 2D

[Documentation](README.md) · [Models](models.md)

This is the second supported analysis pipeline. The saved model preprocesses the
cell channel using the membrane-connected recipe, predicts XY polygons and links
overlapping detections on adjacent sections into 3D instances. It keeps the saved
probability, NMS and linking parameters. The separate experimental preprocessed
3D pipeline has been removed; its shared preprocessing implementation remains
because this 2D model requires it.

The same physical roundness gate, signal-center U-Net, retention, measurements and
IMS/XLSX exports follow. Read the [fine-tuned StarDist report](../reports/stardist2d/README.md)
for the historical test cohort and its original thresholds; historical reported
measurements are not new results at the current 82% default.

The model was fine-tuned from StarDist's `2D_versatile_fluo`; the pretrained BSD
license remains inside its bundle. The fine-tuning script accepts an explicit
collection audit and can recreate the official download cache. A new model must
include its acceptance configuration, preprocessing recipe and linking recipe.
See [training](neural-training.md).
