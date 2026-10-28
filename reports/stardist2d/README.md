# Official StarDist2D fluorescence fine-tuning

Official model: [2D_versatile_fluo](https://github.com/stardist/stardist-models/releases/tag/v0.1). Archive checksum verified against the installed official StarDist registry.

Fine-tuned 40 epochs; best validation epoch 40. Training scans: 215; validation: 73; held-out test: 71.

Acquisition-safe XY sections of complete 3D manual cells; other real pixels unknown=-1; synthetic blank/noise=0. No held-out optimization.

Input is the corrected single cell intensity channel, enhanced with membrane-connected-v1 and enlarged 2x in XY. Up to seven sections per annotated complete cell; all 1,482 eligible training cells included. Neither signal channels nor RGB display colors are network inputs.

Network architecture and initial weights are the official 32-ray, depth-3 2D U-Net. All weights fine-tuned at 3e-5. Unknown real pixels have disabled probability loss; distance background regularization is zero. Synthetic blanks/noise provide limited negative supervision.

Adjacent XY polygons are joined by one-to-one Hungarian IoU matching (minimum 0.2). Missing slices break tracks; no gap filling or contour dilation. The 2D models use the preprocessed 3D model’s raw membrane-evidence and physical size acceptance settings; the conservative 3D baseline keeps its original settings. No roundness or signal-retention filtering. The 2D experiment uses the unchanged official probability threshold and NMS 0.3, with no held-out threshold tuning.

| Method | All contours | Manual matches / 505 | Volume MAPE | Surface error µm |
|---|---:|---:|---:|---:|
| pretrained | 4 | 0 | — | — |
| finetuned | 1399 | 197 | 7.72% | 0.245 |
| conservative | 1153 | 200 | 11.98% | 0.260 |
| preprocessed | 2118 | 265 | 11.19% | 0.263 |

Volume and surface errors are conditional on whole-cell IoU≥0.5. Different methods can match different cells; comparison.json includes pairwise intersections and the intersection across all four methods. Sparse annotations cannot establish full-field precision. All outliers included.

This is a 2D experiment with a separate 3D linking algorithm, not a pretrained 3D model. It can fragment cells where 2D sections disappear. StarDist polygons still require star-convex shapes per section. The model is trained originally for fluorescent nuclei, while FISH contains membrane-stained cells, so domain shift remains.

The signal-center U-Net is unchanged. Fine-tuned 2D is now the Analysis default; the 3D presets remain available. The default-setting signal-to-border benchmark is in [distance measurements](#signal-to-border-distance-evaluation).

Checkpoint: models/cells-stardist2d/best.weights.h5. Select **Fine-tuned StarDist 2D (default)** in the analysis app to use it with the existing signal and measurement pipeline.

![Comparison](comparison.png)

![Held-out examples](heldout-examples.png)

On the 163 manual cells matched by both fine-tuned 2D and preprocessed 3D, volume MAPE is **7.39% versus 11.16%**, and mean symmetric surface error is **0.244 versus 0.259 µm**. This supports a contour-accuracy improvement on the shared matched population. Fine-tuned 2D matches only **197/505 (39.0%)**, versus **265/505 (52.5%)** for preprocessed 3D. Its mean per-cell HD95 is worse (**0.871 versus 0.692 µm**), so lower average error does not mean uniformly better boundaries. These limitations remain despite its selection as the new app default. Signal-to-border accuracy is reported separately in the default-setting distance benchmark.

Fine-tuned 2D signed mean volume error is **+2.03%** (85 of 197 volumes are smaller than manual), so the average volume shrinkage is not present in this matched population. This does not rule out local inward boundary errors.

Verification: 219 non-integration tests passed, including batched 2D inference equivalence and zero unknown-pixel loss gradients. All six synthetic no-cell controls accepted zero contours. A standalone CPU application to an unannotated Imaris scan with explicit cell channel 1 produced 18 calibrated contour objects.


## Signal-to-border distance evaluation

# Fine-tuned StarDist 2D default: signal distances

All **71 held-out test scans** were processed with frozen 2D contours, **minimum roundness 0.72**, the unchanged signal-center U-Net, any-signal cell retention and the production distance-measurement code. No threshold tuning on test. Distances are shortest Euclidean distances from predicted signal centers to the calibrated 3D cell mesh, in micrometers.

| Measurement | Result |
|---|---:|
| Signals matched to manual A–B measurements | 112 |
| Distance MAE | 0.175 µm |
| Distance RMSE | 0.247 µm |
| Median absolute distance error | 0.124 µm |
| Signed mean distance error | -0.040 µm |
| 95th percentile absolute error | 0.474 µm |
| Mean absolute percentage error | 40.16% |
| Median absolute percentage error | 26.07% |
| Acquisition-bootstrap 95% CI for MAE | 0.136–0.211 µm |
| Errors within 0.2 µm | 67.0% |
| Median manual distance | 0.581 µm |
| Matched retained complete cells with volume measurements | 121 |
| Mean absolute volume error after default filters | 5.64% |
| Final retained cells, including unscored cells | 478 |
| Predicted signals, including unscored detections | 524 |

Only complete manual cells matched at whole-cell IoU ≥0.5 qualify. Signal matches are one-to-one, within the same cell/channel and ≤0.6 µm. Ground truth distances are actual stored manual A–B segments, not synthetic distances. Manual signal membership must be unique and the channel unambiguous. All outliers are included. Multiple signal channels are processed independently; CSV channels are zero-based. A channel number can denote different stains in different scans.

Volume error is conditional on 121 matched cells surviving roundness and signal retention; it is not the unfiltered 197-cell contour comparison. Distance accuracy likewise does not measure missed cells or signals. Small manual distances magnify percentage errors. This reused, previously inspected test fold gives exploratory accuracy estimates, not independent prospective validation. These pairs differ from older model evaluations, so changes in their headline averages alone do not establish improvement.

The default GUI was checked through signal detection, point B creation, distance lines, Imaris/Excel export and 2D/3D display. Native Imaris application opening is not part of that verification.

![Distance distributions](distances/distance-distributions.png)
