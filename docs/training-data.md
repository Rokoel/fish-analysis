# Training data

[Documentation](README.md) · [Training](neural-training.md)

Historical training metadata, painted annotations, generated patches and temporary
training scripts were removed. Original scans in the local `FISH` collection and
five release model bundles are preserved. No original scan is distributed in
release bundles. Model manifests retain historical paths as provenance, not live
runtime requirements. Reproducing an old run exactly requires its external original
data/audit; current checkpoints alone do not restore deleted human annotations.

For new work, choose explicit output locations outside the project and audit the
original scans. Keep duplicates and scans from the same acquisition in the same
split. Verify calibration and cell/signal channels before training. Preserve the
split, input fingerprints and model hashes in all subsequent stages.

Manual meshes provide cell references. A is a signal center and a connected A–B
line provides a distance reference. Duplicate scenes/references must not multiply
truth. Unpainted regions are unknown; only explicitly known background is negative.
Cut cells and overlapping annotations need explicit eligibility policies. Signal
supervision is restricted to eligible marked-cell regions with unambiguous channels.

Use [the annotator](training-annotator.md) with explicit `--audit` and `--output`.
See [training](neural-training.md) for reusable audit, data-building and training
commands. Training tools never silently recreate the deleted archive folders.
