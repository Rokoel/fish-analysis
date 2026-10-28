"""Image identity shared by inference, evaluation and export."""
import hashlib
import numpy as np

def image_fingerprint(dataset):
    digest=hashlib.sha256();digest.update(str(dataset.volumes[0].shape).encode())
    for volume in dataset.volumes:digest.update(memoryview(np.ascontiguousarray(volume)))
    return digest.hexdigest()
