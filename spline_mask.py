import numpy as np
from interfaces import OutputMask

colors = [
    (255, 255, 255),
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (0, 255, 255),
    (255, 0, 255),
    (192, 192, 192),
    (128, 128, 128),
    (128, 0, 0),
    (128, 128, 0),
    (0, 128, 0),
    (128, 0, 128),
    (0, 128, 128),
    (0, 0, 128)
]

class SplineMask(OutputMask):
    def __init__(self):
        pass

    def exec(self, splines, shape) -> np.ndarray:
        spline_mask = np.zeros(shape=shape, dtype=np.uint8)
        for key, spline in enumerate(splines):
            idxs = np.unique(np.round(spline).astype(np.uint16), axis=0)
            spline_mask[idxs[:, 0], idxs[:, 1]] = colors[key % len(colors) - 1][::-1]

        return spline_mask
