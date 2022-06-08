import numpy as np
from interfaces import OutputMask
import cv2


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

    def exec(self, splines, shape, output_dilation) -> np.ndarray:
        """
        Visualize spline.
        :param splines: sequence of a spline points
        :type splines: np.array
        :param shape: initial input image shape
        :type shape: tuple
        :param output_dilation: output mask dilation
        :type output_dilation: int
        :return: 2D image of a spline
        :rtype: np.array
        """
        spline_mask = np.zeros(shape=shape, dtype=np.uint8)
        for key, spline in enumerate(splines):
            idxs = np.unique(np.round(spline).astype(np.uint16), axis=0)
            spline_mask[idxs[:, 0], idxs[:, 1]] = colors[key % len(colors) - 1][::-1]

        spline_mask = cv2.dilate(spline_mask, np.ones((3, 3)), iterations=output_dilation)

        return spline_mask
