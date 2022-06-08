import numpy as np
import cv2
from skimage.morphology import skeletonize

from interfaces import Segmentation


class MaskSegmentation(Segmentation):
    def __init__(self, is_masked, hsv_params):
        self.is_masked = is_masked
        self.hsv_params = hsv_params
        self.paths_ends = None

    def exec(self, input_image: np.ndarray, dilate_it: int) -> np.ndarray:
        """
        Process input image.
        :param input_image: 2D image
        :type input_image: np.array
        :param dilate_it: initial num of dilation iterations
        :type dilate_it: int
        :return: list of paths
        :rtype: list
        """
        mask = input_image if self.is_masked else self.set_mask(input_image)
        mask = self.preprocess_image(mask, iterations=dilate_it)
        result, paths_ends = self.process_image(mask)
        self.paths_ends = paths_ends
        return result

    def set_mask(self, frame):
        """
        Process image to extract cable path.
        :param frame: camera input frame (W x H x 3)
        :type frame: np.array
        :return: Preprocessed camera frame (W x H)
        :rtype: np.array
        """
        h = self.hsv_params['hue']
        s = self.hsv_params['saturation']
        v = self.hsv_params['value']
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        o = np.ones_like(hsv)[..., 0].astype(np.float32)
        z = np.zeros_like(o)
        mask = np.where(np.logical_and(
            np.logical_and(hsv[..., 0] >= h["min"], hsv[..., 0] <= h["max"]) if h["min"] < h["max"]  # Hue
            else np.logical_or(hsv[..., 0] >= h["min"], hsv[..., 0] <= h["max"]),
            np.logical_and(np.logical_and(hsv[..., 1] >= s["min"], hsv[..., 1] <= s["max"]),  # Saturation
                           np.logical_and(hsv[..., 2] >= v["min"], hsv[..., 2] <= v["max"]))  # Value
        ),
            o, z)
        #img = np.where(np.logical_and(np.logical_or(hsv[..., 0] < hue_upper, hsv[..., 0] > hue_lower), hsv[..., 1] > saturation_upper), o, z)
        return mask

    def get_spline_image(self, spline_coords, shape):
        """
        Get an array with (x, y) coordinates set in.
        :param spline_coords: x & y coordinates
        :type spline_coords: np.array
        :param shape: frame shape (W x H x 3)
        :type shape: tuple
        :return: spline path as an image
        :rtype: np.array
        """
        img_spline = np.zeros(shape=shape)
        keys_to_remove = []

        for key, value in enumerate(spline_coords):
            if spline_coords[..., 0][key] > shape[0] or spline_coords[..., 0][key] < 0 or \
                    spline_coords[..., 1][key] > shape[1] or spline_coords[..., 1][key] < 0:
                keys_to_remove.append(key)
        spline_coords = np.delete(spline_coords, keys_to_remove, axis=0)

        u = spline_coords[..., 0].astype(int)
        v = spline_coords[..., 1].astype(int)
        # for i in range(-1, 2):
        #    for j in range(-1, 2):
        #        img_spline[u + i, v + j] = 1
        img_spline[u, v] = 1

        return img_spline

    def process_image(self, img):
        """
        Common function for processing the mask. It:
        * finds a sub-image containing mask
        * skeletonizes it
        * removes pixels with more than 3 neighbours
        * finds ends of the DLO
        :param img: mask image
        :type img: np.array
        :return: (skeletonized image, indexes of the pixels which are considered to be DLO ends)
        :rtype: (np.array, list)
        """
        # find a sub-image containing mask
        x, y, w, h = cv2.boundingRect(img.astype(np.uint8))
        # skeletonize
        skel = skeletonize(img[y:y + h, x:x + w], method='lee') / 255.  # > 0
        # remove more than 3 neighbours
        kernel = np.ones((3, 3), dtype=np.float32)
        less_than_3 = cv2.filter2D(skel, -1, kernel / 3) <= 1. + 1e-6
        skel = skel * less_than_3.astype(np.float32)

        # find ends
        less_than_3 = cv2.filter2D(skel, -1, kernel / 2) <= 1
        r = skel * less_than_3.astype(np.float32)
        idx = np.where(r > 0)
        idx = [[idx[1][i] + x, idx[0][i] + y] for i in range(len(idx[0]))]

        img = np.zeros_like(img)
        img[y:y + h, x:x + w] = skel
        return img, idx

    def preprocess_image(self, img, iterations):
        """
        Common function for preprocessing the image. It:
        * finds mask if needed
        * perform morphological open to filter out the noise
        :param img: image or mask
        :type img: np.array
        :param iterations: erode and dilate iterations
        :type iterations: int
        :return: masked and filtered image
        :rtype: np.array
        """
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        x, y, w, h = cv2.boundingRect(img.astype(np.uint8))
        img_part = img[y:y + h, x:x + w]
       # img_part = cv2.erode(img_part, np.ones((3, 3)), iterations=iterations)
        img_part = cv2.dilate(img_part, np.ones((3, 3)), iterations=iterations)
        img = np.zeros_like(img)
        img[y:y + h, x:x + w] = img_part

        return img
