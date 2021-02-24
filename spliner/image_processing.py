import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes
from skimage.morphology import skeletonize, erosion, dilation


def set_mask(frame):
    """
    Process image to extract cable path.
    :param frame: camera input frame (W x H x 3)
    :type frame: np.array
    :return: Preprocessed camera frame (W x H)
    :rtype: np.array
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    o = np.ones_like(hsv)[..., 0].astype(np.float32)
    z = np.zeros_like(o)
    img = np.where(np.logical_and(np.logical_or(hsv[..., 0] < 8., hsv[..., 0] > 170.), hsv[..., 1] > 180), o, z)
    return img


def set_morphology(img):
    """
    Get skeleton representation of paths.
    :param img: preprocessed frame
    :type img: np.array
    :return: processed frame
    :rtype: np.array
    """
    #img = binary_fill_holes(img)
    #img = erosion(img)
    #img = dilation(img)
    img = cv2.erode(img, np.ones((3, 3)))
    img = cv2.dilate(img, np.ones((3, 3)))
    skel = skeletonize(img, method='lee') > 0
    return skel


def get_spline_image(spline_coords, shape):
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
    for i in range(-1, 2):
        for j in range(-1, 2):
            img_spline[u+i, v+j] = 1

    return img_spline


def remove_if_more_than_3_neighbours(img):
    kernel = np.ones((3, 3), dtype=np.float32) / 3
    img = img.astype(np.float32)
    less_than_3 = cv2.filter2D(img, -1, kernel) <= 1
    r = img * less_than_3.astype(np.float32)
    return r.astype(np.bool)


def find_ends(img):
    kernel = np.ones((3, 3), dtype=np.float32) / 2
    img = img.astype(np.float32)
    less_than_3 = cv2.filter2D(img, -1, kernel) <= 1
    r = img * less_than_3.astype(np.float32)
    idx = np.where(r > 0)
    idx = [[idx[1][i], idx[0][i]] for i in range(len(idx[0]))]
    return idx
