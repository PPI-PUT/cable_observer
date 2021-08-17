import cv2
import numpy as np
from skimage.morphology import skeletonize


def set_mask(frame, hsv_params):
    """
    Extract cable using HSV mask.
    :param frame: camera input frame (W x H x 3)
    :type frame: np.array
    :return: binary mask (W x H)
    :rtype: np.array
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h = hsv_params['hue']
    s = hsv_params['saturation']
    v = hsv_params['value']
    o = np.ones_like(hsv)[..., 0].astype(np.float32)
    z = np.zeros_like(o)

    mask = np.where(np.logical_and(
        np.logical_and(hsv[..., 0] >= h["min"], hsv[..., 0] <= h["max"]) if h["min"] < h["max"]  # Hue
        else np.logical_or(hsv[..., 0] >= h["min"], hsv[..., 0] <= h["max"]),
        np.logical_and(np.logical_and(hsv[..., 1] >= s["min"], hsv[..., 1] <= s["max"]),  # Saturation
                       np.logical_and(hsv[..., 2] >= v["min"], hsv[..., 2] <= v["max"]))  # Value
    ),
        o, z)

    # color_mask = frame
    # color_mask[np.where(mask == False)] = [0, 0, 0]

    return mask


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
    # for i in range(-1, 2):
    #    for j in range(-1, 2):
    #        img_spline[u + i, v + j] = 1
    img_spline[u, v] = 1

    return img_spline


def process_image(img):
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
    #t0 = time()
    # find a sub-image containing mask
    x, y, w, h = cv2.boundingRect(img.astype(np.uint8))
    #t1 = time()
    # skeletonize
    skel = skeletonize(img[y:y + h, x:x + w], method='lee') / 255.  # > 0
    #t2 = time()
    # remove more than 3 neighbours
    kernel = np.ones((3, 3), dtype=np.float32)
    less_than_3 = cv2.filter2D(skel, -1, kernel / 3) <= 1. + 1e-6
    #less_than_3 = cv2.filter2D(skel, -1, kernel / 3, borderType=cv2.BORDER_ISOLATED) <= 1.# + 1e-6
    skel = skel * less_than_3.astype(np.float32)
    #t3 = time()

    # find ends
    less_than_3 = cv2.filter2D(skel, -1, kernel / 2, borderType=cv2.BORDER_ISOLATED) <= 1
    r = skel * less_than_3.astype(np.float32)
    idx = np.where(r > 0)
    idx = [[idx[1][i] + x, idx[0][i] + y] for i in range(len(idx[0]))]
    #t4 = time()
    # In case there is no endpoint, then pick random point on skeleton
    if len(idx) == 0:
        idxs = np.where(skel)
        idx = [[idxs[1][0] + x, idxs[0][0] + y]]

    img = np.zeros_like(img)
    img[y:y + h, x:x + w] = skel
    #t5 = time()
    #print("IM_PROC 1:", t1 - t0)
    #print("IM_PROC 2:", t2 - t1)
    #print("IM_PROC 3:", t3 - t2)
    #print("IM_PROC 4:", t4 - t3)
    #print("IM_PROC 5:", t5 - t4)
    return img, idx


def preprocess_image(img):
    """
    Common function for preprocessing the image. It:
    * perform morphological open to filter out the noise
    :param img: binary image
    :type img: np.array
    :return: filtered image
    :rtype: np.array
    """
    x, y, w, h = cv2.boundingRect(img.astype(np.uint8))
    img_part = img[y:y + h, x:x + w]
    img_part = cv2.erode(img_part, np.ones((3, 3)))
    img_part = cv2.dilate(img_part, np.ones((3, 3)))
    img = np.zeros_like(img)
    img[y:y + h, x:x + w] = img_part

    #img = cv2.erode(img, np.ones((3, 3)))
    #img = cv2.dilate(img, np.ones((3, 3)))
    return img
