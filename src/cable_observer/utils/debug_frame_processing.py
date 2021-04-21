from time import time
import numpy as np
from .image_processing import get_spline_image
import cv2


def time_counter(method):
    def func_wrapper(*args, **kwargs):
        time_start = time()
        output = method(*args, **kwargs)
        duration = time() - time_start
        print("{} (debug method) last {} ms".format(method.__name__, round(duration*1000, 5)))
        return output
    return func_wrapper


class DebugFrameProcessing:
    def __init__(self, frame, cps, poc, last_spline_coords,
                 spline_coords, spline_params, skeleton, mask,  lower_bound, upper_bound, t):
        self._frame = frame
        self._cps = cps
        self._poc = poc
        self._last_spline_coords = last_spline_coords
        self._spline_coords = spline_coords
        self._spline_params = spline_params
        self._skeleton = skeleton
        self._mask = mask
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound
        self._t = t
        self._idx = None
        self._img_pred = self._img_mask = self._img_frame = self._img_skeleton = self._img_spline = None
        self._img_spline_raw = get_spline_image(spline_coords=self._spline_coords, shape=frame.shape)

        # Run debug
        self.run_debug_sequence()

    def coords_to_img(self, coords):
        z = np.zeros_like(coords)
        img = np.stack([coords, z, z], axis=-1)
        img[self._idx[0], self._idx[1], 1] = self._img_spline_raw[self._idx[0], self._idx[1], 1]
        img = np.uint8(img*255)
        return img

    @time_counter
    def set_img_pred(self, N=10, d=0):
        pred = np.zeros_like(self._frame)
        for i in range(N + 1 + 2 * d):
            t = (i - d) / N
            coords = self._lower_bound * t + self._upper_bound * (1 - t)
            uv = np.around(coords).astype(np.int32)
            pred[np.clip(uv[:, 0], 0, pred.shape[0] - 1), np.clip(uv[:, 1], 0, pred.shape[1] - 1)] = 255
        pred = cv2.dilate(pred, np.ones((3, 3)))
        pred = cv2.erode(pred, np.ones((3, 3)))
        self._img_pred = pred

    @time_counter
    def set_img_spline(self):
        img_low = get_spline_image(spline_coords=self._lower_bound, shape=self._frame.shape)
        img_up = get_spline_image(spline_coords=self._upper_bound, shape=self._frame.shape)
        self._img_spline = np.uint8(np.stack([img_low[:, :, 0], self._img_spline_raw[:, :, 1], img_up[:, :, 2]], axis=-1)*255)
        self._idx = np.where(np.any(self._img_spline, axis=-1))

    @time_counter
    def set_img_frame(self):
        img_frame = self._frame
        for i in range(-2, 3):
            for j in range(-2, 3):
                img_frame[self._idx[0] + i, self._idx[1] + j] = self._img_spline[self._idx[0], self._idx[1]]
        # frame[idx[0], idx[1]] = (255*img_spline_raw[idx[0], idx[1]]).astype(np.uint8)
        self._img_frame = img_frame

    @time_counter
    def set_img_mask(self):
        self._img_mask = self.coords_to_img(coords=self._mask)

    @time_counter
    def set_img_skeleton(self):
        self._img_skeleton = self.coords_to_img(coords=self._skeleton)

    @time_counter
    def set_params(self, k=25):
        coeffs = self._spline_params['coeffs'].astype(np.int32)
        for i in range(-2, 3):
            for j in range(-2, 3):
                self._frame[np.clip(coeffs[0] + i, 0, self._frame.shape[0] - 1), np.clip(coeffs[1] + j, 0, self._frame.shape[1] - 1), :] = \
                    np.array([0, 0, 255], dtype=np.uint8)
        self._cps.append(coeffs)
        d = int(self._spline_coords.shape[0] / k) + 1
        self._poc.append(self._spline_coords[::d])

    def run_debug_sequence(self):
        self.set_img_pred()
        self.set_img_spline()
        self.set_img_mask()
        self.set_img_skeleton()
        self.set_img_frame()
        self.set_params()

    def print_t(self):
        ts = 0
        for i in range(1, len(self._t)):
            print("Timer no {}: {} ms".format(i, round((self._t[i] - self._t[i - 1])*1000, 5)))
            ts += (self._t[i] - self._t[i - 1])
        print("Timers sum: {} ms".format(round(ts*1000, 5)))

    def get_params(self):
        return self._cps, self._poc, self._last_spline_coords

    @property
    def img_spline(self):
        return self._img_spline

    @property
    def img_skeleton(self):
        return self._img_skeleton

    @property
    def img_mask(self):
        return self._img_mask

    @property
    def img_pred(self):
        return self._img_pred

    @property
    def img_frame(self):
        return self._img_frame
