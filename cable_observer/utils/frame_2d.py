#!/usr/bin/env python3

# Copyright 2023 Perception for Physical Interaction Laboratory at Poznan University of Technology
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from time import perf_counter
from typing import Dict

import cv2
import numpy as np
import numpy.typing as npt
from skimage.morphology import skeletonize
from numba import njit
from numba.typed import List
from numba.core import types

try:
    from utils.frame import Frame
except ImportError:
    from cable_observer.utils.frame import Frame


class Frame2D(Frame):
    def __init__(self, *, hsv_ranges: List[int] = [0, 0, 0, 179, 255, 255]) -> None:
        self._hsv_ranges = np.array(hsv_ranges, dtype=np.uint8)
        self._mask = np.array([], dtype=np.uint8)
        self._mask_roi = np.array([], dtype=np.uint8)
        self._mask_roi_coords = np.array([], dtype=np.int64)
        self._depth = np.array([], dtype=np.float64)
        self._skeleton = np.array([], dtype=np.uint8)
        self._ends_idxs = List.empty_list(types.int64[:])

    def execute(self, img: npt.NDArray[np.uint8]) -> Dict[str, float]:
        t1 = perf_counter()
        if len(img.shape) == 3 and img.shape[2] == 3:
            hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            self._mask = self.set_hsv_mask(hsv_img=hsv_img, hsv_ranges=self._hsv_ranges)
        else:
            self._mask = self.set_binary_mask(img=img)
        t2 = perf_counter()
        self.set_morphology()
        t3 = perf_counter()
        self.set_skeleton()
        t4 = perf_counter()
        return {
            "mask": (t2 - t1)*1000,
            "morphology": (t3 - t2)*1000,
            "skeleton": (t4 - t3)*1000
        }

    @property
    def mask(self) -> npt.NDArray[np.uint8]:
        return self._mask

    @property
    def mask_roi(self) -> npt.NDArray[np.uint8]:
        return self._mask_roi

    @property
    def mask_roi_coords(self) -> npt.NDArray[np.int64]:
        return self._mask_roi_coords

    @property
    def skeleton(self) -> npt.NDArray[np.uint8]:
        return self._skeleton

    @property
    def ends_idxs(self) -> List[types.int64[:]]:
        return self._ends_idxs

    @property
    def depth(self) -> npt.NDArray[np.float64]:
        return self._depth

    def set_morphology(self, *, erode: bool = True, dilate: bool = True) -> None:
        self._mask_roi_coords = cv2.boundingRect(self._mask)
        self._mask_roi = self._mask[self._mask_roi_coords[1]:
                                    self._mask_roi_coords[1] + self._mask_roi_coords[3],
                                    self._mask_roi_coords[0]:
                                    self._mask_roi_coords[0] + self._mask_roi_coords[2]]
        if erode:
            cv2.erode(src=self._mask_roi, kernel=np.ones((3, 3)), dst=self._mask_roi)
        if dilate:
            cv2.dilate(src=self._mask_roi, kernel=np.ones((3, 3)), dst=self._mask_roi)
        mask_morph = np.zeros_like(self._mask)
        mask_morph[self._mask_roi_coords[1]:
                   self._mask_roi_coords[1] + self._mask_roi_coords[3],
                   self._mask_roi_coords[0]:
                   self._mask_roi_coords[0] + self._mask_roi_coords[2]] = self._mask_roi

        self._mask = mask_morph

    def set_skeleton(self) -> None:
        skeleton_roi = skeletonize(self._mask_roi, method="lee")
        kernel = np.ones((3, 3), dtype=np.float64)
        less_than_3 = cv2.filter2D(skeleton_roi, -1, kernel / 3) <= 1.0 + 1e-6
        less_than_2 = (
            cv2.filter2D(skeleton_roi, -1, kernel / 2, borderType=cv2.BORDER_ISOLATED) <= 1
        )

        # remove more than 3 neighbours
        skeleton_roi = skeleton_roi * less_than_3

        # find ends
        mask_ends = skeleton_roi * less_than_2
        ends_idxs = np.array(np.where(mask_ends), dtype=np.int64)
        ends_idxs += [[self._mask_roi_coords[1]], [self._mask_roi_coords[0]]]

        # in case there is no endpoint, then pick random point on skeleton
        if len(ends_idxs) == 0:
            ends_idxs = np.array(np.where(skeleton_roi), dtype=np.int64)
            ends_idxs += [[self._mask_roi_coords[1]], [self._mask_roi_coords[0]]]

        skeleton = np.zeros_like(self._mask)
        skeleton[self._mask_roi_coords[1]:
                 self._mask_roi_coords[1] + self._mask_roi_coords[3],
                 self._mask_roi_coords[0]:
                 self._mask_roi_coords[0] + self._mask_roi_coords[2]] = skeleton_roi

        self._skeleton = skeleton
        self._ends_idxs = ends_idxs

    @staticmethod
    @njit(target_backend='cuda', fastmath=True)
    def set_binary_mask(img: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        return np.where(img > 0, 1, 0).astype(np.uint8)

    @staticmethod
    @njit(target_backend='cuda', fastmath=True)
    def set_hsv_mask(hsv_img: npt.NDArray[np.uint8],
                     hsv_ranges: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        mask = np.zeros_like(hsv_img[..., 0], dtype=np.uint8)

        if hsv_ranges[0] < hsv_ranges[3]:
            for y in range(hsv_img.shape[0]):
                for x in range(hsv_img.shape[1]):
                    if (hsv_img[y, x, 0] >= hsv_ranges[0] and hsv_img[y, x, 0] <= hsv_ranges[3]) \
                        and hsv_img[y, x, 1] >= hsv_ranges[1] \
                        and hsv_img[y, x, 1] <= hsv_ranges[4] \
                        and hsv_img[y, x, 2] >= hsv_ranges[2] \
                        and hsv_img[y, x, 2] <= hsv_ranges[5]:
                        mask[y, x] = 1
        else:
            for y in range(hsv_img.shape[0]):
                for x in range(hsv_img.shape[1]):
                    if (hsv_img[y, x, 0] >= hsv_ranges[0] or hsv_img[y, x, 0] <= hsv_ranges[3]) \
                        and hsv_img[y, x, 1] >= hsv_ranges[1] \
                        and hsv_img[y, x, 1] <= hsv_ranges[4] \
                        and hsv_img[y, x, 2] >= hsv_ranges[2] \
                        and hsv_img[y, x, 2] <= hsv_ranges[5]:
                        mask[y, x] = 1
        return mask
