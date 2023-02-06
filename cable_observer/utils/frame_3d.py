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
from typing import Dict, List

import cv2
import numpy as np
import numpy.typing as npt
from numba import njit

try:
    from utils.frame_2d import Frame2D
except ImportError:
    from cable_observer.utils.frame_2d import Frame2D


class Frame3D(Frame2D):
    def __init__(self, *, hsv_ranges: List[int] = [0, 0, 0, 179, 255, 255],
                 depth_ranges: List[float] = [0.0, 10000.0], depth_scale: float = 1.0) -> None:
        super().__init__(hsv_ranges=hsv_ranges)
        self._depth_ranges = np.array(depth_ranges, dtype=np.float64)
        self._depth_scale = np.float64(depth_scale)

    def execute(self, img: npt.NDArray[np.uint8],
                depth: npt.NDArray[np.float64]) -> Dict[str, float]:
        t1 = perf_counter()
        if len(img.shape) == 3 and img.shape[2] == 3:
            hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            self._mask = self.set_hsv_mask(hsv_img=hsv_img, hsv_ranges=self._hsv_ranges)
        else:
            self._mask = self.set_binary_mask(img=img)
        t2 = perf_counter()
        self._depth = self.set_depth_roi(
            depth=depth, depth_ranges=self._depth_ranges, depth_scale=self._depth_scale)
        t3 = perf_counter()
        self.set_morphology()
        t4 = perf_counter()
        self.set_skeleton()
        t5 = perf_counter()
        return {
            "mask": (t2 - t1)*1000,
            "depth roi": (t3 - t2)*1000,
            "morphology": (t4 - t3)*1000,
            "skeleton": (t5 - t4)*1000
        }

    @staticmethod
    @njit(target_backend='cuda', fastmath=True)
    def set_depth_roi(depth: npt.NDArray[np.float64],
                      depth_ranges: npt.NDArray[np.float64],
                      depth_scale: np.float64) -> npt.NDArray[np.float64]:
        # numpy implementation
        # depth_roi = np.where((depth >= depth_ranges[0]) & (
        #     depth <= depth_ranges[1]), depth * depth_scale, 0.0)

        # numba implementation
        depth_roi = np.zeros_like(depth)
        for u in range(depth.shape[0]):
            for v in range(depth.shape[1]):
                if depth[u, v] >= depth_ranges[0] and depth[u, v] <= depth_ranges[1]:
                    depth_roi[u, v] = depth[u, v] * depth_scale

        return depth_roi
