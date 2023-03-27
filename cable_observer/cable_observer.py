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

try:
    # from utils.frame_2d import Frame2D
    from utils.frame_3d import Frame3D
    from utils.deformable_linear_object import DeformableLinearObject
except ImportError:
    # from cable_observer.utils.frame_2d import Frame2D
    from cable_observer.utils.frame_3d import Frame3D
    from cable_observer.utils.deformable_linear_object import DeformableLinearObject


class CableObserver:
    """
    Supervisor class for cable observer.

    variables suffixes:
    _idxs - numpy indexing (y, x) or (y, x, z)
    _coords - standard indexing (x, y) or (x, y, z)
    types:
    np.uint8 (numpy) or types.uint8 (numba) - image pixels values
    np.int64 (numpy) or types.int64 (numba) - array indices
    np.float64 (numpy) or types.float64 (numba) - depth values and rest of floating point values
    """

    def __init__(self) -> None:
        self._frame3d = None
        self._dlo = None

        self._debug = False
        self._depth_ranges = [0, 10000]
        self._depth_scale = 0.001
        self._hsv_ranges = [0, 0, 0, 179, 255, 255]
        self._min_length = 10
        self._num_of_knots = 25
        self._num_of_pts = 256
        self._vector_dir_len = 5
        self._z_vertical_shift = 0

    def get_mask(self):
        return self._frame3d.mask * 255

    def set_parameters(self, **kwargs) -> None:
        for arg in kwargs:
            if hasattr(self, "_" + arg):
                setattr(self, "_" + arg, kwargs[arg])

        self._frame3d = Frame3D(hsv_ranges=self._hsv_ranges,
                                depth_ranges=self._depth_ranges, depth_scale=self._depth_scale)
        self._dlo = DeformableLinearObject(num_of_knots=self._num_of_knots,
                                           num_of_pts=self._num_of_pts,
                                           vector_dir_len=self._vector_dir_len,
                                           z_vertical_shift=self._z_vertical_shift)

    def track(self, frame, depth):
        t1 = perf_counter()
        stamps = self._frame3d.execute(img=frame, depth=depth)
        stamps_dlo = self._dlo.execute(frame=self._frame3d)
        t2 = perf_counter()

        if stamps_dlo is not None:
            stamps |= stamps_dlo

        if self._debug:
            output = ""
            for key in stamps.keys():
                output += f"{key}: {stamps[key]:.3f} ms\t"
            output += f"Total: {(t2 - t1) * 1000:.3f} ms"
            print(output)

        return self._dlo.spline_coords_3d
