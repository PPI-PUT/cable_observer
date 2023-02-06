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

from abc import ABC, abstractmethod
from typing import Dict

import numpy as np
import numpy.typing as npt
from numba import types
from numba.typed import List


class Frame(ABC):
    def __init__(self) -> None:
        self._mask = np.array([], dtype=np.uint8)
        self._mask_roi = np.array([], dtype=np.uint8)
        self._mask_roi_coords = np.array([], dtype=np.int64)
        self._depth = np.array([], dtype=np.float64)
        self._skeleton = np.array([], dtype=np.uint8)
        self._ends_idxs = List.empty_list(types.int64[:])

    @abstractmethod
    def execute(self) -> Dict[str, float]:
        pass

    @property
    @abstractmethod
    def mask(self) -> npt.NDArray[np.uint8]:
        return self._mask

    @property
    @abstractmethod
    def mask_roi(self) -> npt.NDArray[np.uint8]:
        return self._mask_roi

    @property
    @abstractmethod
    def mask_roi_coords(self) -> npt.NDArray[np.int64]:
        return self._mask_roi_coords

    @property
    @abstractmethod
    def skeleton(self) -> npt.NDArray[np.uint8]:
        return self._skeleton

    @property
    @abstractmethod
    def ends_idxs(self) -> List[types.int64[:]]:
        return self._ends_idxs

    @property
    @abstractmethod
    def depth(self) -> npt.NDArray[np.float32]:
        return self._depth
