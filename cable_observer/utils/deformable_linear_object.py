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
from typing import Union

import numpy as np
import numpy.typing as npt
from numba import njit
from numba.typed import List, Dict
from numba.core import types
from scipy.interpolate import LSQUnivariateSpline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

try:
    from utils.frame import Frame
except ImportError:
    from cable_observer.utils.frame import Frame


class DeformableLinearObject:
    def __init__(self, *, min_length: int = 10,
                 num_of_knots: int = 25, num_of_pts: int = 256,
                 vector_dir_len: int = 5, z_vertical_shift: int = 0) -> None:
        self._num_of_knots = np.int64(num_of_knots)
        self._num_of_pts = np.int64(num_of_pts)
        self._vector_dir_len = np.int64(vector_dir_len)
        self._min_length = np.int64(min_length)
        self._z_vertical_shift = np.int64(z_vertical_shift)

        self._T = np.linspace(0., 1., num_of_pts, dtype=np.float64)
        self._previous_spline_coords_3d = np.array([], dtype=np.float64)
        self._spline_coords_3d = np.array([], dtype=np.float64)
        self._spline_coeffs_3d = np.array([], dtype=np.float64)

        #spline_idxs = np.linspace(0, num_of_pts - 1, num_of_pts,
        #                                dtype=np.int64)[:, np.newaxis]
        spline_idxs = self._T[:, np.newaxis]
        poly = PolynomialFeatures(degree=4, include_bias=False)
        self._poly_features = poly.fit_transform(spline_idxs)
        self._poly_reg_model = LinearRegression()

    def execute(self, frame: Frame) -> Dict[str, float]:
        t1 = perf_counter()
        if frame.skeleton.max() == 0.0:
            return

        t2 = perf_counter()
        paths_coords_2d, paths_lengths_2d = self.generate_paths(frame=frame)

        t3 = perf_counter()
        paths_coords_2d_filtered = List.empty_list(types.ListType(types.int64[:]))
        paths_lengths_2d_filtered = List.empty_list(types.float64)
        self.paths_filter(
            paths_coords_2d_filtered=paths_coords_2d_filtered,
            paths_lengths_2d_filtered=paths_lengths_2d_filtered, paths_coords_2d=paths_coords_2d,
            paths_lengths_2d=paths_lengths_2d, min_length=self._min_length)

        t4 = perf_counter()
        paths_coords_2d_sorted, paths_lengths_2d_sorted = self.sort_paths(
            paths_coords_2d=paths_coords_2d_filtered, paths_lengths_2d=paths_lengths_2d_filtered)

        t5 = perf_counter()
        paths_coords_z = List.empty_list(types.ListType(types.float64))
        self.get_paths_coords_z(
            paths_coords_z=paths_coords_z, paths_coords_2d=paths_coords_2d_sorted,
            depth=frame.depth, z_vertical_shift=self._z_vertical_shift)

        t6 = perf_counter()
        gaps_lengths_2d = List.empty_list(types.float64)
        self.get_gaps_lengths(
            gaps_lengths_2d=gaps_lengths_2d, paths_coords_2d=paths_coords_2d_sorted)

        t7 = perf_counter()
        linspaces_2d = List.empty_list(types.float64[:])
        self.get_linspaces(paths_coords_2d=paths_coords_2d_sorted,
                           paths_lengths_2d=paths_lengths_2d_sorted,
                           gaps_lengths_2d=gaps_lengths_2d, linspaces_2d=linspaces_2d)
        linspace_2d = np.concatenate(linspaces_2d)

        t8 = perf_counter()
        full_path_coords_3d = List.empty_list(types.float64[:])
        self.concatenate_paths_3d(
            paths_coords_2d=paths_coords_2d_sorted, paths_lengths_2d=paths_lengths_2d_sorted,
            gaps_lengths_2d=gaps_lengths_2d, paths_coords_z=paths_coords_z,
            full_path_coords_3d=full_path_coords_3d)

        t9 = perf_counter()
        spline_coords_3d, spline_coeffs_3d = self.fit_spline(
            path_coords_3d=full_path_coords_3d, linspace_2d=linspace_2d)

        t10 = perf_counter()
        self._spline_coords_3d, self._spline_coeffs_3d = self.validate_spline_order(
            spline_coords=spline_coords_3d, spline_coeffs=spline_coeffs_3d,
            previous_spline_coords=self._previous_spline_coords_3d)

        t11 = perf_counter()

        self._previous_spline_coords_3d = spline_coords_3d

        return {
            "skeleton check": (t2 - t1)*1000,
            "generate_paths": (t3 - t2)*1000,
            "paths_filter": (t4 - t3)*1000,
            "sort_paths": (t5 - t4)*1000,
            "get_paths_coords_z": (t6 - t5)*1000,
            "get_gaps_lengths": (t7 - t6)*1000,
            "get_linspaces": (t8 - t7)*1000,
            "concatenate_paths_3d": (t9 - t8)*1000,
            "fit_spline": (t10 - t9)*1000,
            "validate_spline_order": (t11 - t10)*1000
        }

    @property
    def spline_coords_3d(self) -> npt.NDArray[np.float64]:
        return self._spline_coords_3d

    @property
    def spline_coeffs_3d(self) -> npt.NDArray[np.float64]:
        return self._spline_coeffs_3d

    def generate_paths(self, frame: Frame) -> Union[List[List[types.int64[:]]],
                                                    List[types.float64]]:
        paths_coords_2d = List.empty_list(types.ListType(types.int64[:]))
        paths_lengths_2d = List.empty_list(types.float64)
        skeleton_pad = np.pad(array=frame.skeleton, pad_width=1)

        for end_idxs in frame.ends_idxs.T:
            if skeleton_pad[end_idxs[0] + 1, end_idxs[1] + 1] == 0.0:  # with padding shift
                continue
            path_coords_2d = List.empty_list(types.int64[:])
            length_2d = self.walk(path_coords_2d=path_coords_2d,
                                  skeleton=skeleton_pad, end_idxs=end_idxs)
            paths_coords_2d.append(path_coords_2d)
            paths_lengths_2d.append(length_2d)

        return paths_coords_2d, paths_lengths_2d

    @staticmethod
    @njit(target_backend='cuda', fastmath=True)
    def walk(path_coords_2d: List[types.int64[:]], skeleton: npt.NDArray[np.uint8],
             end_idxs: npt.NDArray[np.int64]) -> np.float64:
        path_coords_2d.append(np.array([end_idxs[1], end_idxs[0]]))
        is_finished = False
        length_2d = 0.0

        while not is_finished:
            is_finished = True
            act_coords = path_coords_2d[-1] + 1  # add padding shift
            skeleton[act_coords[1], act_coords[0]] = 0.0
            for dx, dy in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
                if skeleton[act_coords[1] + dy, act_coords[0] + dx]:
                    # substract padding shift
                    path_coords_2d.append(
                        np.array([act_coords[0] + dx - 1, act_coords[1] + dy - 1]))
                    length_2d += np.linalg.norm(np.array([dx, dy], dtype=np.float64))
                    is_finished = False
                    break

        return length_2d

    @staticmethod
    @njit(target_backend='cuda', fastmath=True)
    def paths_filter(paths_coords_2d_filtered: List[List[types.int64[:]]],
                     paths_lengths_2d_filtered: List[types.float64],
                     paths_coords_2d: List[List[types.int64[:]]],
                     paths_lengths_2d: List[types.float64],
                     min_length: np.float64) -> None:

        for key, value in enumerate(paths_lengths_2d):
            if value >= min_length:
                paths_coords_2d_filtered.append(paths_coords_2d[key])
                paths_lengths_2d_filtered.append(value)

    def sort_paths(self, paths_coords_2d: List[List[types.int64[:]]],
                   paths_lengths_2d: List[types.float64]) -> Union[List[List[types.int64[:]]],
                                                                   List[types.float64]]:
        MAX = 1e10
        m = 0.05
        conn = List.empty_list(types.int64[:])
        skips = Dict.empty(
            key_type=types.int64,
            value_type=types.int64,
        )
        stats = Dict.empty(
            key_type=types.int64,
            value_type=types.int64,
        )
        len_paths_coords_2d = len(paths_coords_2d)

        for i in range(len_paths_coords_2d):
            stats[i] = 0
        resultant_paths_coords_2d = List.empty_list(types.ListType(types.int64[:]))
        resultant_paths_lengths_2d = List.empty_list(types.float64)

        # calculate dists between all endings
        begins = np.array([c[0] for c in paths_coords_2d])
        ends = np.array([c[-1] for c in paths_coords_2d])

        begin_vectors = np.array(
            [path_coords_2d[0] -
             path_coords_2d[min(self._vector_dir_len, len(path_coords_2d) - 1)]
             for path_coords_2d in paths_coords_2d],
            dtype=np.float64)
        end_vectors = np.array(
            [path_coords_2d[-1] -
             path_coords_2d[max(-self._vector_dir_len - 1, - len(path_coords_2d))]
             for path_coords_2d in paths_coords_2d],
            dtype=np.float64)
        begin_directions = np.array([np.arctan2(begin_vector[0], begin_vector[1])
                                    for begin_vector in begin_vectors], dtype=np.float64)
        end_directions = np.array([np.arctan2(end_vector[0], end_vector[1])
                                  for end_vector in end_vectors], dtype=np.float64)

        be = np.concatenate([begins, ends], axis=0)
        dists = np.linalg.norm(be[np.newaxis] - be[:, np.newaxis], axis=-1)
        be_dirs = np.concatenate([begin_directions, end_directions], axis=0)
        dists_dirs = np.abs(np.pi - np.abs(be_dirs[np.newaxis] - be_dirs[:, np.newaxis]))

        dists = m * dists + (1 - m) * dists_dirs

        dists[np.arange(2 * len_paths_coords_2d),
              (np.arange(2 * len_paths_coords_2d) + len_paths_coords_2d) %
              (2 * len_paths_coords_2d)] = MAX
        dists[np.arange(2 * len_paths_coords_2d), np.arange(2 * len_paths_coords_2d)] = MAX

        # greadily choose connections
        self.find_order_of_paths(
            conn=conn, skips=skips, stats=stats, dists=dists,
            len_paths_coords_2d=len_paths_coords_2d)

        # find starting index
        z = np.array(conn)
        _, counts = np.unique(z, return_counts=True)
        start_id = 0
        for k in range(len(counts)):
            if counts[k] == 1:
                start_id = k
                break

        # traverse and build path
        self.pick_best_paths(
            paths_coords_2d=paths_coords_2d, paths_lengths_2d=paths_lengths_2d,
            len_paths_coords_2d=len_paths_coords_2d,
            resultant_paths_coords_2d=resultant_paths_coords_2d,
            resultant_paths_lengths_2d=resultant_paths_lengths_2d, start_id=start_id, skips=skips)

        return resultant_paths_coords_2d, resultant_paths_lengths_2d

    @staticmethod
    @njit(target_backend='cuda', fastmath=True)
    def find_order_of_paths(
            conn: List[types.int64[:]],
            skips: Dict[types.int64, types.int64],
            stats: Dict[types.int64, types.int64],
            dists: npt.NDArray[np.int64],
            len_paths_coords_2d: np.int64) -> np.float64:
        MAX = 1e10
        loss = 0.0
        while True:
            m = np.argmin(dists)
            mx = m // (2 * len_paths_coords_2d)
            my = m % (2 * len_paths_coords_2d)
            m_value = dists[mx][my]
            dists[mx] = MAX
            dists[my] = MAX
            if stats[mx % len_paths_coords_2d] == 1 and stats[my % len_paths_coords_2d] == 1 \
                and np.count_nonzero(np.array(
                    list(stats.values())) == 1) == 2:
                if (dists == MAX).all():
                    break
                continue
            dists[:, my] = MAX
            dists[:, mx] = MAX
            if (dists == MAX).all():
                break
            conn.append(np.array([mx % len_paths_coords_2d, my % len_paths_coords_2d]))
            skips[mx] = my
            skips[my] = mx
            stats[mx % len_paths_coords_2d] += 1
            stats[my % len_paths_coords_2d] += 1
            loss += m_value
        return loss

    @staticmethod
    @njit(target_backend='cuda', fastmath=True)
    def pick_best_paths(
            paths_coords_2d: List[List[types.int64[:]]],
            paths_lengths_2d: List[types.float64],
            len_paths_coords_2d: np.int64, resultant_paths_coords_2d: List[List[types.int64[:]]],
            resultant_paths_lengths_2d: List[types.float64],
            start_id: np.int64, skips: Dict[types.int64, types.int64]) -> None:
        act_id = start_id if start_id in skips else start_id + len_paths_coords_2d
        while True:
            path_coords_2d = paths_coords_2d[act_id % len_paths_coords_2d]
            length_2d = paths_lengths_2d[act_id % len_paths_coords_2d]
            if act_id < len_paths_coords_2d:
                path_coords_2d = path_coords_2d[::-1]
            resultant_paths_coords_2d.append(path_coords_2d)
            resultant_paths_lengths_2d.append(length_2d)
            if act_id not in skips:
                break
            act_id = skips[act_id]
            act_id = act_id + len_paths_coords_2d \
                if act_id < len_paths_coords_2d else act_id - len_paths_coords_2d

    @staticmethod
    @njit(target_backend='cuda', fastmath=True)
    def get_paths_coords_z(paths_coords_z: List[List[types.float64]],
                           paths_coords_2d: List[List[types.int64[:]]],
                           depth: npt.NDArray[np.float64], z_vertical_shift=np.float64) -> None:

        for path_coords_2d in paths_coords_2d:
            z_coords = [
                [np.int64(
                    min(
                        max(
                            np.around(
                                path_coord[0] + z_vertical_shift / depth.shape[1] * 2 *
                                (depth.shape[1] / 2 - path_coord[0])),
                            0),
                        depth.shape[1] - 1)),
                 path_coord[1]] for path_coord in path_coords_2d]
            z = [depth[z_coord[1], z_coord[0]] for z_coord in z_coords]
            paths_coords_z.append(List(z))

    @staticmethod
    @njit(target_backend='cuda', fastmath=True)
    def get_gaps_lengths(
            paths_coords_2d: List[List[types.int64[:]]],
            gaps_lengths_2d: List[types.float64]) -> List:
        for key, _ in enumerate(paths_coords_2d[:-1]):
            gap_length = np.linalg.norm(
                (paths_coords_2d[key][-1] - paths_coords_2d[key + 1][0]).astype(np.float64))
            gaps_lengths_2d.append(gap_length)

    @staticmethod
    @njit(target_backend='cuda', fastmath=True)
    def get_linspaces(
            paths_coords_2d: List[List[types.int64[:]]],
            paths_lengths_2d: List[types.float64],
            gaps_lengths_2d: List[types.float64],
            linspaces_2d: List[types.float64[:]]) -> None:
        full_length = sum(paths_lengths_2d) + sum(gaps_lengths_2d)
        curr_value = 0
        for key, path_length_2d in enumerate(paths_lengths_2d):
            linspaces_2d.append(np.linspace(curr_value, curr_value + path_length_2d / full_length,
                                            len(paths_coords_2d[key])))
            curr_value += path_length_2d / full_length
            if key < len(gaps_lengths_2d):
                curr_value += gaps_lengths_2d[key] / full_length

    @staticmethod
    @njit(target_backend='cuda', fastmath=True)
    def concatenate_paths_3d(
            paths_coords_2d: List[List[types.int64[:]]],
            paths_lengths_2d: List[types.float64],
            gaps_lengths_2d: List[types.float64],
            paths_coords_z: List[List[types.float64]],
            full_path_coords_3d: List[types.int64[:]]) -> np.float64:
        for key_1, path_coords_2d in enumerate(paths_coords_2d):
            for key_2, coord_2d in enumerate(path_coords_2d):
                full_path_coords_3d.append(
                    np.array(
                        [coord_2d[0],
                         coord_2d[1],
                         paths_coords_z[key_1][key_2]],
                        dtype=np.float64))
        full_path_length_2d = sum(paths_lengths_2d) + sum(gaps_lengths_2d)
        return full_path_length_2d

    @staticmethod
    @njit(target_backend='cuda', fastmath=True)
    def validate_spline_order(spline_coords: npt.NDArray[np.float64],
                              spline_coeffs: npt.NDArray[np.float64],
                              previous_spline_coords: npt.NDArray[np.float64]) -> \
            npt.NDArray[np.float64]:
        if len(previous_spline_coords) == 0:
            return spline_coords, spline_coeffs

        diff = np.linalg.norm(spline_coords - previous_spline_coords)
        diff_inv = np.linalg.norm(np.fliplr(spline_coords) - previous_spline_coords)

        if diff > diff_inv:
            return np.fliplr(spline_coords), np.fliplr(spline_coeffs)
        else:
            return spline_coords, spline_coeffs

    def fit_spline(self, path_coords_3d: List[types.float64[:]],
                   linspace_2d: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        xyz = np.stack(path_coords_3d, axis=1)
        k = self._num_of_knots - 4
        #d = np.int64((linspace_2d.shape[0] - 2) / k) + 1
        #knots = linspace_2d[1:-1:d]
        d = np.linspace(1., linspace_2d.shape[0] - 2, k).astype(np.int64)
        knots = linspace_2d[d]

        self.x_spline = LSQUnivariateSpline(linspace_2d, xyz[0], knots)
        self.y_spline = LSQUnivariateSpline(linspace_2d, xyz[1], knots)
        # TODO: check if invalidating too far away z's will help
        valid = xyz[2] != 0
        t_v = linspace_2d[valid]
        #knots_v = t_v[1:-1:d]
        d = np.linspace(1., t_v.shape[0] - 2, k).astype(np.int64)
        knots_v = t_v[d]
        z_v = xyz[2][valid]
        self.z_spline = LSQUnivariateSpline(t_v, z_v, knots_v)

        self._poly_reg_model.fit(self._poly_features, self.z_spline(self._T))
        z_coords = self._poly_reg_model.predict(self._poly_features)

        knots_ = np.linspace(0., 1., self._num_of_knots)[1:-1]
        self.x_spline = LSQUnivariateSpline(self._T, self.x_spline(self._T), knots_)
        self.y_spline = LSQUnivariateSpline(self._T, self.y_spline(self._T), knots_)
        self.z_spline = LSQUnivariateSpline(self._T, z_coords, knots_)

        spline_coords = np.stack((
            self.x_spline(self._T),
            self.y_spline(self._T),
            self.z_spline(self._T)
        ))

        spline_coeffs = np.stack((
            self.x_spline.get_coeffs(),
            self.y_spline.get_coeffs(),
            self.z_spline.get_coeffs(),
        ))
        return spline_coords, spline_coeffs
