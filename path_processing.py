from interfaces import PathsProcessing
from path import Path
import numpy as np


class PathProcessing(PathsProcessing):
    def __init__(self, knots, min_path_length, min_path_points, dir_vec_length):
        self.knots = knots
        self.min_path_length = min_path_length
        self.min_path_points = min_path_points
        self.dir_vec_length = dir_vec_length
        self.success = False

    def exec(self, mask_image: np.ndarray, paths_ends: list):
        """
        Generate paths based on mask image.
        :param mask_image: 2D mask image
        :type mask_image: np.array
        :param paths_ends: list of paths ends
        :type paths_ends: list
        :return: spline coordinates, spline parameters
        :rtype: list, list
        """
        if len(paths_ends) == 0:
            raise ValueError("Length of paths_ends cannot equal 0.")

        # Create paths
        paths = []
        paths_coordinates = np.zeros(shape=(0, 2), dtype=np.float)
        skel = np.pad(mask_image, [[1, 1], [1, 1]], 'constant', constant_values=False)
        i = 0
        while len(paths_ends) > 0:
            coordinates, length = self.walk_faster(skel, tuple(paths_ends[0]))
            paths.append(Path(coordinates=coordinates, length=length, knots=self.knots, dir_vec_length=self.dir_vec_length))
            paths_ends.pop(0)
            paths_ends = self.remove_close_points((coordinates[-1][1], coordinates[-1][0]), paths_ends, max_px_gap=1)
            p = paths[-1]
            if length > self.min_path_length:
                np.append(paths_coordinates, np.array([[coordinates[:, 1], coordinates[:, 0]]]))

        # Get rid of too short paths
        paths = [p for p in paths if p.num_points > self.min_path_points]
        #paths = [p for p in paths if p.length > 20.]

        if len(paths) > 1:
            paths = self.sort_paths(paths)
        else:
            paths = [paths]

        spline_params = []
        spline_coords = []
        for p in paths:
            self.success = False
            # Calculate gaps between adjacent paths
            gaps_length = self.get_gaps_length(paths=p)

            # Get a single linespace for a list of paths
            t = self.get_linespaces(paths=p, gaps_length=gaps_length)

            # Merge all paths coordinates
            merged_paths = np.vstack([x() for x in p])
            # Get spline representation for a merged path
            full_length = np.sum([x.length for x in p]) + np.sum(gaps_length)
            merged_path = Path(coordinates=merged_paths, length=full_length, knots=self.knots,
                               dir_vec_length=self.dir_vec_length)
            try:
                spline_coords.append(merged_path.get_spline(t=t))
            except Exception as e:
                print("Performing dilation...")
                return [], []
            self.success = True

            spline_params.append(merged_path.get_spline_params())

        return spline_coords, spline_params

    @staticmethod
    def remove_close_points(last_point, path_ends, max_px_gap=10):
        """
        Remove close points around certain pixel.
        :param last_point: point around which to look
        :type last_point: tuple
        :param path_ends: points to check
        :type path_ends: np.array
        :param max_px_gap: maximum pixel gap between last_point and a single point of path_ends
        :type max_px_gap: int
        :return: points without close points to the last point
        :rtype: np.array
        """
        keys_to_remove = []
        for key, value in enumerate(np.array(path_ends)):
            if (np.fabs(np.array(last_point) - value) < max_px_gap).all():
                keys_to_remove.append(key)
        path_ends = np.delete(np.array(path_ends), keys_to_remove, axis=0)
        return path_ends.tolist()


    @staticmethod
    def walk_faster(skel, start):
        """
        Perform a fast walk through the DLO skeleton 'skel' tarting from 'start'
        :param skel: skeleton of the DLO mask
        :type skel: np.array
        :param start: on of the DLO ends
        :type start: np.array
        :return: sequence of pixels which constitutes a DLO and its length
        :rtype: np.array, float
        """
        path = [(int(start[1]) + 1, int(start[0]) + 1)]
        end = False
        while not end:
            end = True
            act = path[-1]
            skel[act[0], act[1]] = 0.
            for dx, dy in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
                if skel[act[0] + dx, act[1] + dy]:
                    aim_x = act[0] + dx
                    aim_y = act[1] + dy
                    path.append((aim_x, aim_y))
                    end = False
                    break

        path = np.array(path)
        length = np.sum(np.linalg.norm(np.diff(path, axis=0), axis=-1))
        path -= 1
        return path, length

    @staticmethod
    def get_gaps_length(paths):
        """
        Get gaps lengths between adjacent paths.
        :param paths: ordered paths
        :type paths: list
        :return: gaps lengths
        :rtype: list
        """
        gaps_length = []
        for key, value in enumerate(paths[:-1]):
            gap_length = np.linalg.norm(paths[key].end - paths[key + 1].begin)
            gaps_length.append(gap_length)

        return gaps_length

    @staticmethod
    def get_linespaces(paths, gaps_length):
        """
        Get linespace for merged paths.
        :param paths: all paths with correct order
        :type paths: list
        :param gaps_length: all gaps lengths between paths
        :type gaps_length: list
        :return: merged linespaces of all paths
        :rtype: np.array
        """
        full_length = np.sum([d.length for d in paths]) + np.sum(gaps_length)
        t_linespaces = []
        curr_value = 0
        for key, value in enumerate([d.length for d in paths]):
            t_linespace = np.linspace(curr_value, curr_value + value / full_length,
                                      [d.num_points for d in paths][key])
            curr_value += value / full_length
            if key < len(gaps_length):
                curr_value += gaps_length[key] / full_length
            t_linespaces.append(t_linespace)
        t = np.hstack(t_linespaces)
        return t

    @staticmethod
    def sort_paths(paths):
        """
        Put multiple defined paths in the order which minimizes the penalty term
        :param paths: list of Path objects
        :type paths: list(Path)
        :return: sequence of paths which minimizes the penalty term
        :rtype: list(Path)
        """
        # calculate dists between all endings
        m = 0.05
        #m = 0.1
        MAX = 1e10
        l = len(paths)
        begins = np.array([p.begin for p in paths])
        ends = np.array([p.end for p in paths])
        begin_directions = np.array([p.begin_direction for p in paths])
        end_directions = np.array([p.end_direction for p in paths])
        be = np.concatenate([begins, ends], axis=0)
        dists_gaps = np.linalg.norm(be[np.newaxis] - be[:, np.newaxis], axis=-1)
        be_dirs = np.concatenate([begin_directions, end_directions], axis=0)
        dists_dirs = np.abs(np.pi - np.abs(be_dirs[np.newaxis] - be_dirs[:, np.newaxis]))
        dists_curv = dists_dirs / (dists_gaps / 100 + 1e-10)

        dists = m * dists_gaps + (1 - m) * dists_dirs

        dists[np.arange(2 * l), (np.arange(2 * l) + l) % (2 * l)] = MAX
        dists[np.arange(2 * l), np.arange(2 * l)] = MAX
        dists[dists > 2.0] = MAX

        # greadily choose connections
        conn = []
        skips = {}
        while True:
            m = np.argmin(dists)
            mx = m // (2 * l)
            my = m % (2 * l)
            dists[mx] = MAX
            dists[:, my] = MAX
            dists[my] = MAX
            dists[:, mx] = MAX
            conn.append([mx % l, my % l])
            skips[mx] = my
            skips[my] = mx
            if (dists == MAX).all():
                break

        # find starting index
        z = np.array(conn)
        unique, counts = np.unique(z, return_counts=True)
        starting_points = [unique[i] for i in range(len(unique)) if counts[i] == 1]

        resultant_paths = []
        used_path_indices = []
        while starting_points:
            act_id = starting_points[0] if starting_points[0] in skips else starting_points[0] + l
            rp = []
            while True:
                if act_id % l in starting_points:
                    starting_points.remove(act_id % l)
                p = paths[act_id % l]
                used_path_indices.append(act_id % l)
                if act_id < l:
                    p.flip_path()
                rp.append(p)
                if act_id not in skips:
                    break
                act_id = skips[act_id]
                act_id = act_id + l if act_id < l else act_id - l
            resultant_paths.append(rp)
        for i in range(len(paths)):
            if i not in used_path_indices:
                resultant_paths.append([paths[i]])

        # traverse and build path
        #act_id = start_id if start_id in skips else start_id + l
        #while True:
        #    p = paths[act_id % l]
        #    if act_id < l:
        #        p.flip_path()
        #    resultant_paths.append(p)
        #    if act_id not in skips:
        #        break
        #    act_id = skips[act_id]
        #    act_id = act_id + l if act_id < l else act_id - l

        return resultant_paths
