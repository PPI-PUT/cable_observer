from time import time
from .path import Path
import numpy as np


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
    try:
        path_ends.remove(list(last_point))
    except ValueError:
        pass
    return path_ends
    # keys_to_remove = []
    # for key, value in enumerate(np.array(path_ends)):
    #   if (np.fabs(np.array(last_point) - value) < max_px_gap).all():
    #       keys_to_remove.append(key)
    # path_ends = np.delete(np.array(path_ends), keys_to_remove, axis=0)
    # return path_ends.tolist()


dxy = np.meshgrid(np.linspace(-1, 1, 3), np.linspace(-1, 1, 3))
dxy = np.stack(dxy, axis=-1)
dxy = dxy[:, :, ::-1]


def walk_fast(skel, start):
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
    card = 1
    while card > 0:
        act = path[-1]
        skel[act[0], act[1]] = 0.

        patch = skel[act[0] - 1: act[0] + 2, act[1] - 1: act[1] + 2]
        xy = dxy + act
        b = xy[patch]

        card = b.shape[0]
        if card == 1:
            aim = b[0].astype(np.int32)
        else:
            break

        path.append((aim[0], aim[1]))
    path = np.array(path)
    length = np.sum(np.linalg.norm(np.diff(path, axis=0), axis=-1))
    path -= 1
    return path, length


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
    # path = [(int(start[1]), int(start[0]))]
    end = False
    while not end:
        end = True
        act = path[-1]
        skel[act[0], act[1]] = 0.
        for dx, dy in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
            if skel[act[0] + dx, act[1] + dy]:
                # if 0 <= act[0] + dx < skel.shape[0] and 0 <= act[1] + dy < skel.shape[1] and skel[act[0] + dx, act[1] + dy]:
                aim_x = act[0] + dx
                aim_y = act[1] + dy
                path.append((aim_x, aim_y))
                end = False
                break

    path = np.array(path)
    length = np.sum(np.linalg.norm(np.diff(path, axis=0), axis=-1))
    path -= 1
    return path, length


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


def get_linspaces(paths, gaps):
    """
    Get linspace for merged paths.
    :param paths: all paths with correct order
    :type paths: list
    :param gaps: all gaps lengths between paths
    :type gaps: list
    :return: merged linspaces of all paths
    :rtype: np.array
    """
    full_length = np.sum([d.length for d in paths]) + np.sum(gaps)
    t_linespaces = []
    curr_value = 0
    for key, value in enumerate([d.length for d in paths]):
        t_linspace = np.linspace(curr_value, curr_value + value / full_length,
                                 [d.num_points for d in paths][key])
        curr_value += value / full_length
        if key < len(gaps):
            curr_value += gaps[key] / full_length
        t_linespaces.append(t_linspace)
    t = np.hstack(t_linespaces) if len(t_linespaces) > 0 else np.arange(0, 1, 0.1)
    return t


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
    MAX = 1e10
    l = len(paths)
    begins = np.array([p.begin for p in paths])
    ends = np.array([p.end for p in paths])
    begin_directions = np.array([p.begin_direction for p in paths])
    end_directions = np.array([p.end_direction for p in paths])
    be = np.concatenate([begins, ends], axis=0)
    dists = np.linalg.norm(be[np.newaxis] - be[:, np.newaxis], axis=-1)
    be_dirs = np.concatenate([begin_directions, end_directions], axis=0)
    dists_dirs = np.abs(np.pi - np.abs(be_dirs[np.newaxis] - be_dirs[:, np.newaxis]))

    dists = m * dists + (1 - m) * dists_dirs

    dists[np.arange(2 * l), (np.arange(2 * l) + l) % (2 * l)] = MAX
    dists[np.arange(2 * l), np.arange(2 * l)] = MAX

    # greadily choose connections
    conn = []
    skips = {}
    stats = {i: 0 for i in range(l)}

    
    def find_order_of_paths(conn, skips, stats, dists, loss):
        """The most straightforward way to obtain the order of paths, which greedily chooses the connection
        with the smallest distance"""
        while True:
            m = np.argmin(dists)
            mx = m // (2 * l)
            my = m % (2 * l)
            m_value = dists[mx][my]
            dists[mx] = MAX
            dists[my] = MAX
            if stats[mx % l] == 1 and stats[my % l] == 1 and np.count_nonzero(np.array(list(stats.values())) == 1) == 2:
                if (dists == MAX).all():
                    break
                continue
            dists[:, my] = MAX
            dists[:, mx] = MAX
            if (dists == MAX).all():
                break
            conn.append([mx % l, my % l])
            skips[mx] = my
            skips[my] = mx
            stats[mx % l] += 1
            stats[my % l] += 1
            loss += m_value
        return conn, skips, stats, loss

    def find_order_of_paths_recursively(conn, skips, stats, dists, loss):
        """The most straightforward way to obtain the order of paths, which greedily chooses the connection
        with the smallest distance, but does it recursively instead of the infinite while"""
        conn = list(conn)
        skips = dict(skips)
        stats = dict(stats)
        dists = np.array(dists)
        m = np.argmin(dists)
        mx = m // (2 * l)
        my = m % (2 * l)
        m_value = dists[mx][my]
        dists[mx] = MAX
        dists[my] = MAX
        # TODO check if this is working properly!!!
        if stats[mx % l] == 1 and stats[my % l] == 1 and np.count_nonzero(np.array(list(stats.values())) == 1) == 2:
            if (dists == MAX).all():
                return conn, skips, stats, loss
            conn, skips, stats, loss = find_order_of_paths_recursively(conn, skips, stats, dists, loss)
        dists[:, my] = MAX
        dists[:, mx] = MAX
        if (dists == MAX).all():
            return conn, skips, stats, loss
        loss += m_value
        conn.append([mx % l, my % l])
        skips[mx] = my
        skips[my] = mx
        stats[mx % l] += 1
        stats[my % l] += 1
        conn, skips, stats, loss = find_order_of_paths_recursively(conn, skips, stats, dists, loss)
        return conn, skips, stats, loss

    def find_order_of_paths_two_cases(conn, skips, stats, dists, loss, idx=None):
        """Sophisticated way to obtain the order of paths, which takes into account not only the best connection
        but also a second one if it is close to the best"""
        conn = list(conn)
        skips = dict(skips)
        stats = dict(stats)
        dists = np.array(dists)
        if idx is None:
            s = np.argsort(dists, axis=None)
            ds = dists.flatten()
            s1 = ds[s[0]]
            s2 = ds[s[2]]
            if s2 < s1 * 1.2:
                conn1, skips1, stats1, loss1 = find_order_of_paths_two_cases(conn, skips, stats, dists, loss, idx=s[0])
                conn2, skips2, stats2, loss2 = find_order_of_paths_two_cases(conn, skips, stats, dists, loss, idx=s[2])
                if loss1 < loss2:
                    return conn1, skips1, stats1, loss1
                else:
                    return conn2, skips2, stats2, loss2
            idx = s[0]
        m = idx
        mx = m // (2 * l)
        my = m % (2 * l)
        m_value = dists[mx][my]
        dists[mx] = MAX
        dists[my] = MAX
        # TODO check if this is working properly!!!
        if stats[mx % l] == 1 and stats[my % l] == 1 and np.count_nonzero(np.array(list(stats.values())) == 1) == 2:
            if (dists == MAX).all():
                return conn, skips, stats, loss
            conn, skips, stats, loss = find_order_of_paths_two_cases(conn, skips, stats, dists, loss)
        dists[:, my] = MAX
        dists[:, mx] = MAX
        if (dists == MAX).all():
            return conn, skips, stats, loss
        loss += m_value
        conn.append([mx % l, my % l])
        skips[mx] = my
        skips[my] = mx
        stats[mx % l] += 1
        stats[my % l] += 1
        conn, skips, stats, loss = find_order_of_paths_two_cases(conn, skips, stats, dists, loss)
        return conn, skips, stats, loss


    t0 = time()
    #conn, skips, stats, loss = find_order_of_paths_two_cases(conn, skips, stats, dists, 0.)
    conn, skips, stats, loss = find_order_of_paths(conn, skips, stats, dists, 0.)
    t1 = time()
    a = t1 - t0

    # find starting index
    z = np.array(conn)
    unique, counts = np.unique(z, return_counts=True)
    start_id = 0
    for k in range(len(counts)):
        if counts[k] == 1:
            start_id = k
            break

    # traverse and build path
    resultant_paths = []
    act_id = start_id if start_id in skips else start_id + l
    while True:
        p = paths[act_id % l]
        if act_id < l:
            p.flip_path()
        resultant_paths.append(p)
        if act_id not in skips:
            break
        act_id = skips[act_id]
        act_id = act_id + l if act_id < l else act_id - l

    return resultant_paths


def generate_paths(skeleton, depth, paths_ends, params_path):
    """
    Get total length of paths and gaps.
    :param skeleton: skeleton image
    :type skeleton: np.array
    :param paths_ends: ending coordinates of paths
    :type paths_ends: list
    :param params_path: path parameters
    :type params_path: dict
    :return: all paths
    :rtype: list
    """
    paths = []
    skel = np.zeros((skeleton.shape[0] + 2, skeleton.shape[1] + 2), dtype=np.bool)
    skel[1:-1, 1:-1] = skeleton
    while len(paths_ends) > 0:
        coordinates, length = walk_faster(skel, tuple(paths_ends[0]))
        z_coordinates = depth[coordinates[:, 0] - 1, coordinates[:, 1] - 1]
        paths.append(Path(coordinates=coordinates, z_coordinates=z_coordinates, length=length,
                          num_of_knots=params_path['num_of_knots'],
                          num_of_pts=params_path['num_of_pts'],
                          max_width=params_path['max_width'],
                          width_step=params_path['width_step'],
                          vector_dir_len=params_path['vector_dir_len']))
        paths_ends.pop(0)
    return paths


def select_paths(paths, min_points=3, min_length=10):
    """
    Get rid of too short paths.
    :param paths: list of Path objects
    :type paths: list(Path)
    :param min_points: minimum points within path
    :type min_points: int
    :param min_length: minimum path length
    :type min_length: int
    :return: list of Path objects
    :rtype: list(Path)
    """
    paths = [p for p in paths if p.num_points > min_points]
    paths = [p for p in paths if p.length > min_length]
    return paths


def concatenate_paths(paths):
    """
    Concatenate all paths coordinates.
    :param paths: list of Path objects
    :type paths: list(Path)
    :return: list of Path objects
    :rtype: list(Path)
    """
    return np.vstack([p() for p in paths]), np.concatenate([p.z_coordinates for p in paths], axis=0)


def inverse_path(path, last_spline_coords, t, between_grippers):
    """
    Get total length of paths and gaps.
    :param path: Path object
    :type path: Path
    :param last_spline_coords: spline coordinates from previous frame
    :type last_spline_coords: list
    :param t: path linspace
    :type t: np.array
    :param between_grippers: boolean which decides if take care only about the cable
                             between horizontally oriented grippers
                             (extracts the part of a cable between extreme spline extrema)
    :type between_grippers: bool
    :return: spline coordinates, spline parameters
    :rtype: list, dict
    """
    spline_coords = path.get_spline(t=t, between_grippers=between_grippers)
    spline_params = path.get_spline_params()
    if last_spline_coords is not None:
        dist1 = np.sum(np.abs(last_spline_coords[0] - spline_coords[0]) + np.abs(last_spline_coords[-1] - spline_coords[-1]))
        dist2 = np.sum(np.abs(last_spline_coords[-1] - spline_coords[0]) + np.abs(last_spline_coords[0] - spline_coords[-1]))
        if dist2 < dist1:
            spline_coords = spline_coords[::-1]
            spline_params['coeffs'] = spline_params['coeffs'][:, ::-1]
    return spline_coords, spline_params


def get_paths_and_gaps_length(paths, gaps):
    """
    Get total length of paths and gaps.
    :param paths: list of Path objects
    :type paths: list(Path)
    :param gaps: all gaps lengths between paths
    :type gaps: list
    :return: path length
    :rtype: float
    """
    return np.sum([p.length for p in paths]) + np.sum(gaps)
