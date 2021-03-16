import numpy as np
from copy import deepcopy


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
