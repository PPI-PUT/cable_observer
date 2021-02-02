import numpy as np


def find_ends(img, max_px_gap):
    """
    Find the ends of paths assuming that each end has one neighbor.
    :param img: skeletonized paths
    :type img: np.array
    :param max_px_gap: maximum pixel gap between paths ends in terms of remove similar points
    :type max_px_gap: int
    :return: all coordinates with a single neighbor
    :rtype: np.array
    """
    # Add border to image frame
    img = np.pad(img, pad_width=1, mode='constant', constant_values=0)

    # Find all coordinates with less than 2 neighbors
    path_indices = np.nonzero(img)
    path_indices = np.vstack((path_indices[0], path_indices[1])).T
    path_ends_workspace = np.empty((0, 2))
    for current_cord in path_indices:
        neighbors = 0
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                if img[current_cord[0] + i, current_cord[1] + j]:
                    neighbors += 1
        if neighbors < 2:
            path_ends_workspace = np.append(path_ends_workspace, [np.flip(current_cord)], axis=0)

    # Skip similar coordinates using maximum gap between points
    keys_to_skip = []
    for key, value in enumerate(path_ends_workspace):
        for current_cord in path_ends_workspace[key + 1:]:
            if (np.fabs(current_cord - value) <= max_px_gap).all():
                keys_to_skip.append(key)
    path_ends_workspace = np.delete(path_ends_workspace, keys_to_skip, axis=0)
    path_ends_workspace = list(map(list, path_ends_workspace - 1))
    return path_ends_workspace


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


def walk_fast(skel, start):
    skel = np.pad(skel, [[1, 1], [1, 1]], 'constant', constant_values=False)
    length = 0
    path = [(int(start[1]) + 1, int(start[0]) + 1)]
    colors = ['r', 'g', 'b']
    dxy = np.meshgrid(np.linspace(-1, 1, 3), np.linspace(-1, 1, 3))
    dxy = np.stack(dxy, axis=-1)
    dxy = dxy[:, :, ::-1]
    card = 1
    i = 0
    while card > 0:
        act = path[-1]

        skel[act[0], act[1]] = 0.
        if len(path) > 3:
            skel[path[-4][0], path[-4][1]] = 1.

        patch = skel[act[0] - 1: act[0] + 2, act[1] - 1: act[1] + 2]
        xy = dxy + act
        b = xy[patch]

        card = b.shape[0]
        if card == 1:
            aim = b[0].astype(np.int32)
        elif card > 1:
            v = np.array(act) - np.array(path[-min(5, len(path))])
            direction = v / np.linalg.norm(v)
            new = np.array(act) + direction
            dists = np.linalg.norm(new - b, axis=-1)
            min_idx = np.argmin(dists)
            aim = b[min_idx].astype(np.int32)
        else:
            break

        length += np.linalg.norm(np.array(act) - np.array(aim))
        path.append((aim[0], aim[1]))
        i += 1
    path = [(a[0] - 1, a[1] - 1) for a in path]
    return path, length


def keys_to_sequence(keys):
    """
    Remap keys sequence to indices sequence.
    :param keys: keys sequence
    :type keys: list
    :return: indices sequence
    :rtype: list
    """
    max_workspace = max(keys) + 1
    keys_workspace = keys
    sequence = np.zeros_like(keys)
    for key, value in enumerate(keys):
        idx_min = keys_workspace.index(min(keys_workspace))
        sequence[idx_min] = key
        keys_workspace[idx_min] = max_workspace
    return sequence


def reorder_paths(paths, sequence):
    """
    Change paths order based on new sequence.
    :param paths: non-ordered paths
    :type paths: list
    :param sequence: indices sequence
    :type sequence: list
    :return: ordered paths
    :rtype: list
    """
    new_paths_sequence = []
    for path_key in sequence:
        new_paths_sequence.append(paths[path_key])
    return new_paths_sequence


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
