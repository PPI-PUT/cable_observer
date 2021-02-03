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


def walk(img, skel, start, r, d):
    pad = int(1.5*r)
    skel = np.pad(skel, [[pad, pad], [pad, pad]], 'constant', constant_values=False)
    img = np.pad(img, [[pad, pad], [pad, pad]], 'constant', constant_values=False)
    path = [(int(start[1]) + pad, int(start[0]) + pad)]
    #path = [start]
    #colors = ['r', 'g', 'b']
    aim = np.array([0, 0])
    patch = np.array([True, True])
    radius = 1
    length = 0.
    for i in range(100):
        #act = np.array([path[-1][1], path[-1][0]])
        act = np.array(path[-1])
        act_r = np.around(act).astype(np.int32)
        if i > 0:
            #prev = np.array([path[-2][1], path[-2][0]])
            prev = np.array(path[-2])
            new = act + (act - prev)
            new = np.around(new).astype(np.int32)
            patch = skel[new[0] - d: new[0] + d + 1, new[1] - d: new[1] + d + 1]
            xy = np.meshgrid(np.linspace(new[1] - d, new[1] + d, 2 * d + 1),
                             np.linspace(new[0] - d, new[0] + d, 2 * d + 1))
            xy = np.stack(xy, axis=-1)
            xy = xy[:, :, ::-1]
            dist = np.linalg.norm(xy - act[np.newaxis, np.newaxis], axis=-1)
            radius = np.logical_and(r - 2 < dist, dist < r + 2).astype(np.float32)
            print(xy.shape)
            print(patch.shape)
            print(radius.shape)
            aim = np.sum(xy * patch[..., np.newaxis] * radius[..., np.newaxis], axis=(0, 1)) / np.sum(patch * radius)
        aim_r = np.around(aim).astype(np.int32)
        aim_r_neighbourhood = skel[aim_r[0] - 1: aim_r[0] + 2, aim_r[1] - 1: aim_r[1] + 2].astype(np.int32)
        aim_missed = np.sum(aim_r_neighbourhood) == 0 or not (patch * radius).any()
        if i == 0 or aim_missed:
            pts = []
            n = np.linspace(-r, r, 2 * r + 1)[skel[act_r[0] - r: act_r[0] + r + 1, act_r[1] - r].astype(np.bool)]
            n_pts = np.stack([act_r[0] + n, (act_r[1] - r) * np.ones_like(n)], axis=-1)
            pts.append(n_pts)
            e = np.linspace(-r, r, 2 * r + 1)[skel[act_r[0] + r, act_r[1] - r: act_r[1] + r + 1].astype(np.bool)]
            e_pts = np.stack([(act_r[0] + r) * np.ones_like(e), act_r[1] + e], axis=-1)
            pts.append(e_pts)
            s = np.linspace(-r, r, 2 * r + 1)[skel[act_r[0] - r: act_r[0] + r + 1, act_r[1] + r].astype(np.bool)]
            s_pts = np.stack([act_r[0] + s, (act_r[1] + r) * np.ones_like(s)], axis=-1)
            pts.append(s_pts)
            w = np.linspace(-r, r, 2 * r + 1)[skel[act_r[0] - r, act_r[1] - r: act_r[1] + r + 1].astype(np.bool)]
            w_pts = np.stack([(act_r[0] - r) * np.ones_like(w), act_r[1] + w], axis=-1)
            pts.append(w_pts)
            pts = np.concatenate(pts, axis=0)

            if len(path) > 1:
                #prev = np.array([path[-2][1], path[-2][0]])
                prev = np.array(path[-2])
                dists = np.sum(np.abs(pts - prev), axis=-1)
                pts = pts[dists >= r - 1]

            if len(pts) == 0:
                break
            aim = np.mean(pts, axis=0)
        x_path = np.round(np.linspace(act[1], aim[1], 10)).astype(np.int32)
        y_path = np.round(np.linspace(act[0], aim[0], 10)).astype(np.int32)
        img_path = img[y_path, x_path]
        if np.mean(img_path) < 0.8:
            break

        length += np.linalg.norm(np.array(act) - np.array(aim))
        #plt.plot([act[1], aim[1]], [act[0], aim[0]], colors[i % 3])
        path.append((aim[0], aim[1]))
    path = [(a[0] - pad, a[1] - pad) for a in path]
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


def get_errors(spline_params, spline_params_buffer):
    """
    Get errors:
        * length_error - length difference
        * coeffs_error_max - max coefficients difference [x, y]
        * residuals_error - residuals difference [x, y]
        * derivatives_error_max - max 1st derivative difference within set of control points [x, y]
    :param spline_params: current spline params
    :type spline_params: np.array
    :param spline_params_buffer: previous spline params
    :type spline_params_buffer: np.array
    :return: spline errors
    :rtype: dict
    """
    length_error = np.absolute(spline_params['length'] - spline_params_buffer['length'])
    coeffs_error_sum = np.absolute(spline_params['coeffs'] - spline_params_buffer['coeffs'])
    coeffs_error_max = np.max(coeffs_error_sum, axis=1)
    residuals_error = np.absolute(spline_params['residuals'] - spline_params_buffer['residuals'])
    derivatives_error = np.absolute(spline_params['derivatives'] - spline_params_buffer['derivatives'])
    derivatives_error_max = np.max(derivatives_error[..., 0], axis=1)
    errors = {"length_error": length_error, "coeffs_error_max":  coeffs_error_max, "residuals_error": residuals_error,
              "derivatives_error_max": derivatives_error_max}
    return errors