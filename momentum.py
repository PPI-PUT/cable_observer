from time import time, sleep
import sys
from copy import deepcopy
from scipy.interpolate import LSQUnivariateSpline
from skimage.morphology import skeletonize, medial_axis, erosion, dilation, square
import cv2
import numpy as np
import matplotlib.pyplot as plt


def find_ends(img, max_px_gap):
    """
    Find the ends of paths assuming that each end has one neighbor.
    :param img: skeletonized paths
    :type img: np.ndarray
    :param max_px_gap: maximum pixel gap between paths ends in terms of remove similar points
    :type max_px_gap: int
    :return: all coordinates with a single neighbor
    :rtype: list
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
    :type path_ends: np.ndarray
    :param max_px_gap: maximum pixel gap between last_point and a single point of path_ends
    :type max_px_gap: int
    :return: points without close points to the last point
    :rtype: list
    """
    keys_to_remove = []
    for key, value in enumerate(np.array(path_ends)):
        if (np.fabs(np.array(last_point) - value) < max_px_gap).all():
            keys_to_remove.append(key)
    path_ends = np.delete(np.array(path_ends), keys_to_remove, axis=0)
    return path_ends.tolist()


def merge_paths(paths_c):
    best_connections = []
    for key, value in enumerate(paths_c):
        for key2, value2 in enumerate(paths_c[key + 1:]):
            connections = []

            # Calculate gap length between two paths (begins and ends)
            gap_length = np.linalg.norm(np.subtract(value['coords'][0], value2['coords'][0]))
            connections.append({"gap_length": gap_length, "keys": [key, key + 1 + key2]})
            gap_length = np.linalg.norm(np.subtract(value['coords'][0], value2['coords'][-1]))
            connections.append({"gap_length": gap_length, "keys": [key, key + 1 + key2]})
            gap_length = np.linalg.norm(np.subtract(value['coords'][-1], value2['coords'][0]))
            connections.append({"gap_length": gap_length, "keys": [key, key + 1 + key2]})
            gap_length = np.linalg.norm(np.subtract(value['coords'][-1], value2['coords'][-1]))
            connections.append({"gap_length": gap_length, "keys": [key, key + 1 + key2]})

            # Chose best connection - shortest gap
            best_connection = min(connections, key=lambda k: k['gap_length'])
            best_connections.append(best_connection)

    # Sort connections between paths by shortest gaps
    best_connections = sorted(best_connections, key=lambda k: k['gap_length'])

    # The assumption that each path has one connection minimum and two maximum
    # Find out at least one connection for all of single paths is a priority
    first_occurrence = []
    all_occurrences = []
    connections_to_remove = []
    best_connections_workspace = deepcopy(best_connections)
    true_best_connections = []
    for key, value in enumerate(best_connections):
        key_0 = value['keys'][0]
        key_1 = value['keys'][1]
        if (first_occurrence.count(key_0) < 1 or first_occurrence.count(key_1) < 1) \
                and all_occurrences.count(key_0) < 2 and all_occurrences.count(key_1) < 2:
            true_best_connections.append(value)
            first_occurrence.append(key_0)
            first_occurrence.append(key_1)
            first_occurrence = list(set(first_occurrence))
            all_occurrences.append(key_0)
            all_occurrences.append(key_1)
            connections_to_remove.append(key)
    best_connections_workspace = list(np.delete(np.array(best_connections_workspace), connections_to_remove))

    # Append rest of connections
    for connection in best_connections_workspace:
        true_best_connections.append(connection)

    # Get rid of unnecessary connections -> with N paths there is always N - 1 connections
    true_best_connections = true_best_connections[:len(paths_c) - 1]

    # Find first path with a single connection
    paths_keys_order = []
    keys = np.hstack([d['keys'] for d in true_best_connections])

    for key in range(len(paths_c)):
        if np.sum(keys == key) == 1:
            paths_keys_order.append(key)
            break

    # Find paths & connections order
    best_connections_workspace = deepcopy(true_best_connections)
    best_connections_lengths_ordered = []
    i = 0
    while len(best_connections_workspace) > 0:
        i += 1
        for key, value in enumerate(best_connections_workspace):
            if paths_keys_order[-1] in best_connections_workspace[key]['keys']:
                best_connections_workspace[key]['keys'].remove(paths_keys_order[-1])
                paths_keys_order.append(value['keys'][0])
                best_connections_lengths_ordered.append(best_connections_workspace.pop(key)['gap_length'])
                break
        if i > 10:
            pass

    # Reorder paths
    ordered_paths = [paths_c[i] for i in paths_keys_order]

    # Merge & flip paths
    merged_paths = [ordered_paths[0]['coords']]
    for key, value in enumerate([d['coords'] for d in ordered_paths][1:]):
        if np.linalg.norm(np.subtract(merged_paths[-1][-1], value[0])) > np.linalg.norm(
                np.subtract(merged_paths[-1][0], value[0])):
            merged_paths[-1] = np.flip(merged_paths[-1], axis=0).tolist()
        if np.linalg.norm(np.subtract(merged_paths[-1][-1], value[0])) > np.linalg.norm(
                np.subtract(merged_paths[-1][-1], value[-1])):
            merged_paths.append(np.flip(value, axis=0).tolist())
        else:
            merged_paths.append(value)

    merged_paths = np.vstack(np.array(merged_paths, dtype=object))

    return ordered_paths, merged_paths.tolist(), best_connections_lengths_ordered


def get_linespaces(ordered_paths, gaps_length):
    """
    Get linespace for merged paths.
    :param ordered_paths: all paths with correct order
    :type ordered_paths: list
    :param gaps_length: all gaps lengths between paths
    :type gaps_length: list
    :return: merged linespaces of all paths
    :rtype: np.array
    """
    full_length = np.sum([d["length"] for d in ordered_paths]) + np.sum(gaps_length)
    t_linespaces = []
    curr_value = 0
    for key, value in enumerate([d["length"] for d in ordered_paths]):
        t_linespace = np.linspace(curr_value, curr_value + value / full_length,
                                  [d["num_points"] for d in ordered_paths][key])
        curr_value += value / full_length
        if key < len(gaps_length):
            curr_value += gaps_length[key] / full_length
        t_linespaces.append(t_linespace)
    t = np.hstack(t_linespaces)
    return t


def walk(img, skel, start, r, d):
    path = [start]
    colors = ['r', 'g', 'b']
    aim = np.array([0, 0])
    patch = np.array([True, True])
    radius = 1
    length = 0.
    for i in range(100):
        act = np.array([path[-1][1], path[-1][0]])
        act_r = np.around(act).astype(np.int32)
        if i > 0:
            prev = np.array([path[-2][1], path[-2][0]])
            new = act + (act - prev)
            new = np.around(new).astype(np.int32)
            patch = skel[new[0] - d: new[0] + d + 1, new[1] - d: new[1] + d + 1]
            xy = np.meshgrid(np.linspace(new[1] - d, new[1] + d, 2 * d + 1),
                             np.linspace(new[0] - d, new[0] + d, 2 * d + 1))
            xy = np.stack(xy, axis=-1)
            xy = xy[:, :, ::-1]
            dist = np.linalg.norm(xy - act[np.newaxis, np.newaxis], axis=-1)
            radius = np.logical_and(r - 2 < dist, dist < r + 2).astype(np.float32)
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
                prev = np.array([path[-2][1], path[-2][0]])
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
        plt.plot([act[1], aim[1]], [act[0], aim[0]], colors[i % 3])
        path.append((aim[1], aim[0]))
    return path, length


def walk_fast(skel, start, img):
    length = 0
    path = [(int(start[1]), int(start[0]))]
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
            dir = v / np.linalg.norm(v)
            new = np.array(act) + dir
            dists = np.linalg.norm(new - b, axis=-1)
            min_idx = np.argmin(dists)
            aim = b[min_idx].astype(np.int32)
        else:
            break

        length += np.linalg.norm(np.array(act) - np.array(aim))
        #plt.plot([act[1], aim[1]], [act[0], aim[0]], colors[i % 3])
        path.append((aim[0], aim[1]))
        i += 1
    return path, length


def main(img):
    # img = plt.imread("test_v2.png")
    # img = plt.imread("test_v3.png")
    # img = plt.imread("test_v4.png")
    # img = plt.imread("test_v5.png")
    # img = plt.imread("test_v6.png")
    # plt.subplot(121)
    # plt.imshow(img)

    t1 = time()
    img_copy = deepcopy(img)
    img = erosion(img)
    img = dilation(img)
    img = dilation(img)
    img = dilation(img)
    img = dilation(img)
    skel = skeletonize(img)
    t2 = time()
    path_ends = find_ends(img=skel, max_px_gap=5)
    t3 = time()
    path_ends_workspace = path_ends.copy()

    paths = []

    while len(path_ends_workspace) > 0:
        #path, path_length = walk(img, skel, tuple(path_ends_workspace[0]), 10, 3)
        path, path_length = walk_fast(skel, tuple(path_ends_workspace[0]), img_copy)
        paths.append({"coords": path, "num_points": len(path), "length": path_length})
        path_ends_workspace.pop(0)
        path_ends_workspace = remove_close_points(path[-1], path_ends_workspace)

    paths = [p for p in paths if len(p['coords']) > 1]
    t4 = time()

    if len(paths) > 1:
        ordered_paths, merged_paths, gaps_length = merge_paths(paths)
        xys = np.stack(merged_paths, axis=0)
        t = get_linespaces(ordered_paths, gaps_length)
    else:
        xys = np.stack(paths[0]['coords'], axis=0)
        t = np.linspace(0., 1., xys.shape[0])

    t5 = time()
    T = np.linspace(0., 1., 128)
    k = 7
    knots = np.linspace(0., 1., k)[1:-1]
    x_spline = LSQUnivariateSpline(t, xys[:, 1], knots)
    y_spline = LSQUnivariateSpline(t, xys[:, 0], knots)
    x = x_spline(T)
    y = y_spline(T)
    t6 = time()

    print("SKELETONIZE:", t2 - t1)
    #print("FIND ENDS:", t3 - t2)
    #print("WALK:", t4 - t3)
    #print("MERGE:", t5 - t4)
    #print("FIT SPLINE:", t6 - t5)
    print("SUM:", t6 - t1)

    # plt.plot(x, y, 'g')
    # plt.subplot(122)
    # plt.plot(T, x, label="x")
    # plt.plot(t, xys[:, 1], 'x', label="px")
    # plt.plot(T, y, label="y")
    # plt.plot(t, xys[:, 0], 'x', label="py")
    # plt.legend()
    # plt.show()
    return x, y

def get_spline(x, y, width, height):
    """
    Get an array with (x, y) coordinates set in.
    :param x:
    :type x:
    :param y:
    :type y:
    :param width:
    :type width:
    :param height:
    :type height:
    :return:
    :rtype:
    """
    spline_frame = np.zeros(shape=(height, width))
    keys_to_remove = []

    for key, value in enumerate(x):
        if x[key] > height or x[key] < 0 or y[key] > width or y[key] < 0:
            keys_to_remove.append(key)
    x = np.delete(x, keys_to_remove)
    y = np.delete(y, keys_to_remove)

    spline_frame[tuple(x.astype(int)), tuple(y.astype(int))] = 1

    return spline_frame

def get_cable(frame):
    """
    Process image to extract cable path.
    :param frame: camera input frame
    :type frame: np.ndarray
    :return: Preprocessed camera frame
    :rtype: np.ndarray
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    o = np.ones_like(hsv)[..., 0].astype(np.float32)
    z = np.zeros_like(o)
    img = np.where(np.logical_and(np.logical_or(hsv[..., 0] < 7., hsv[..., 0] > 170.), hsv[..., 1] > 120), o, z)
    return img

if __name__ == "__main__":
    cap = cv2.VideoCapture(6)

    # Skip blank frames
    for i in range(100):
        ret, frame = cap.read()

    while True:
        ret, frame = cap.read()

        # Preprocess image
        img = get_cable(frame)

        # Get spline coordinates
        x, y = main(img)

        # Convert spline coordinates to image frame
        spline_frame = get_spline(x=y, y=x, width=640, height=480)

        # Draw outputs
        cv2.imshow('spline', spline_frame)
        cv2.imshow('frame', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
           break

    cap.release()
    cv2.destroyAllWindows()