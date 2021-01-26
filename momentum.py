from time import time
import sys
from scipy.interpolate import LSQUnivariateSpline
from skimage.morphology import skeletonize, medial_axis

import numpy as np
import matplotlib.pyplot as plt


def find_ends(img, max_px_gap):
    # Find all coordinates with less than 2 neighbors
    path_indices = np.nonzero(img)
    path_indices = np.vstack((path_indices[0], path_indices[1])).T
    path_ends_c = np.empty((0, 2))
    for current_cord in path_indices:
        neighbors = 0
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                if img[current_cord[0] + i, current_cord[1] + j]:
                    neighbors += 1
        if neighbors < 2:
            path_ends_c = np.append(path_ends_c, [np.flip(current_cord)], axis=0)

    # Skip similar coordinates using maximum gap between points
    keys_to_skip = []
    for key, value in enumerate(path_ends_c):
        for current_cord in path_ends_c[key + 1:]:
            if (np.fabs(current_cord - value) <= max_px_gap).all():
                keys_to_skip.append(key)
    path_ends_c = np.delete(path_ends_c, keys_to_skip, axis=0)

    path_ends_c = list(map(list, path_ends_c))
    return path_ends_c


def remove_close_points(last_point, path_ends_c, max_px_gap=10):
    keys_to_remove = []
    for key, value in enumerate(np.array(path_ends_c)):
        if (np.fabs(np.array(last_point) - value) < max_px_gap).all():
            keys_to_remove.append(key)
    path_ends_c = np.delete(np.array(path_ends_c), keys_to_remove, axis=0)
    return path_ends_c.tolist()


# TODO
def merge_paths(paths_c, paths_length):
    merged_path = []
    best_connections = []
    for key, value in enumerate(paths_c):
        for key2, value2 in enumerate(paths_c[key + 1:]):
            connection = {}
            distance = sys.maxsize
            if np.linalg.norm(np.subtract(value[0], value2[0])) < distance:
                distance = np.linalg.norm(np.subtract(value[0], value2[0]))
                connection = {"gap_length": distance, "key_1": key, "key_2": key + 1 + key2,
                              "idx_1": 0, "idx_2": 0}
            if np.linalg.norm(np.subtract(value[0], value2[-1])) < distance:
                distance = np.linalg.norm(np.subtract(value[0], value2[-1]))
                connection = {"gap_length": distance, "key_1": key, "key_2": key + 1 + key2,
                              "idx_1": 0, "idx_2": -1}
            if np.linalg.norm(np.subtract(value[-1], value2[0])) < distance:
                distance = np.linalg.norm(np.subtract(value[-1], value2[0]))
                connection = {"gap_length": distance, "key_1": key, "key_2": key + 1 + key2,
                              "idx_1": -1, "idx_2": 0}
            if np.linalg.norm(np.subtract(value[-1], value2[-1])) < distance:
                distance = np.linalg.norm(np.subtract(value[-1], value2[-1]))
                connection = {"gap_length": distance, "key_1": key, "key_2": key + 1 + key2,
                              "idx_1": -1, "idx_2": -1}
            best_connections.append(connection)
    best_connections = sorted(best_connections, key=lambda k: k['gap_length'])
    best_connections = best_connections[:len(best_connections) - 1]

    # TODO
    # HERE HARD CODED FOR TESTS
    # for connection in best_connections:
    paths_c[2].reverse()
    merged_path.append(paths_c[2])
    paths_c[0].reverse()
    merged_path.append(paths_c[0])
    merged_path.append(paths_c[1])
    merged_path = np.array(merged_path, dtype=object)
    merged_path = np.vstack(merged_path)

    paths_length = [paths_length[2], paths_length[0], paths_length[1]]
    gaps_length = [best_connections[1]['gap_length'], best_connections[0]['gap_length']]
    paths_points_num = [len(paths_c[2]), len(paths_c[0]), len(paths_c[1])]
    return merged_path.tolist(), paths_length, gaps_length, paths_points_num


def get_linespaces(paths_length, paths_points_num, gaps_length, T_scale=128):
    all_length = np.sum(paths_length) + np.sum(gaps_length)
    paths_points_num_scaled = (np.array(paths_points_num) * T_scale / np.sum(np.array(paths_points_num))).astype(
        int).tolist()
    t_linespaces = []
    curr_value = 0
    for key, value in enumerate(paths_length):
        t_linespace = np.linspace(curr_value, curr_value + value / all_length, paths_points_num[key])
        curr_value += value / all_length
        if key < len(gaps_length):
            curr_value += gaps_length[key] / all_length
        t_linespaces.append(t_linespace)

    # t = np.linspace(0., 1., xys.shape[0])
    # T = np.linspace(0., 1., 128)
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

def walk_fast(skel, start):
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
            v = np.array(act) - np.array(path[-5])
            dir = v / np.linalg.norm(v)
            new = np.array(act) + dir
            dists = np.linalg.norm(new - b, axis=-1)
            min_idx = np.argmin(dists)
            aim = b[min_idx].astype(np.int32)
        else:
            break

        length += np.linalg.norm(np.array(act) - np.array(aim))
        # plt.plot([start[0], aim[1]], [start[1], aim[0]], 'b')
        plt.plot([act[1], aim[1]], [act[0], aim[0]], colors[i % 3])
        path.append((aim[0], aim[1]))
        i += 1
    return path, length


# img = plt.imread("a1.png")[..., 0]
# start = (156, 324)
img = plt.imread("test_v3.png")
# img = plt.imread("test1.png")[..., 0]
start = (140, 103)
# start = (218, 154)
# img = plt.imread("0019.png")#[..., 0]
# img = plt.imread("zgiecie.png")#[..., 0]
# start = (304, 242)
# start = (286, 223)
# img = plt.imread("u.png")#[..., 0]
# img = plt.imread("u_hard.png")#[..., 0]
# start = (62, 117)
# plt.imshow(img)
# plt.show()
skel = skeletonize(img)

t_start = time()

path_ends = find_ends(img=skel, max_px_gap=5)
print(path_ends)

t_duration = time() - t_start
print("Find ends duration: ", t_duration)

plt.subplot(121)
plt.imshow(img)
# plt.plot(start[0], start[1], 'rx')

paths = []
paths_length = []
path_ends_workspace = path_ends.copy()
t1 = time()

while len(path_ends_workspace) > 0:
    #path, length = walk(img, skel, tuple(path_ends_workspace[0]), 10, 3)
    path, path_length = walk_fast(skel, tuple(path_ends_workspace[0]))
    paths_length.append(path_length)
    paths.append(path)
    path_ends_workspace.pop(0)
    path_ends_workspace = remove_close_points(path[-1], path_ends_workspace)

merged_paths, paths_length, gaps_length, paths_points_num = merge_paths(paths, paths_length)
print(merged_paths)
print(paths_length)
t2 = time()
print(t2 - t1)
xys = np.stack(merged_paths, axis=0)

t = get_linespaces(paths_length, paths_points_num, gaps_length)
T = np.linspace(0., 1., 128)
k = 7
knots = np.linspace(0., 1., k)[1:-1]
x_spline = LSQUnivariateSpline(t, xys[:, 0], knots)
#x_p = np.polyfit(t, xys[:, 0], k)
y_spline = LSQUnivariateSpline(t, xys[:, 1], knots)
#y_p = np.polyfit(t, xys[:, 1], k)
x = x_spline(T)
y = y_spline(T)
#x = np.polyval(x_p, T)
#y = np.polyval(y_p, T)
t3 = time()
print(t3 - t1)
plt.plot(x, y, 'g')
plt.subplot(122)
plt.plot(T, x, label="x")
plt.plot(t, xys[:, 0], 'x', label="px")
plt.plot(T, y, label="y")
plt.plot(t, xys[:, 1], 'x', label="py")
plt.legend()

plt.show()
