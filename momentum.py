from time import time
from scipy.interpolate import UnivariateSpline
from skimage.morphology import skeletonize

import numpy as np
import matplotlib.pyplot as plt

r = 15
#img = plt.imread("a1.png")[..., 0]
#start = (156, 324)
img = plt.imread("test.png")[..., 0]
start = (140, 103)
#img = plt.imread("0019.png")#[..., 0]
#img = plt.imread("zgiecie.png")#[..., 0]
#start = (304, 242)
#start = (286, 223)
#plt.imshow(img)
#plt.show()
img = skeletonize(img)

#plt.subplot(121)
plt.imshow(img)
plt.plot(start[0], start[1], 'rx')
path = [start]
t1 = time()
colors = ['r', 'g', 'b']
aim = np.array([0, 0])
patch = np.array([True, True])
radius = 1
for i in range(100):
    #print(i)
    act = np.array([path[-1][1], path[-1][0]])
    act_r = np.around(act).astype(np.int32)
    if i > 0:
        prev = np.array([path[-2][1], path[-2][0]])
        new = act + (act - prev)
        new = np.around(new).astype(np.int32)
        d = r - 2
        patch = img[new[0] - d: new[0] + d + 1, new[1] - d: new[1] + d + 1]
        xy = np.meshgrid(np.linspace(new[1] - d, new[1] + d, 2 * d + 1), np.linspace(new[0] - d, new[0] + d, 2 * d + 1))
        xy = np.stack(xy, axis=-1)
        xy = xy[:, :, ::-1]
        dist = np.linalg.norm(xy - act[np.newaxis, np.newaxis], axis=-1)
        radius = np.logical_and(r - 2 < dist, dist < r + 2).astype(np.float32)
        aim = np.sum(xy * patch[..., np.newaxis] * radius[..., np.newaxis], axis=(0, 1)) / np.sum(patch * radius)
    aim_r = np.around(aim).astype(np.int32)
    aim_r_neighbourhood = img[aim_r[0] - 1: aim_r[0] + 2, aim_r[1] - 1: aim_r[1] + 2].astype(np.int32)
    aim_missed = np.sum(aim_r_neighbourhood) == 0 or not (patch * radius).any()
    if i == 0 or aim_missed:
        pts = []
        n = np.linspace(-r, r, 2*r + 1)[img[act_r[0] - r: act_r[0] + r + 1, act_r[1] - r].astype(np.bool)]
        n_pts = np.stack([act_r[0] + n, (act_r[1] - r) * np.ones_like(n)], axis=-1)
        pts.append(n_pts)
        e = np.linspace(-r, r, 2*r + 1)[img[act_r[0] + r, act_r[1] - r: act_r[1] + r + 1].astype(np.bool)]
        e_pts = np.stack([(act_r[0] + r) * np.ones_like(e), act_r[1] + e], axis=-1)
        pts.append(e_pts)
        s = np.linspace(-r, r, 2*r + 1)[img[act_r[0] - r: act_r[0] + r + 1, act_r[1] + r].astype(np.bool)]
        s_pts = np.stack([act_r[0] + s, (act_r[1] + r) * np.ones_like(s)], axis=-1)
        pts.append(s_pts)
        w = np.linspace(-r, r, 2*r + 1)[img[act_r[0] - r, act_r[1] - r: act_r[1] + r + 1].astype(np.bool)]
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

    plt.plot([act[1], aim[1]], [act[0], aim[0]], colors[i % 3])
    path.append((aim[1], aim[0]))
t2 = time()
print(path)
print(t2 - t1)
path = path[:10] + path[20:]
xys = np.stack(path, axis=0)
t = np.linspace(0., 1., xys.shape[0])
T = np.linspace(0., 1., 128)
k = 1
#k = 10
#k = 100
x_spline = UnivariateSpline(t, xys[:, 0], s=k)
y_spline = UnivariateSpline(t, xys[:, 1], s=k)
x = x_spline(T)
y = y_spline(T)
t3 = time()
print(t3 - t1)
plt.plot(x, y, 'g')
#plt.subplot(122)
#plt.plot(T, x, label="x")
#plt.plot(T, y, label="y")
#plt.legend()

plt.show()