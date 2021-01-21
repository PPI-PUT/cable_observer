from time import time
from scipy.interpolate import UnivariateSpline
from skimage.morphology import skeletonize

import numpy as np
import matplotlib.pyplot as plt

#img = plt.imread("a1.png")[..., 0]
#start = (156, 324)
#img = plt.imread("test.png")[..., 0]
#start = (140, 103)
img = plt.imread("0019.png")#[..., 0]
#img = plt.imread("zgiecie.png")#[..., 0]
#start = (304, 242)
#start = (286, 223)
start = (223, 286)
#plt.imshow(img)
#plt.show()
skel = skeletonize(img)

#plt.subplot(121)
plt.imshow(skel)
plt.plot(start[1], start[0], 'rx')
path = [start]
t1 = time()
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
        p = path[-5:]
        v = np.array(act) - np.array(path[-5])
        dir = v / np.linalg.norm(v)
        new = np.array(act) + dir
        dists = np.linalg.norm(new - b, axis=-1)
        min_idx = np.argmin(dists)
        aim = b[min_idx].astype(np.int32)
    else:
        break

    #plt.plot([start[0], aim[1]], [start[1], aim[0]], 'b')
    plt.plot([act[1], aim[1]], [act[0], aim[0]], colors[i % 3])
    path.append((aim[0], aim[1]))
    i += 1
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
x_spline = UnivariateSpline(t, xys[:, 1], s=k)
y_spline = UnivariateSpline(t, xys[:, 0], s=k)
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