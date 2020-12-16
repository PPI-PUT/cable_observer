from random import random
from time import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
from tqdm import tqdm

N = 7

t = np.linspace(0., 1., 128)
#path = "../data/train/bezier/"
path = "../data/val/bezier/"
for j in tqdm(range(1000)):
    cp = [[0.05, 0.5]]
    dir = 0.0
    max_step = 0.1
    max_dangle = 1.0
    c_drift = 0.4 * 2 * (np.random.random() - 0.5)
    for i in range(N):
        ncp = cp[-1] + (max_step * random() + 0.05) * np.array([np.cos(dir), np.sin(dir)])# + 0.05
        dir += max_dangle * 2 * (random() - 0.5) + c_drift
        cp.append(ncp)


    def B(n, i, t):
        return comb(n, i) * t ** i * (1 - t) ** (n - i)


    # calculate bezier curve
    Bs = np.stack([B(N, k, t) for k in range(0, N + 1)], axis=-1)
    uvs = Bs @ cp

    U, S, V = np.linalg.svd(uvs)
    mean = np.mean(uvs, 0, keepdims=True)
    ruvs = uvs - mean
    cov = ruvs.T @ ruvs
    U, S, V = np.linalg.svd(cov)
    ruvs = (V @ ruvs.T).T
    ruvs += 0.5

    #plt.subplot(121)
    #us = uvs[..., 0]# * W
    #vs = uvs[..., 1]# * H
    #plt.plot(us, vs, linewidth=7.)
    #ax = plt.gca()
    #ax.set_facecolor((.0, 0., 0.))
    #plt.xlim(0., 1.)
    #plt.ylim(0., 1.)
    #plt.subplot(122)

    rus = ruvs[..., 0]# * W
    rvs = ruvs[..., 1]# * H
    plt.plot(rus, rvs, linewidth=1. + 3*np.random.random())
    w, h = 5.97, 6
    plt.gcf().set_size_inches(w, h)
    ax = plt.gca()
    ax.set_facecolor((.0, 0., 0.))
    plt.xlim(0., 1.)
    plt.ylim(0., 1.)


    cps = cp - mean
    cps = (V @ cps.T).T
    cps += 0.5

    plt.axis("off")
    #plt.plot(cps[:, 0], cps[:, 1], "rx")
    plt.savefig(path + str(j).zfill(4) + ".png", bbox_inches='tight', pad_inches=0)
    np.savetxt(path + str(j).zfill(4) + ".cps", cps)
    plt.clf()