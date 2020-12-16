from time import time

import tensorflow as tf
import numpy as np


def yaml2list(q):
    return q["x"], q["y"], q["z"], q["w"]


def Rot(fi):
    c = tf.cos(fi)
    s = tf.sin(fi)
    L = tf.stack([c, s], -1)
    R = tf.stack([-s, c], -1)
    return tf.stack([L, R], -1)


def poly3_params(uv1, uv2, duv1, duv2):
    u1 = uv1[:, 0]
    v1 = uv1[:, 1]
    u2 = uv2[:, 0]
    v2 = uv2[:, 1]
    du1 = duv1[:, 0]
    dv1 = duv1[:, 1]
    du2 = duv2[:, 0]
    dv2 = duv2[:, 1]

    A = np.array([[0, 0, 0, 1], [0, 0, 1, 0], [1, 1, 1, 1], [3, 2, 1, 0]], np.float32)[np.newaxis]
    u = tf.stack([u1, du1, u2, du2], axis=-1)[..., np.newaxis]
    a = tf.linalg.solve(A, u)

    v = tf.stack([v1, dv1, v2, dv2], axis=-1)[..., np.newaxis]
    b = tf.linalg.solve(A, v)
    return a, b


def poly3local_params(xk, yk, thk):
    o = tf.ones_like(xk)
    z = tf.zeros_like(xk)
    a1 = tf.stack([z, z, z, o], axis=-1)
    a2 = tf.stack([z, z, o, z], axis=-1)
    a3 = tf.stack([3 * xk ** 2, 2 * xk, o, z], axis=-1)
    a4 = tf.stack([xk ** 3, xk ** 2, xk, o], axis=-1)
    A = tf.stack([a1, a2, a3, a4], axis=-1)
    dy = tf.tan(thk)
    u = tf.stack([z, z, dy, yk], axis=-1)[..., np.newaxis]
    a = tf.linalg.inv(A) @ u
    return a
