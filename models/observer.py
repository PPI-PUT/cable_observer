from math import pi
from time import time

import tensorflow as tf
import numpy as np

from matplotlib import pyplot as plt

from utils.utils import poly3_params


class CableObserverNetwork(tf.keras.Model):
    def __init__(self):
        super(CableObserverNetwork, self).__init__()
        self.features = [
            tf.keras.layers.Conv2D(16, 3, padding='same', activation=tf.keras.activations.relu),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Conv2D(32, 3, padding='same', activation=tf.keras.activations.relu),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Conv2D(64, 3, padding='same', activation=tf.keras.activations.relu),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Conv2D(128, 3, padding='same', activation=tf.keras.activations.relu),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Conv2D(256, 3, padding='same', activation=tf.keras.activations.relu),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Conv2D(512, 3, padding='same', activation=tf.keras.activations.relu),
        ]

        self.fc = [
            tf.keras.layers.Dense(1024, activation=tf.keras.activations.relu),
        ]

        self.uv = [
            tf.keras.layers.Dense(256, activation=tf.keras.activations.relu),
            tf.keras.layers.Dense(8, activation=tf.keras.activations.sigmoid),
        ]

        self.duv = [
            tf.keras.layers.Dense(256, activation=tf.keras.activations.relu),
            tf.keras.layers.Dense(8, activation=tf.keras.activations.tanh),
        ]

    def call(self, x):
        bs = x.shape[0]
        for layer in self.features:
            x = layer(x)
        x = tf.reshape(x, (bs, -1))
        for layer in self.fc:
            x = layer(x)
        uv = x
        for layer in self.uv:
            uv = layer(uv)
        duv = x
        for layer in self.duv:
            duv = layer(duv)
        return uv, duv


class Cable(tf.keras.Model):
    #def __init__(self, uv=np.array([0.5, 0.1, 0.5, 0.2, 0.5, 0.3, 0.5, 0.4]), duv=(np.random.random((8, )) - 0.5)):
    def __init__(self, uv=np.array([0.5, 0.0, 0.5, 0.3, 0.5, 0.6, 0.5, 1.0]), duv=np.array([0., 1., 0., 1., 0., 1., 0., 1.])):
        super(Cable, self).__init__()
        #self.uv = tf.Variable(np.random.random((1, 8)).astype(np.float32), trainable=True)
        self.uv = tf.Variable(uv.astype(np.float32)[np.newaxis], trainable=True, name="uv")
        self.duv = tf.Variable(duv.astype(np.float32)[np.newaxis], trainable=True, name="duv")

    def call(self, x):
        #return tf.keras.activations.sigmoid(self.uv), tf.keras.activations.tanh(self.duv)
        return self.uv, self.duv


def poly_loss(poly, data):
    #t1 = time()
    uvs, duvs = poly
    us = []
    vs = []
    #dus = []
    #dvs = []
    for i in range(3):
        #d1 = time()
        p1 = uvs[:, 2 * i:2 * (i + 1)]
        p2 = uvs[:, 2 * (i + 1):2 * (i + 2)]
        dp1 = duvs[:, 2 * i:2 * (i + 1)]
        dp2 = duvs[:, 2 * (i + 1):2 * (i + 2)]
        #d2 = time()
        a, b = poly3_params(p1, p2, dp1, dp2)
        #d3 = time()
        t = tf.linspace(0., 1., 64)
        s = tf.stack([t ** 3, t ** 2, t, tf.ones_like(t)], axis=-1)
        u = s @ a
        v = s @ b
        #ds = tf.stack([3 * t ** 2, 2 * t, tf.ones_like(t), tf.zeros_like(t)], axis=-1)
        #du = ds @ a
        #dv = ds @ b
        us.append(u)
        vs.append(v)
        #dus.append(du)
        #dvs.append(dv)
        #d4 = time()
    us = tf.concat(us, 1)
    vs = tf.concat(vs, 1)
    batch = []
    #d5 = time()
    #print("D1", d2 - d1)
    #print("D2", d3 - d2)
    #print("D3", d4 - d3)
    #print("D4", d5 - d4)
    #t2 = time()
    for i in range(data.shape[0]):
        idx = np.where(data[i, ..., 1] == 1.)
        idx_idx = np.random.randint(0, idx[0].shape[0] - 1, 256)
        idx_cu = idx[0][idx_idx]
        idx_cv = idx[1][idx_idx]
        idx_c = np.stack([idx_cu, idx_cv], axis=-1)
        batch.append(idx_c)
    #t3 = time()
    idx_c = np.stack(batch, axis=0) / np.array([data.shape[1], data.shape[2]])[np.newaxis, np.newaxis]
    uvs = tf.concat([us, vs], axis=-1)

    dist = tf.abs(uvs[:, :, tf.newaxis] - idx_c[:, tf.newaxis])
    dist = tf.reduce_sum(dist, axis=-1)
    shape_loss = tf.reduce_min(dist, axis=1)
    shape_loss = tf.reduce_mean(shape_loss, axis=-1)
    reverse_loss = tf.reduce_min(dist, axis=-1)
    reverse_loss = tf.reduce_mean(reverse_loss, axis=-1)
    #t4 = time()

    diff_u = us[:, 1:] - us[:, :-1]
    diff_v = vs[:, 1:] - vs[:, :-1]
    ds = tf.abs(diff_u) + tf.abs(diff_v)
    length = tf.reduce_sum(ds, axis=(1, 2))

    outside_loss = tf.nn.relu(uvs - 1.) + tf.nn.relu(-uvs)
    outside_loss = tf.reduce_sum(outside_loss, axis=(1, 2))

    model_loss = shape_loss + outside_loss + 1e0 * reverse_loss + 1e-2 * length
    #t5 = time()
    #print("T1", t2 - t1)
    #print("T2", t3 - t2)
    #print("T3", t4 - t3)
    #print("T4", t5 - t4)
    return model_loss, us, vs, shape_loss, outside_loss, reverse_loss, length, 0.


def _plot(u, v, data):
    bs = data.shape[0]
    poly_img = np.zeros_like(data[..., :1])
    u = tf.clip_by_value(u, 0., 1.)
    u = tf.cast(tf.round(u * (data.shape[1] - 1)), tf.int32).numpy()
    v = tf.clip_by_value(v, 0., 1.)
    v = tf.cast(tf.round(v * (data.shape[2] - 1)), tf.int32).numpy()
    poly_img[np.arange(bs)[:, np.newaxis], u[np.arange(bs), :, 0], v[np.arange(bs), :, 0]] = 1.
    #plt.imshow(poly_img[0, ..., 0])
    #plt.show()
    return poly_img
