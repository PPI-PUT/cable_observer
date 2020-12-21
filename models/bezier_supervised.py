from scipy.special import comb
import tensorflow as tf
import numpy as np

N = 7


# N = 15

class Encoder(tf.keras.Model):
    def __init__(self):
        super(Encoder, self).__init__()
        self.features = [
            tf.keras.layers.Conv2D(8, 3, padding='same', activation=tf.keras.activations.tanh),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Conv2D(16, 3, padding='same', activation=tf.keras.activations.tanh),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Conv2D(32, 3, padding='same', activation=tf.keras.activations.tanh),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Conv2D(64, 3, padding='same', activation=tf.keras.activations.tanh),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Conv2D(128, 3, padding='same', activation=tf.keras.activations.tanh),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Conv2D(256, 3, padding='same', activation=tf.keras.activations.tanh),
        ]

        self.fc = [
            #tf.keras.layers.Dense(1024, activation=tf.keras.activations.tanh),
            tf.keras.layers.Dense(256, activation=tf.keras.activations.tanh),
        ]

    def call(self, x):
        bs = x.shape[0]
        for layer in self.features:
            x = layer(x)
        x = tf.reshape(x, (bs, -1))
        for layer in self.fc:
            x = layer(x)
        return x

class Decoder(tf.keras.Model):
    def __init__(self):
        super(Decoder, self).__init__()

        self.fc = [
            #tf.keras.layers.Dense(1024, activation=tf.keras.activations.tanh),
            tf.keras.layers.Dense(4*4*256, activation=tf.keras.activations.tanh),
        ]

        self.features = [
            tf.keras.layers.Conv2DTranspose(128, 3, padding='same', activation=tf.keras.activations.tanh),
            tf.keras.layers.UpSampling2D(),
            tf.keras.layers.Conv2DTranspose(64, 3, padding='same', activation=tf.keras.activations.tanh),
            tf.keras.layers.UpSampling2D(),
            tf.keras.layers.Conv2DTranspose(32, 3, padding='same', activation=tf.keras.activations.tanh),
            tf.keras.layers.UpSampling2D(),
            tf.keras.layers.Conv2DTranspose(16, 3, padding='same', activation=tf.keras.activations.tanh),
            tf.keras.layers.UpSampling2D(),
            tf.keras.layers.Conv2DTranspose(8, 3, padding='same', activation=tf.keras.activations.tanh),
            tf.keras.layers.UpSampling2D(),
            tf.keras.layers.Conv2DTranspose(2, 3, padding='same', activation=None),
        ]

    def call(self, x):
        bs = x.shape[0]
        for layer in self.fc:
            x = layer(x)
        x = tf.reshape(x, (bs, 4, 4, 256))
        for layer in self.features:
            x = layer(x)
        return x


class CableNetwork(tf.keras.Model):
    def __init__(self):
        super(CableNetwork, self).__init__()
        self.encoder = Encoder()

        self.fc = [
            #tf.keras.layers.Dense(1024, activation=tf.keras.activations.tanh),
            tf.keras.layers.Dense(256, activation=tf.keras.activations.tanh),
            tf.keras.layers.Dense(256, activation=tf.keras.activations.tanh),
            tf.keras.layers.Dense(2 * (N + 1), activation=lambda x: tf.keras.activations.sigmoid(x)),
        ]


    def call(self, x):
        bs = x.shape[0]
        x = self.encoder(x)
        #x = tf.reshape(x, (bs, -1))
        for layer in self.fc:
            x = layer(x)
        uv = tf.reshape(x, (-1, N + 1, 2))
        return uv


def _plot(cps, img):
    """Plots uv curve on the image plane"""
    poly_img = np.zeros_like(img[..., 0])
    bs = img.shape[0]

    t = np.linspace(0., 1., 128)

    def B(n, i, t):
        return comb(n, i) * t ** i * (1 - t) ** (n - i)

    # calculate bezier curve
    Bs = np.stack([B(N, k, t) for k in range(0, N + 1)], axis=-1)
    uvs = Bs @ cps
    v = uvs[..., 0]
    u = uvs[..., 1]
    u = tf.clip_by_value(u, 0., 1.)
    u = tf.cast(tf.round(u * (img.shape[2] - 1)), tf.int32).numpy()
    v = tf.clip_by_value(v, 0., 1.)
    v = tf.cast(tf.round(v * (img.shape[1] - 1)), tf.int32).numpy()
    # poly_img[u, v] = 1.
    d = 2
    for i in range(-d, d):
        for j in range(-d, d):
            ul = tf.clip_by_value(u + i, 0, img.shape[1] - 1).numpy()
            vl = tf.clip_by_value(v + j, 0, img.shape[2] - 1).numpy()
            # tf.scatter_nd(np.stack([ul, vl], axis=-1), batch_dim)
            poly_img[np.arange(bs)[:, np.newaxis], ul[np.arange(bs)], vl[np.arange(bs)]] = 1.
    # cpu = tf.cast(tf.round(cps[:, 0] * (img.shape[0] - 1)), tf.int32).numpy()
    # cpv = tf.cast(tf.round(cps[:, 1] * (img.shape[1] - 1)), tf.int32).numpy()
    # d = 4
    # for i in range(-d, d):
    #    for j in range(-d, d):
    #        ul = tf.clip_by_value(cpu + i, 0, img.shape[0] - 1)
    #        vl = tf.clip_by_value(cpv + j, 0, img.shape[1] - 1)
    #        poly_img[ul, vl] = 1.
    return poly_img
