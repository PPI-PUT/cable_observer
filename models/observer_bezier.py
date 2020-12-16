from scipy.special import comb
import tensorflow as tf
import numpy as np

N = 7
#N = 15


class Cable(tf.keras.Model):
    """Bezier curve object with optimizable control points"""
    def __init__(self, xy=None):
        super(Cable, self).__init__()
        # initialize with horizontally oriented straight cable
        if xy is None:
            s = np.linspace(0.1, 0.9, N + 1)
            xy = np.stack([0.5 * np.ones(N + 1), s], axis=-1)
        self.xy = tf.Variable(xy.astype(np.float32), trainable=True, name="xy")
        # self.xy = tf.Variable(np.random.uniform(size=(N+1, 2)).astype(np.float32), trainable=True, name="xy")

    def call(self, x):
        return self.xy


def curve_loss(control_points, data):
    """Calcualtes the loss of the Bezier curve made with control_points in order to match the shape of the masked
    object from data """
    t = np.linspace(0., 1., 128)

    def B(n, i, t):
        return comb(n, i) * t ** i * (1 - t) ** (n - i)

    # calculate bezier curve
    Bs = np.stack([B(N, k, t) for k in range(0, N + 1)], axis=-1)
    uvs = Bs @ control_points
    us = uvs[..., 0]
    vs = uvs[..., 1]

    # calculate bezier curve first derivative
    dBs = np.stack([B(N - 1, k, t) for k in range(0, N)], axis=-1)
    duvs = N * dBs @ (control_points[1:] - control_points[:-1])
    dus = duvs[..., 0]
    dvs = duvs[..., 1]

    # calculate bezier curve second derivative
    ddBs = np.stack([B(N - 2, k, t) for k in range(0, N - 1)], axis=-1)
    dduvs = N * (N - 1) * ddBs @ (control_points[2:] - 2 * control_points[1:-1] + control_points[:-2])

    ddus = dduvs[..., 0]
    ddvs = dduvs[..., 1]

    # calculate bezier curve curvature
    curvature = tf.abs(ddvs * dus - ddus * dvs) / (dus ** 2 + dvs ** 2) ** (3. / 2)

    # get the mask as the point cloud
    idx = np.where(data[..., 1] == 1.)
    idx_c = np.stack(idx, axis=-1)
    idx_c = idx_c / np.array([data.shape[0], data.shape[1]])[np.newaxis]

    # calculate mean minimum distance between point cloud and bezier curve and reverse
    dist = tf.abs(uvs[:, tf.newaxis] - idx_c[tf.newaxis])
    dist = tf.reduce_sum(dist, axis=-1)
    shape_loss = tf.reduce_min(dist, axis=-2)
    shape_loss = tf.reduce_mean(shape_loss, axis=-1)
    reverse_loss = tf.reduce_min(dist, axis=-1)
    reverse_loss = tf.reduce_mean(reverse_loss, axis=-1)
    # t4 = time()

    # calculate curve length
    diff_uv = uvs[1:] - uvs[:-1]
    diff_uv = tf.abs(diff_uv)
    length = tf.reduce_sum(diff_uv)

    # calculate if curve is outside the drawing area
    outside_loss = tf.nn.relu(uvs - 1.) + tf.nn.relu(-uvs)
    outside_loss = tf.reduce_sum(outside_loss)

    # calculate the derivative of the curvature
    dcurv = curvature[1:] - curvature[:-1]
    curvature_loss = tf.abs(dcurv)
    curvature_loss = tf.reduce_mean(curvature_loss)

    model_loss = shape_loss + 1 * outside_loss + reverse_loss + 1e-3 * curvature_loss
    return model_loss, us, vs, shape_loss, outside_loss, reverse_loss, length


def _plot(u, v, data, control_points):
    """Plots uv curve on the image plane"""
    poly_img = np.zeros_like(data[..., 0])
    u = tf.clip_by_value(u, 0., 1.)
    u = tf.cast(tf.round(u * (data.shape[0] - 1)), tf.int32).numpy()
    v = tf.clip_by_value(v, 0., 1.)
    v = tf.cast(tf.round(v * (data.shape[1] - 1)), tf.int32).numpy()
    poly_img[u, v] = 1.
    cpu = tf.cast(tf.round(control_points[:, 0] * (data.shape[0] - 1)), tf.int32).numpy()
    cpv = tf.cast(tf.round(control_points[:, 1] * (data.shape[1] - 1)), tf.int32).numpy()
    d = 2
    for i in range(-d, d):
        for j in range(-d, d):
            ul = tf.clip_by_value(cpu + i, 0, data.shape[0] - 1)
            vl = tf.clip_by_value(cpv + j, 0, data.shape[1] - 1)
            poly_img[ul, vl] = 1.
    return poly_img
