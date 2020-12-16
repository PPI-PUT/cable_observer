from time import sleep, time

import matplotlib.pyplot as plt
import numpy as np
import cv2
from multiprocessing import Process
import tensorflow as tf

from models.observer_bezier import Cable, curve_loss, _plot

class J(tf.keras.Model):
    def __init__(self):
        super(J, self).__init__()
        self.J = tf.Variable(np.random.normal(size=(16, 12)).astype(np.float32), trainable=True, name="xy")

    def call(self, x):
        return self.J

model = J()
optimizer = tf.keras.optimizers.Adam(1e-2)

d_B = 0.01 * np.random.normal(size=(16, 1))
d_q = 0.01 * np.random.normal(size=(12, 1))

for epoch in range(300):
    with tf.GradientTape(persistent=True) as tape:
        J = model(None)
        error = d_B - J @ d_q
        model_loss = tf.reduce_sum(tf.abs(error))

    grads = tape.gradient(model_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    print(model_loss)

J = model(None)
error = d_B - J @ d_q
print(error)

