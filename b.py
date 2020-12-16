from glob import glob
from time import sleep, time

import matplotlib.pyplot as plt
import numpy as np
import cv2
from multiprocessing import Process
import tensorflow as tf

from models.observer_bezier import Cable, curve_loss, _plot


#for i, p in enumerate(glob("./dd/*.png")):
for i, p in enumerate(glob("./data/train/red_cable_easy/*.png")):
    model = Cable()
    optimizer = tf.keras.optimizers.Adam(1e-2)
    data = plt.imread(p)
    for epoch in range(200):
        with tf.GradientTape(persistent=True) as tape:
            curve = model(None)
            model_loss, u, v, shape_loss, outside_loss, reverse_loss, length = curve_loss(curve, data)
        grads = tape.gradient(model_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        output_img = _plot(u, v, data, curve)
        cv2.imshow("a", data[..., 1])
        # cv2.imshow("a", self.data[..., 1:])
        cv2.imshow("b", output_img)
        cv2.waitKey(1)
    if model_loss < 0.01:
        plt.imsave("res/" + str(i).zfill(3) + ".png", output_img)
    print(i, model_loss)

