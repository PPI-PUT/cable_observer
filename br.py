import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
from time import time
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from skimage.morphology import skeletonize, medial_axis

from models.observer_spline import Cable, curve_loss, _plot


class CameraReader:
    """Reads images from camera and optimize Bezier curve to match the shape of masked cable"""
    def __init__(self):
        self.cap = cv2.VideoCapture(2)
        #self.cap = cv2.VideoCapture(6)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 0)
        #self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        #self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        self.img = None
        self.data = None
        self.model = Cable()
        self.data = plt.imread("experiments/a1.png")
        #self.data = tf.tile(plt.imread("b.jpg").astype(np.float32)[..., np.newaxis], (1, 1, 3)) / 255.
        self.coarse_optimizer = tf.keras.optimizers.Adam(1e-2)
        self.fine_optimizer = tf.keras.optimizers.Adam(3e-3)
        # wait a moment to get the images ready for capturing
        for i in range(10):
            self.read_img(plot=True, pause=True)

    def read_img(self, plot=False, pause=False):
        # read image
        ret, frame = self.cap.read()
        # segment image
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        o = tf.cast(tf.ones_like(hsv), tf.float32)[..., 0]
        z = tf.zeros_like(o)
        img = tf.where(tf.logical_and(tf.logical_or(hsv[..., 0] < 15., hsv[..., 0] > 170.), hsv[..., 1] > 50), o, z)
        s = img
        # erode to get less points
        #img = \
        #tf.nn.erosion2d(img[tf.newaxis, ..., tf.newaxis], tf.zeros((3, 3, 1)), strides=[1, 1, 1, 1], padding="SAME",
                        #data_format="NHWC", dilations=[1, 1, 1, 1])[0, ..., 0]
        img = skeletonize(img.numpy())
        # img = (img - np.min(img)) / (np.max(img) - np.min(img))
        if plot:
            plt.subplot(1, 3, 1)
            plt.imshow(hsv)
            plt.subplot(1, 3, 2)
            plt.imshow(s)
            plt.subplot(1, 3, 3)
            plt.imshow(img)
        if pause:
            plt.pause(0.1)
            # pass
        else:
            pass
            # plt.show()
        # prepare for publishing -- can be optimized
        background = img < 0.5
        cable = img > 0.5
        data = tf.cast(tf.stack([background, cable], axis=-1), tf.float32)
        self.img = img
        self.data = data

    def track_img(self, iterations=5, coarse=False):
        """Optimize actual Bezier curve to match new image readings for a number of iterations"""
        for epoch in range(iterations):
            t0 = time()
            with tf.GradientTape(persistent=True) as tape:
                t1 = time()
                curve = self.model(None)
                t2 = time()
                model_loss, u, v, shape_loss, outside_loss, reverse_loss, length = curve_loss(curve, self.data)
            t3 = time()
            grads = tape.gradient(model_loss, self.model.trainable_variables)
            t4 = time()
            if coarse:
                self.coarse_optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
            else:
                self.fine_optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
            t5 = time()
            # t6 = time()
            # print("MODEL:", t2 - t1)
            # print("LOSS:", t3 - t2)
            # print("GRAD:", t4 - t3)
            # print("OPT:", t5 - t4)
            #print(model_loss)
            print("ADJUST:", t5 - t0)
        output_img = _plot(u, v, self.data, curve.numpy())

        #plt.subplot(1, 2, 1)
        #plt.imshow(self.data[..., 1])
        #plt.subplot(1, 2, 2)
        #plt.imshow(output_img)
        # plt.show()
        #plt.savefig(str(epoch).zfill(3) + ".png")
        # cv2.imshow("a", self.data[..., 1:])

        cv2.imshow("a", self.data.numpy()[..., 1])
        cv2.imshow("b", output_img)
        cv2.waitKey(1)


reader = CameraReader()
reader.track_img(300)
#reader.track_img(100, coarse=True)
while True:
    t1 = time()
    reader.read_img()
    t2 = time()
    reader.track_img(2)
    #reader.track_img(2, coarse=True)
    t3 = time()
    print("TIMES:")
    print("READ:", t2 - t1)
    print("PLOT + ADJUST:", t3 - t2)

reader.cap.release()
