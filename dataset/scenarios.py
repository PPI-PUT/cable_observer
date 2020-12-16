import os
from glob import glob
from random import shuffle

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# tf.enable_eager_execution()
import yaml

from utils.utils import yaml2list


def manipulation_dataset(path):
    def read_img(cps, img_path):
        img = tf.io.read_file(img_path)
        img = tf.io.decode_png(img, channels=1)
        img = tf.image.convert_image_dtype(img, tf.float32)
        #img = tf.image.resize(img, (256, 256))
        img = tf.image.resize(img, (128, 128))
        background = img < 0.5
        cable = img > 0.5
        return cps, tf.cast(tf.concat([background, cable], axis=-1), tf.float32)

    def read_cps(path):
        data = np.loadtxt(path)
        return data

    imgs = [(read_cps(f), f.replace(".cps", ".png")) for f in glob(os.path.join(path, "*.cps"))]
    #imgs = np.array(imgs)

    def gen():
         for i in range(len(imgs)):
             yield imgs[i]

    N = int(1e3)
    #ds = tf.data.Dataset.from_tensor_slices(imgs) \
    #    .shuffle(buffer_size=N, reshuffle_each_iteration=True).map(read_img, num_parallel_calls=8)

    ds = tf.data.Dataset.from_generator(gen, (tf.float32, tf.string,)) \
         .shuffle(buffer_size=N, reshuffle_each_iteration=True).map(read_img, num_parallel_calls=8)
    return ds, N
