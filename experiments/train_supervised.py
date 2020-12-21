import inspect
import os
import sys
from time import time

import numpy as np
from matplotlib import pyplot as plt


#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

# add parent (root) to pythonpath
from dataset import scenarios
from argparse import ArgumentParser

import tensorflow as tf
from tqdm import tqdm

from utils.execution import ExperimentHandler, LoadFromFile
from models.bezier_supervised import _plot, CableNetwork

tf.random.set_seed(444)

_tqdm = lambda t, s, i: tqdm(
    ncols=80,
    total=s,
    bar_format='%s epoch %d | {l_bar}{bar} | Remaining: {remaining}' % (t, i))


def _ds(title, ds, ds_size, i, batch_size):
    with _tqdm(title, ds_size, i) as pbar:
        for i, data in enumerate(ds):
            yield (i, data)
            pbar.update(batch_size)

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

def main(args):
    # 1. Get datasets
    train_ds, train_size = scenarios.manipulation_dataset(args.scenario_path)
    val_ds, val_size = scenarios.manipulation_dataset(args.scenario_path.replace("train", "val"))

    val_ds = val_ds \
        .batch(args.batch_size) \
        .prefetch(args.batch_size)

    # 2. Define model
    model = CableNetwork()
    model.encoder.load_weights("./trained_AE/encoder_best_1034")

    # 3. Optimization

    optimizer = tf.keras.optimizers.Adam(args.eta)
    l2_reg = tf.keras.regularizers.l2(1e-5)

    # 4. Restore, Log & Save
    experiment_handler = ExperimentHandler(args.working_path, args.out_name, args.log_interval, model, optimizer)

    # 5. Run everything
    train_step, val_step = 0, 0
    best_loss = 1e10
    for epoch in range(args.num_epochs):
        # workaround for tf problems with shuffling
        dataset_epoch = train_ds.shuffle(train_size)
        dataset_epoch = dataset_epoch.batch(args.batch_size).prefetch(args.batch_size)

        # 5.1. Training Loop
        experiment_handler.log_training()
        acc = []
        for i, data in _ds('Train', dataset_epoch, train_size, epoch, args.batch_size):
            # 5.1.1. Make inference of the model, calculate losses and record gradients
            with tf.GradientTape(persistent=True) as tape:
                gt_cps, img = data
                predicted_cps = model(img)
                print(predicted_cps[0])
                model_loss = tf.keras.losses.mean_squared_error(gt_cps, predicted_cps)#tf.reduce_sum(tf.abs(predicted_cps - gt_cps), axis=(-1, -2))
                model_loss = tf.reduce_sum(model_loss, axis=-1)
                output_img = _plot(predicted_cps, img)
            # 5.1.2 Take gradients (if necessary apply regularization like clipping),
            grads = tape.gradient(model_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # 5.1.3 Calculate statistics
            acc.append(model_loss)

            # 5.1.4 Save logs for particular interval
            with tf.summary.record_if(train_step % args.log_interval == 0):
                tf.summary.scalar('losses/model_loss', tf.reduce_mean(model_loss), step=train_step)
                tf.summary.image('img/input', img[..., 1:], step=train_step)
                tf.summary.image('img/output', output_img[..., tf.newaxis], step=train_step)

            # 5.1.5 Update meta variables
            train_step += 1
        epoch_loss = tf.reduce_mean(tf.concat(acc, -1))

        # 5.1.6 Take statistics over epoch
        with tf.summary.record_if(True):
            tf.summary.scalar('epoch/model_loss', epoch_loss, step=epoch)

        #experiment_handler.flush()
        #continue

        # 5.2. Validation Loop
        experiment_handler.log_validation()
        acc = []
        for i, data in _ds('Validation', val_ds, val_size, epoch, args.batch_size):
            gt_cps, img = data
            predicted_cps = model(img)
            print(predicted_cps[0])
            model_loss = tf.keras.losses.mean_squared_error(gt_cps,
                                                            predicted_cps)  # tf.reduce_sum(tf.abs(predicted_cps - gt_cps), axis=(-1, -2))
            model_loss = tf.reduce_sum(model_loss, axis=-1)
            output_img = _plot(predicted_cps, img)

            acc.append(model_loss)

            # 5.1.4 Save logs for particular interval
            with tf.summary.record_if(val_step % args.log_interval == 0):
                tf.summary.scalar('losses/model_loss', tf.reduce_mean(model_loss), step=val_step)
                tf.summary.image('img/input', img[..., 1:], step=val_step)
                tf.summary.image('img/output', output_img[..., tf.newaxis], step=val_step)

            # 5.2.4 Update meta variables
            val_step += 1
        epoch_loss = tf.reduce_mean(tf.concat(acc, -1))

        # 5.1.6 Take statistics over epoch
        with tf.summary.record_if(True):
            tf.summary.scalar('epoch/model_loss', epoch_loss, step=epoch)

        # 5.3 Save last and best
        if epoch_loss < best_loss:
            experiment_handler.save_best()
            best_loss = epoch_loss
        ##experiment_handler.save_last()

        experiment_handler.flush()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config-file', action=LoadFromFile, type=open)
    parser.add_argument('--scenario-path', type=str)
    parser.add_argument('--working-path', type=str, default='./working_dir')
    parser.add_argument('--num-epochs', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--log-interval', type=int, default=5)
    parser.add_argument('--out-name', type=str)
    parser.add_argument('--eta', type=float, default=5e-4)
    args, _ = parser.parse_known_args()
    main(args)
