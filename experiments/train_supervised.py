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

    # 3. Optimization

    optimizer = tf.keras.optimizers.Adam(args.eta)
    l2_reg = tf.keras.regularizers.l2(1e-5)

    # 4. Restore, Log & Save
    experiment_handler = ExperimentHandler(args.working_path, args.out_name, args.log_interval, model, optimizer)

    # 5. Run everything
    train_step, val_step = 0, 0
    best_accuracy = 0.0
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
        epoch_accuracy = tf.reduce_mean(tf.concat(acc, -1))

        # 5.1.6 Take statistics over epoch
        with tf.summary.record_if(True):
            tf.summary.scalar('epoch/model_loss', epoch_accuracy, step=epoch)

        experiment_handler.flush()
        continue

        ## 5.2. Validation Loop
        #experiment_handler.log_validation()
        #acc = []
        #for i, data in _ds('Validation', val_ds, val_size, epoch, args.batch_size):
        #    poly = model(data)
        #    model_loss =
        #    model_loss = fi_l_loss + fi_r_loss + ort_l_loss + ort_r_loss

        #    # 5.1.3 Calculate statistics
        #    fi_th = 0.05
        #    fi_th_2 = 0.1
        #    fi_l = tf.cast(fi_l_loss < fi_th, tf.float32)
        #    fi_r = tf.cast(fi_r_loss < fi_th, tf.float32)
        #    fi_l_2 = tf.cast(fi_l_loss < fi_th_2, tf.float32)
        #    fi_r_2 = tf.cast(fi_r_loss < fi_th_2, tf.float32)
        #    ort_l_pred = 1. - tf.abs(tf.cast(ort_l_pred > 0.5, tf.float32) - ort_l)
        #    ort_r_pred = 1. - tf.abs(tf.cast(ort_r_pred > 0.5, tf.float32) - ort_r)

        #    all = tf.concat([fi_l[:, tf.newaxis], fi_r[:, tf.newaxis], ort_l_pred, ort_r_pred], axis=-1)
        #    good_moves = tf.cast(tf.equal(tf.reduce_sum(all, axis=-1), 8.), tf.float32)
        #    acc.append(good_moves)
        #    all2 = tf.concat([fi_l_2[:, tf.newaxis], fi_r_2[:, tf.newaxis], ort_l_pred, ort_r_pred], axis=-1)
        #    good_moves2 = tf.cast(tf.equal(tf.reduce_sum(all2, axis=-1), 8.), tf.float32)
        #    acc2.append(good_moves2)

        #    # 5.1.4 Save logs for particular interval
        #    with tf.summary.record_if(val_step % args.log_interval == 0):
        #        tf.summary.scalar('losses/model_loss', tf.reduce_mean(model_loss), step=val_step)
        #        tf.summary.scalar('losses/fi_l_loss', tf.reduce_mean(fi_l_loss), step=val_step)
        #        tf.summary.scalar('losses/fi_r_loss', tf.reduce_mean(fi_r_loss), step=val_step)
        #        tf.summary.scalar('losses/ort_l_loss', tf.reduce_mean(ort_l_loss), step=val_step)
        #        tf.summary.scalar('losses/ort_r_loss', tf.reduce_mean(ort_r_loss), step=val_step)
        #        tf.summary.scalar('acc/fi_l_0_02_acc', tf.reduce_mean(fi_l), step=val_step)
        #        tf.summary.scalar('acc/fi_r_0_02_acc', tf.reduce_mean(fi_r), step=val_step)
        #        tf.summary.scalar('acc/fi_l_0_05_acc', tf.reduce_mean(fi_l_2), step=val_step)
        #        tf.summary.scalar('acc/fi_r_0_05_acc', tf.reduce_mean(fi_r_2), step=val_step)
        #        tf.summary.scalar('acc/ort_l_acc', tf.reduce_mean(ort_l_pred), step=val_step)
        #        tf.summary.scalar('acc/ort_r_acc', tf.reduce_mean(ort_r_pred), step=val_step)
        #        tf.summary.scalar('acc/good_moves_0_05', tf.reduce_mean(good_moves), step=val_step)
        #        tf.summary.scalar('acc/good_moves_0_1', tf.reduce_mean(good_moves2), step=val_step)

        #    # 5.2.4 Update meta variables
        #    val_step += 1
        #epoch_accuracy = tf.reduce_mean(tf.concat(acc, -1))
        #epoch_accuracy2 = tf.reduce_mean(tf.concat(acc2, -1))

        ## 5.1.6 Take statistics over epoch
        #with tf.summary.record_if(True):
        #    tf.summary.scalar('epoch/good_paths_0_05', epoch_accuracy, step=epoch)
        #    tf.summary.scalar('epoch/good_paths_0_1', epoch_accuracy2, step=epoch)

        ## 5.3 Save last and best
        #if epoch_accuracy > best_accuracy:
        #    experiment_handler.save_best()
        #    best_accuracy = epoch_accuracy
        ##experiment_handler.save_last()

        #experiment_handler.flush()


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
