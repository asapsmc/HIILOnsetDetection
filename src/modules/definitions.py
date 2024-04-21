import itertools as it
import logging
import math
import os
import pickle
import shutil
import sys
import warnings
from collections.abc import MutableMapping
from pathlib import Path

import keras.backend as K
import keras.optimizers
import madmom
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from keras.callbacks import Callback, LambdaCallback
from keras.layers import Activation, Conv1D, SpatialDropout1D
from keras.utils import Sequence
from madmom.audio.signal import FramedSignalProcessor, SignalProcessor
from madmom.audio.spectrogram import (FilteredSpectrogramProcessor,
                                      LogarithmicSpectrogramProcessor,
                                      SpectrogramDifferenceProcessor)
from madmom.audio.stft import ShortTimeFourierTransformProcessor
from madmom.processors import ParallelProcessor, SequentialProcessor
from scipy.ndimage import maximum_filter1d

import modules.utils as utl

''' TF_CPP_MIN_LOG_LEVEL
0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printed
3 = INFO, WARNING, and ERROR messages are not printed
'''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

FPS = 100           # set the frame rate as FPS frames per second
MASK_VALUE = -1

warnings.filterwarnings('ignore')


if tf.test.gpu_device_name():
    print('TF: Default GPU Device:{}'.format(tf.test.gpu_device_name()))
else:
    print('TF: Using CPU')


class LRFinder_TCNv2:
    """
    Plots the change of the loss function of a Keras model when the learning rate is exponentially increasing.
    See for details:
    https://towardsdatascience.com/estimating-optimal-learning-rate-for-a-deep-neural-network-ce32f2556ce0
    """

    def __init__(self, model):
        self.model = model
        self.losses = []
        self.lrs = []
        self.best_loss = 1e9

    def on_batch_end(self, batch, logs):
        # Log the learning rate
        lr = K.get_value(self.model.optimizer.lr)
        self.lrs.append(lr)

        # Log the loss
        loss = logs['loss']
        self.losses.append(loss)

        # Check whether the loss got too large or NaN
        if batch > 5 and (math.isnan(loss) or loss > self.best_loss * 4):
            self.model.stop_training = True
            return

        if loss < self.best_loss:
            self.best_loss = loss

        # Increase the learning rate for the next batch
        lr *= self.lr_mult
        K.set_value(self.model.optimizer.lr, lr)

    def find(self, x_train, y_train, start_lr, end_lr, xy_train=None, batch_size=64, epochs=1, **kw_fit):
        # If x_train contains data for multiple inputs, use length of the first input.
        # Assumption: the first element in the list is single input; NOT a list of inputs.
        N = x_train[0].shape[0] if isinstance(x_train, list) else x_train.shape[0]

        # Compute number of batches and LR multiplier
        num_batches = epochs * N / batch_size
        self.lr_mult = (float(end_lr) / float(start_lr)) ** (float(1) / float(num_batches))
        # Save weights into a file
        initial_weights = self.model.get_weights()

        # Remember the original learning rate
        original_lr = K.get_value(self.model.optimizer.lr)

        # Set the initial learning rate
        K.set_value(self.model.optimizer.lr, start_lr)

        callback = LambdaCallback(on_batch_end=lambda batch, logs: self.on_batch_end(batch, logs))

        self.model.fit(xy_train, batch_size=batch_size, epochs=epochs,
                       steps_per_epoch=len(xy_train), callbacks=[callback], **kw_fit)
        # self.model.fit(x_train, y_train,
        #               batch_size=batch_size, epochs=epochs,
        #               callbacks=[callback],
        #               **kw_fit)

        # Restore the weights to the state before model fitting
        self.model.set_weights(initial_weights)

        # Restore the original learning rate
        K.set_value(self.model.optimizer.lr, original_lr)

    # def find_generator(self, generator, start_lr, end_lr, epochs=1, steps_per_epoch=None, **kw_fit):
    #    if steps_per_epoch is None:
    #        try:
    #            steps_per_epoch = len(generator)
    #        except (ValueError, NotImplementedError) as e:
    #            raise e('`steps_per_epoch=None` is only valid for a'
    #                    ' generator based on the '
    #                    '`keras.utils.Sequence`'
    #                    ' class. Please specify `steps_per_epoch` '
    #                    'or use the `keras.utils.Sequence` class.')
    #    self.lr_mult = (float(end_lr) / float(start_lr)) ** (float(1) / float(epochs * steps_per_epoch))
#
    #    # Save weights into a file
    #    initial_weights = self.model.get_weights()
#
    #    # Remember the original learning rate
    #    original_lr = K.get_value(self.model.optimizer.lr)
#
    #    # Set the initial learning rate
    #    K.set_value(self.model.optimizer.lr, start_lr)
#
    #    callback = LambdaCallback(on_batch_end=lambda batch,
    #                              logs: self.on_batch_end(batch, logs))
    #    callbacks = [LambdaCallback(on_batch_begin=lambda batch, logs: self.on_batch_begin(
    #        batch, logs)), LambdaCallback(on_batch_end=lambda batch, logs: self.on_batch_end(batch, logs))]
#
    #    self.model.fit_generator(generator=generator,
    #                             epochs=epochs,
    #                             steps_per_epoch=steps_per_epoch,
    #                             # callbacks=[callback],
    #                             callbacks=callbacks,
    #                             **kw_fit)
#
    #    # Restore the weights to the state before model fitting
    #    self.model.set_weights(initial_weights)
#
    #    # Restore the original learning rate
    #    K.set_value(self.model.optimizer.lr, original_lr)

    def plot_loss(self, n_skip_beginning=10, n_skip_end=5, x_scale='log', ax=None):
        """
        Plots the loss.
        Parameters:
            n_skip_beginning - number of batches to skip on the left.
            n_skip_end - number of batches to skip on the right.
        """
        if ax is not None:
            ax.set_ylabel("loss")
            ax.set_xlabel("learning rate (log scale)")
            ax.plot(self.lrs[n_skip_beginning:-n_skip_end], self.losses[n_skip_beginning:-n_skip_end])
            ax.set_xscale(x_scale)
        else:
            plt.ylabel("loss")
            plt.xlabel("learning rate (log scale)")
            plt.plot(self.lrs[n_skip_beginning:-n_skip_end], self.losses[n_skip_beginning:-n_skip_end])
            plt.xscale(x_scale)
            plt.show()
        return ax

    def plot_loss_change(self, sma=1, n_skip_beginning=10, n_skip_end=5, y_lim=(-0.01, 0.01), ax=None):
        """
        Plots rate of change of the loss function.
        Parameters:
            sma - number of batches for simple moving average to smooth out the curve.
            n_skip_beginning - number of batches to skip on the left.
            n_skip_end - number of batches to skip on the right.
            y_lim - limits for the y axis.
        """
        derivatives = self.get_derivatives(sma)[n_skip_beginning:-n_skip_end]
        lrs = self.lrs[n_skip_beginning:-n_skip_end]
        if ax is not None:
            ax.set_ylabel("rate of loss change")
            ax.set_xlabel("learning rate (log scale)")
            ax.plot(lrs, derivatives)
            ax.set_xscale('log')
            if y_lim is not None:
                ax.set_ylim(y_lim)
        else:
            plt.ylabel("rate of loss change")
            plt.xlabel("learning rate (log scale)")
            plt.plot(lrs, derivatives)
            plt.xscale('log')
            plt.ylim(y_lim)
            plt.show()
        return ax

    def get_derivatives(self, sma):
        assert sma >= 1
        derivatives = [0] * sma
        for i in range(sma, len(self.lrs)):
            derivatives.append((self.losses[i] - self.losses[i - sma]) / sma)
        return derivatives

    def get_best_lr(self, sma, n_skip_beginning=10, n_skip_end=5):
        derivatives = self.get_derivatives(sma)
        best_der_idx = np.argmin(derivatives[n_skip_beginning:-n_skip_end])
        return self.lrs[n_skip_beginning:-n_skip_end][best_der_idx]


class LRFinder:
    """
    Plots the change of the loss function of a Keras model when the learning rate is exponentially increasing.
    See for details:
    https://towardsdatascience.com/estimating-optimal-learning-rate-for-a-deep-neural-network-ce32f2556ce0
    """

    def __init__(self, model):
        self.model = model
        self.losses = []
        self.lrs = []
        self.best_loss = 1e9

    def on_batch_end(self, batch, logs):
        # Log the learning rate
        lr = K.get_value(self.model.optimizer.lr)
        self.lrs.append(lr)

        # Log the loss
        loss = logs['loss']
        self.losses.append(loss)

        # Check whether the loss got too large or NaN
        if batch > 5 and (math.isnan(loss) or loss > self.best_loss * 4):
            self.model.stop_training = True
            return

        if loss < self.best_loss:
            self.best_loss = loss

        # Increase the learning rate for the next batch
        lr *= self.lr_mult
        K.set_value(self.model.optimizer.lr, lr)

    def find(self, x_train, y_train, start_lr, end_lr, xy_train=None, batch_size=64, epochs=1, **kw_fit):
        # If x_train contains data for multiple inputs, use length of the first input.
        # Assumption: the first element in the list is single input; NOT a list of inputs.
        N = x_train[0].shape[0] if isinstance(x_train, list) else x_train.shape[0]

        # Compute number of batches and LR multiplier
        num_batches = epochs * N / batch_size
        self.lr_mult = (float(end_lr) / float(start_lr)) ** (float(1) / float(num_batches))
        # Save weights into a file
        initial_weights = self.model.get_weights()

        # Remember the original learning rate
        original_lr = K.get_value(self.model.optimizer.lr)

        # Set the initial learning rate
        K.set_value(self.model.optimizer.lr, start_lr)

        callback = LambdaCallback(on_batch_end=lambda batch, logs: self.on_batch_end(batch, logs))

        self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=[callback], **kw_fit)
        # self.model.fit(xy_train, batch_size=batch_size, epochs=epochs,
        #               steps_per_epoch=len(xy_train), callbacks=[callback], **kw_fit)

        # Restore the weights to the state before model fitting
        self.model.set_weights(initial_weights)

        # Restore the original learning rate
        K.set_value(self.model.optimizer.lr, original_lr)

    def plot_loss(self, n_skip_beginning=10, n_skip_end=5, x_scale='log', title=None, ax=None):
        """
        Plots the loss.
        Parameters:
            n_skip_beginning - number of batches to skip on the left.
            n_skip_end - number of batches to skip on the right.
        """
        if ax is not None:
            ax.set_ylabel("loss")
            ax.set_xlabel("learning rate (log scale)")
            ax.plot(self.lrs[n_skip_beginning:-n_skip_end], self.losses[n_skip_beginning:-n_skip_end])
            ax.set_xscale(x_scale)
            if title is not None:
                ax.set_title(title)
        else:
            plt.ylabel("loss")
            plt.xlabel("learning rate (log scale)")
            plt.plot(self.lrs[n_skip_beginning:-n_skip_end], self.losses[n_skip_beginning:-n_skip_end])
            plt.xscale(x_scale)
            if title is not None:
                plt.title(title)
            plt.show()
        return ax

    def plot_loss_change(self, sma=1, n_skip_beginning=10, n_skip_end=5, y_lim=(-0.01, 0.01), title=None, ax=None):
        """
        Plots rate of change of the loss function.
        Parameters:
            sma - number of batches for simple moving average to smooth out the curve.
            n_skip_beginning - number of batches to skip on the left.
            n_skip_end - number of batches to skip on the right.
            y_lim - limits for the y axis.
        """
        derivatives = self.get_derivatives(sma)[n_skip_beginning:-n_skip_end]
        lrs = self.lrs[n_skip_beginning:-n_skip_end]
        if ax is not None:
            ax.set_ylabel("rate of loss change")
            ax.set_xlabel("learning rate (log scale)")
            ax.plot(lrs, derivatives)
            ax.set_xscale('log')
            if title is not None:
                ax.set_title(title)
            if y_lim is not None:
                ax.set_ylim(y_lim)
        else:
            plt.ylabel("rate of loss change")
            plt.xlabel("learning rate (log scale)")
            plt.plot(lrs, derivatives)
            plt.xscale('log')
            plt.ylim(y_lim)
            if title is not None:
                plt.title(title)
            plt.show()
        return ax

    def get_derivatives(self, sma):
        assert sma >= 1
        derivatives = [0] * sma
        for i in range(sma, len(self.lrs)):
            derivatives.append((self.losses[i] - self.losses[i - sma]) / sma)
        return derivatives

    def get_best_lr(self, sma, n_skip_beginning=10, n_skip_end=5):
        derivatives = self.get_derivatives(sma)
        best_der_idx = np.argmin(derivatives[n_skip_beginning:-n_skip_end])
        return self.lrs[n_skip_beginning:-n_skip_end][best_der_idx]


class LRTensorBoard(tf.keras.callbacks.TensorBoard):
    def __init__(self, log_dir, **kwargs):  # add other arguments to __init__ if you need
        super().__init__(log_dir=log_dir, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs.update({'lr': K.eval(self.model.optimizer.lr)})
        super().on_epoch_end(epoch, logs)


class EarlyStoppingByLossVal(Callback):
    def __init__(self, monitor='loss', value=0.03, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        if current < self.value:
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True


def residual_block_v1(x, dilation_rate, activation, num_filters, kernel_size, padding, dropout_rate=0, name=''):
    original_x = x
    conv = Conv1D(num_filters, kernel_size=kernel_size,
                  dilation_rate=dilation_rate, padding='same',
                  name=name + '_%d_dilated_conv' % (dilation_rate))(x)
    x = Activation(activation, name=name + '_%d_activation' % (dilation_rate))(conv)
    x = SpatialDropout1D(dropout_rate, name=name + '_%d_spatial_dropout_%.2f' % (dilation_rate, dropout_rate))(x)
    x = Conv1D(num_filters, 1, padding='same', name=name + '_%d_conv_1x1' % (dilation_rate))(x)
    res_x = tf.keras.layers.add([original_x, x], name=name + '_%d_residual' % (dilation_rate))
    return res_x, x


class TCN_v1():
    def __init__(self, num_filters=8, kernel_size=5, dilations=[1, 2, 4, 8, 16, 32, 64, 128],
                 activation='elu', dropout_rate=0.15, name='tcn'):
        self.name = name
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.dilations = dilations
        self.kernel_size = kernel_size
        self.num_filters = num_filters

    def __call__(self, inputs):
        x = inputs
        for d in self.dilations:
            x, _ = residual_block_v1(x, d, self.activation, self.num_filters,
                                     self.kernel_size, self.dropout_rate, name=self.name)
        x = Activation(self.activation, name=self.name + '_activation')(x)
        return x


def residual_block_v2(x, i, activation, nb_filters, kernel_size, padding, dropout_rate=0, name=''):
    # name of the layer
    name = name + '_dilation_%d' % i
    # 1x1 conv. of input (so it can be added as residual)
    res_x = Conv1D(nb_filters, 1, padding='same', name=name + '_1x1_conv_residual')(x)
    # dilated convolution
    conv_1 = Conv1D(filters=nb_filters, kernel_size=kernel_size,
                    dilation_rate=i, padding=padding,
                    name=name + '_dilated_conv_1')(x)
    conv_2 = Conv1D(filters=nb_filters, kernel_size=kernel_size,
                    dilation_rate=i * 2, padding=padding,
                    name=name + '_dilated_conv_2')(x)
    concat = tf.keras.layers.concatenate([conv_1, conv_2], name=name + '_concat')
    x = Activation(activation, name=name + '_activation')(concat)
    x = SpatialDropout1D(dropout_rate, name=name + '_spatial_dropout_%f' % dropout_rate)(x)
    # 1x1 conv.
    x = Conv1D(nb_filters, 1, padding='same', name=name + '_1x1_conv')(x)
    return tf.keras.layers.concatenate([res_x, x], name=name + '_merge_residual'), x


class TCN_v2:
    def __init__(self,
                 nb_filters=20,
                 kernel_size=5,
                 dilations=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
                 activation='elu',
                 padding='same',
                 dropout_rate=0.15,  # was 0.15
                 name='tcn',
                 init_conv=0):
        self.name = name
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.dilations = dilations
        self.kernel_size = kernel_size
        self.nb_filters = nb_filters
        self.padding = padding
        self.init_conv = init_conv

        if padding != 'causal' and padding != 'same':
            raise ValueError("Only 'causal' or 'same' padding are compatible for this layer.")

    def __call__(self, inputs):
        x = inputs
        skip_connections = []
        for i, nb_filters in zip(self.dilations, self.nb_filters):
            x, skip_out = residual_block_v2(x, i, self.activation, nb_filters,
                                            self.kernel_size, self.padding, self.dropout_rate, name=self.name)
            skip_connections.append(skip_out)
        x = Activation(self.activation, name=self.name + '_activation')(x)
        return x, tf.keras.layers.add(skip_connections, name=self.name + '_merge_skip_connections')


# define pre-processor

# this now includes parameters for the start time, stop time
# and the rate to speed up or slow down for data augmentation
class PreProcessor(SequentialProcessor):
    def __init__(
            self, frame_sizes=[2048],
            num_bands=[12],
            fps=FPS, log=np.log, add=1e-6, diff=None, start=None, stop=None, daug_rate=1.):
        # resample to a fixed sample rate in order to get always the same
        # number of filter bins
        sig = SignalProcessor(num_channels=1, sample_rate=44100, start=start, stop=stop)
        # process multi-resolution spec & diff in parallel
        multi = ParallelProcessor([])
        for frame_size, num_bands in zip(frame_sizes, num_bands):
            # split audio signal in overlapping frames
            # Update the FPS to create the data augmentation
            frames = FramedSignalProcessor(frame_size=frame_size, fps=int(np.round(fps * daug_rate)))
            # compute STFT
            stft = ShortTimeFourierTransformProcessor()
            # filter the magnitudes
            filt = FilteredSpectrogramProcessor(num_bands=num_bands)
            # scale them logarithmically
            spec = LogarithmicSpectrogramProcessor(log=log, add=add)
            # stack positive differences
            if diff:
                diff = SpectrogramDifferenceProcessor(positive_diffs=True,
                                                      stack_diffs=np.hstack)
            # process each frame size with spec and diff sequentially
            multi.append(SequentialProcessor((frames, stft, filt, spec, diff)))
        # instantiate a SequentialProcessor
        super(PreProcessor, self).__init__((sig, multi, np.hstack))


class Dataset(object):
    def __init__(self, path, name=None, audio_suffix='.flac', onset_suffix='.onsets', daug_rate=1., start=None, stop=None):
        self.path = path
        if name is None:
            name = os.path.basename(path)
        self.name = name
        self.daug_rate = daug_rate
        self.start = start
        self.stop = stop
        # populate lists containing audio and annotation files
        audio_files = madmom.utils.search_files(self.path + '/audio', audio_suffix)
        annotation_files = madmom.utils.search_files(self.path + '/annotations/onsets/', onset_suffix)
        # match annotation to audio files
        self.files = []
        self.audio_files = []
        self.annotation_files = []

        for audio_file in audio_files:
            matches = madmom.utils.match_file(audio_file, annotation_files,
                                              suffix=audio_suffix, match_suffix=onset_suffix)
            if len(matches):
                self.audio_files.append(audio_file)
                self.files.append(os.path.basename(audio_file[:-len(audio_suffix)]))
                if len(matches) == 1:
                    self.annotation_files.append(matches[0])
                else:
                    self.annotation_files.append(None)

    def __len__(self):
        return len(self.files)

    def pre_process(self, pre_processor, num_threads=1):
        self.x = []
        for i, f in enumerate(self.audio_files):
            # sys.stderr.write('\rpre-processing file %d of %d' % (i + 1, len(self.audio_files)))
            # sys.stderr.flush()
            self.x.append(pre_processor(f))

    def load_splits(self, path=None, fold_suffix='.fold'):
        path = path if path is not None else self.path + '/splits'
        self.split_files = madmom.utils.search_files(path, fold_suffix, recursion_depth=1)
        # populate folds
        self.folds = []
        for i, split_file in enumerate(self.split_files):
            fold_idx = []
            with open(split_file) as f:
                for file in f:
                    file = file.strip()
                    # get matching file idx
                    try:
                        idx = self.files.index(file)
                        fold_idx.append(idx)
                    except ValueError:
                        # file could be not available, e.g. in Ballrom set a few duplicates were found
                        warnings.warn('no matching audio/annotation files: %s' % file)
                        continue
            # set indices for fold
            self.folds.append(np.array(fold_idx))

    def load_annotations(self, widen=None):
        self.annotations = []
        # self.tempo_annotations = []
        # self.downbeat_annotations = []
        for f in self.annotation_files:
            if f is None:
                beats = np.array([])
            else:
                beats = madmom.io.load_beats(f)
                if beats.ndim > 1:
                    beats = beats[:, 0]

            if self.stop is not None:
                beats = beats[beats <= self.stop]  # discard any after stop point

            if self.start is not None:
                beats = beats - self.start  # subtract the offset for the start point
                beats = beats[beats > 0]  # and remove any negative beats

            beats = beats * self.daug_rate  # update to reflect the data augmentation
            self.annotations.append(beats)
            # self.tempo_annotations.append(np.array([]))
            # self.downbeat_annotations.append(np.array([]))

    def add_dataset(self, dataset):
        self.files.extend(dataset.files)
        self.audio_files.extend(dataset.audio_files)
        self.annotation_files.extend(dataset.annotation_files)
        self.x.extend(dataset.x)
        self.annotations.extend(dataset.annotations)
        # self.tempo_annotations.extend(dataset.tempo_annotations)
        # self.downbeat_annotations.extend(dataset.downbeat_annotations)

    def dump(self, filename=None):
        if filename is None:
            filename = '%s/%s.pkl' % (self.path, self.name)
        pickle.dump(self, open(filename, 'wb'), protocol=2)


def build_masked_loss(loss_function, mask_value=MASK_VALUE):
    """Builds a loss function that masks based on targets

    Args:
        loss_function: The loss function to mask
        mask_value: The value to mask in the targets

    Returns:
        function: a loss function that acts like loss_function with masked inputs
    """

    def masked_loss_function(y_true, y_pred):
        mask = K.cast(K.not_equal(y_true, mask_value), K.floatx())
        return loss_function(y_true * mask, y_pred * mask)

    return masked_loss_function


def masked_accuracy(y_true, y_pred):
    total = K.sum(K.not_equal(y_true, MASK_VALUE))
    correct = K.sum(K.equal(y_true, K.round(y_pred)))
    return correct / total


# code based on: https://github.com/CyberZHG/keras-radam


class RAdam(keras.optimizers.Optimizer):
    """RAdam optimizer.

    # Arguments
        learning_rate: float >= 0. Learning rate.
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
        decay: float >= 0. Learning rate decay over each update.
        weight_decay: float >= 0. Weight decay for each param.
        amsgrad: boolean. Whether to apply the AMSGrad variant of this
            algorithm from the paper "On the Convergence of Adam and
            Beyond".
        total_steps: int >= 0. Total number of training steps. Enable warmup by setting a positive value.
        warmup_proportion: 0 < warmup_proportion < 1. The proportion of increasing steps.
        min_lr: float >= 0. Minimum learning rate after warmup.
    # References
        - [Adam - A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980v8)
        - [On the Convergence of Adam and Beyond](https://openreview.net/forum?id=ryQu7f-RZ)
        - [On The Variance Of The Adaptive Learning Rate And Beyond](https://arxiv.org/pdf/1908.03265v1.pdf)
    """

    def __init__(
        self,
        name,
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=None,
        decay=0.0,
        weight_decay=0.0,
        amsgrad=False,
        total_steps=0,
        warmup_proportion=0.1,
        min_lr=0.0,
        **kwargs
    ):
        learning_rate = kwargs.pop('lr', learning_rate)
        super(RAdam, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.learning_rate = K.variable(learning_rate, name='learning_rate')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.decay = K.variable(decay, name='decay')
            self.weight_decay = K.variable(weight_decay, name='weight_decay')
            self.total_steps = K.variable(total_steps, name='total_steps')
            self.warmup_proportion = K.variable(warmup_proportion, name='warmup_proportion')
            self.min_lr = K.variable(min_lr, name='min_lr')
        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.initial_decay = decay
        self.initial_weight_decay = weight_decay
        self.initial_total_steps = total_steps
        self.amsgrad = amsgrad
        self.name = name

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr

        if self.initial_decay > 0:
            lr = lr * (1.0 / (1.0 + self.decay * K.cast(self.iterations, K.dtype(self.decay))))

        t = K.cast(self.iterations, K.floatx()) + 1

        if self.initial_total_steps > 0:
            warmup_steps = self.total_steps * self.warmup_proportion
            decay_steps = K.maximum(self.total_steps - warmup_steps, 1)
            decay_rate = (self.min_lr - lr) / decay_steps
            lr = K.switch(
                t <= warmup_steps,
                lr * (t / warmup_steps),
                lr + decay_rate * K.minimum(t - warmup_steps, decay_steps),
            )

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p), name='m_' + str(i)) for (i, p) in enumerate(params)]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p), name='v_' + str(i)) for (i, p) in enumerate(params)]

        if self.amsgrad:
            vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p), name='vhat_' + str(i)) for (i, p) in enumerate(params)]
        else:
            vhats = [K.zeros(1, name='vhat_' + str(i)) for i in range(len(params))]

        self.weights = [self.iterations] + ms + vs + vhats

        beta_1_t = K.pow(self.beta_1, t)
        beta_2_t = K.pow(self.beta_2, t)

        sma_inf = 2.0 / (1.0 - self.beta_2) - 1.0
        sma_t = sma_inf - 2.0 * t * beta_2_t / (1.0 - beta_2_t)

        for p, g, m, v, vhat in zip(params, grads, ms, vs, vhats):
            m_t = (self.beta_1 * m) + (1.0 - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1.0 - self.beta_2) * K.square(g)

            m_corr_t = m_t / (1.0 - beta_1_t)
            if self.amsgrad:
                vhat_t = K.maximum(vhat, v_t)
                v_corr_t = K.sqrt(vhat_t / (1.0 - beta_2_t))
                self.updates.append(K.update(vhat, vhat_t))
            else:
                v_corr_t = K.sqrt(v_t / (1.0 - beta_2_t))

            r_t = K.sqrt((sma_t - 4.0) / (sma_inf - 4.0) * (sma_t - 2.0) / (sma_inf - 2.0) * sma_inf / sma_t)

            p_t = K.switch(sma_t >= 5, r_t * m_corr_t / (v_corr_t + self.epsilon), m_corr_t)

            if self.initial_weight_decay > 0:
                p_t += self.weight_decay * p

            p_t = p - lr * p_t

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    @property
    def lr(self):
        return self.learning_rate

    @lr.setter
    def lr(self, learning_rate):
        self.learning_rate = learning_rate

    def get_name(self):
        return self.name

    def get_config(self):
        config = {
            'learning_rate': float(K.get_value(self.learning_rate)),
            'beta_1': float(K.get_value(self.beta_1)),
            'beta_2': float(K.get_value(self.beta_2)),
            'decay': float(K.get_value(self.decay)),
            'weight_decay': float(K.get_value(self.weight_decay)),
            'epsilon': self.epsilon,
            'amsgrad': self.amsgrad,
            'total_steps': float(K.get_value(self.total_steps)),
            'warmup_proportion': float(K.get_value(self.warmup_proportion)),
            'min_lr': float(K.get_value(self.min_lr)),
        }
        base_config = super(RAdam, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

# code based on: https://github.com/CyberZHG/keras-lookahead


class Lookahead(keras.optimizers.Optimizer):
    """The lookahead mechanism for optimizers.

    Default parameters follow those provided in the original paper.
    # Arguments
        optimizer: An existed optimizer.
        sync_period: int > 0. The synchronization period.
        slow_step: float, 0 < alpha < 1. The step size of slow weights.
    # References
        - [Lookahead Optimizer: k steps forward, 1 step back]
          (https://arxiv.org/pdf/1907.08610v1.pdf)
    """

    def __init__(self, name, optimizer, sync_period=5, slow_step=0.5, **kwargs):
        super(Lookahead, self).__init__(**kwargs)
        self.name = name
        self.optimizer = keras.optimizers.get(optimizer)
        with K.name_scope(self.__class__.__name__):
            self.sync_period = K.variable(sync_period, dtype='int64', name='sync_period')
            self.slow_step = K.variable(slow_step, name='slow_step')

    @property
    def lr(self):
        return self.optimizer.lr

    @lr.setter
    def lr(self, lr):
        self.optimizer.lr = lr

    @property
    def learning_rate(self):
        return self.optimizer.learning_rate

    @learning_rate.setter
    def learning_rate(self, learning_rate):
        self.optimizer.learning_rate = learning_rate

    @property
    def iterations(self):
        return self.optimizer.iterations

    def get_updates(self, loss, params):
        sync_cond = K.equal((self.iterations + 1) // self.sync_period * self.sync_period, (self.iterations + 1))
        slow_params = [K.variable(K.get_value(p), name='sp_{}'.format(i)) for i, p in enumerate(params)]
        self.updates = self.optimizer.get_updates(loss, params)
        slow_updates = []
        for p, sp in zip(params, slow_params):
            sp_t = sp + self.slow_step * (p - sp)
            slow_updates.append(
                K.update(
                    sp,
                    K.switch(
                        sync_cond,
                        sp_t,
                        sp,
                    ),
                )
            )
            slow_updates.append(
                K.update_add(
                    p,
                    K.switch(
                        sync_cond,
                        sp_t - p,
                        K.zeros_like(p),
                    ),
                )
            )
        self.updates += slow_updates
        self.weights = self.optimizer.weights + slow_params
        return self.updates

    def get_config(self):
        config = {
            'optimizer': keras.optimizers.serialize(self.optimizer),
            'sync_period': int(K.get_value(self.sync_period)),
            'slow_step': float(K.get_value(self.slow_step)),
        }
        base_config = super(Lookahead, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def get_name(self):
        return self.name

    @classmethod
    def from_config(cls, config):
        optimizer = keras.optimizers.deserialize(config.pop('optimizer'))
        return cls(optimizer, **config)


class Fold(object):

    def __init__(self, folds, fold):
        self.folds = folds
        self.fold = fold

    @property
    def test(self):
        # fold N for testing
        return np.unique(self.folds[self.fold])

    @property
    def val(self):
        # fold N+1 for validation
        return np.unique(self.folds[(self.fold + 1) % len(self.folds)])

    @property
    def train(self):
        # all remaining folds for training
        train = np.hstack(self.folds)
        train = np.setdiff1d(train, self.val)
        train = np.setdiff1d(train, self.test)
        return train


def get_fold(path=None, suffix='.fold', old_format=False, filename=None):
    split_files = madmom.utils.search_files(path, suffix='*.fold',
                                            recursion_depth=1)

    ct = -1
    for split_file in split_files:
        ct = ct + 1
        with open(split_file) as f:
            if filename in f.read():
                foldnum = ct

    return foldnum


def cnn_pad(data, pad_frames):
    """Pad the data by repeating the first and last frame N times."""
    pad_start = np.repeat(data[:1], pad_frames, axis=0)
    pad_stop = np.repeat(data[-1:], pad_frames, axis=0)
    return np.concatenate((pad_start, data, pad_stop))


class DataSequence_TCNv2XPTO(Sequence):

    def __init__(self, x, beats, fps=FPS, pad_frames=None):
        self.x = x
        self.pad_frames = pad_frames
        self.y = [madmom.utils.quantize_events(o, fps=fps, length=len(d))
                  for o, d in zip(beats, self.x)]
        # self.tempo = np.ones(300, dtype=np.float32) * MASK_VALUE
        # self.downbeats = [np.ones(len(d), dtype=np.float32) * MASK_VALUE
        #                  for o, d in zip(beats, self.x)]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = np.array(cnn_pad(self.x[idx], self.pad_frames))[np.newaxis, ..., np.newaxis]
        y = self.y[idx][np.newaxis, ..., np.newaxis]
        # y['tempo'] = self.tempo[idx][np.newaxis, ...]
        # y['downbeats'] = self.downbeats[idx][np.newaxis, ..., np.newaxis]
        return x, y

    def widen_beat_targets(self, size=3, value=0.5):
        for y in self.y:
            if np.allclose(y, MASK_VALUE):
                continue
            np.maximum(y, maximum_filter1d(y, size=size) * value, out=y)


class DataSequence_TCNv2(Sequence):

    def __init__(self, x, beats, fps=FPS, pad_frames=None):
        self.x = x
        self.pad_frames = pad_frames

        self.beats = [madmom.utils.quantize_events(o, fps=fps, length=len(d))
                      for o, d in zip(beats, self.x)]
        # self.y = [madmom.utils.quantize_events(o, fps=fps, length=len(d))
        #          for o, d in zip(beats, self.x)]
        self.tempo = np.ones(300, dtype=np.float32) * MASK_VALUE
        self.downbeats = [np.ones(len(d), dtype=np.float32) * MASK_VALUE
                          for o, d in zip(beats, self.x)]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = np.array(cnn_pad(self.x[idx], self.pad_frames))[np.newaxis, ..., np.newaxis]
        y = {}
        y['beats'] = self.beats[idx][np.newaxis, ..., np.newaxis]
        y['tempo'] = self.tempo[idx][np.newaxis, ...]
        y['downbeats'] = self.downbeats[idx][np.newaxis, ..., np.newaxis]
        # y = self.y[idx][np.newaxis, ..., np.newaxis]
        return x, y

    def widen_beat_targets(self, size=3, value=0.5):
        for y in self.beats:
            # skip masked beat targets
            if np.allclose(y, MASK_VALUE):
                continue
            np.maximum(y, maximum_filter1d(y, size=size) * value, out=y)


class DataSequence_TCNv1(Sequence):

    def __init__(self, x, y, fps=FPS, pad_frames=None):
        self.x = x
        self.y = [madmom.utils.quantize_events(o, fps=fps, length=len(d))
                  for o, d in zip(y, self.x)]
        self.pad_frames = pad_frames

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = np.array(cnn_pad(self.x[idx], self.pad_frames))[np.newaxis, ..., np.newaxis]
        y = self.y[idx][np.newaxis, ..., np.newaxis]
        return x, y

    def widen_targets(self, size=3, value=0.5):
        for y in self.y:
            np.maximum(y, maximum_filter1d(y, size=size) * value, out=y)


def get_widened_data_sequence(arch, train_db, val_db=None):
    if arch == utl.TCNV2:
        train = DataSequence_TCNv2(train_db.x, train_db.annotations, pad_frames=2)
        # this is a bit more widening than in ISMIR 2020,
        # to try to account for possibly poor localisation of the annotations
        # that are used for the fine-tuning
        train.widen_beat_targets(size=3, value=0.5)
        train.widen_beat_targets(size=3, value=0.25)
        train.widen_beat_targets(size=5, value=0.125)
        if val_db is not None:
            # BUG HERE
            val = DataSequence_TCNv2(val_db.x, val_db.annotations, pad_frames=2)
            # val.widen_beat_targets()
            val.widen_beat_targets(size=3, value=0.5)
            val.widen_beat_targets(size=3, value=0.25)
            val.widen_beat_targets(size=5, value=0.125)
        else:
            val = None
    elif arch == utl.TCNV1:
        train = DataSequence_TCNv1(train_db.x, train_db.annotations, pad_frames=2)
        train.widen_targets()
        if val_db is not None:
            val = DataSequence_TCNv1(val_db.x, val_db.annotations, pad_frames=2)
            val.widen_targets()
        else:
            val = None
    else:
        print("ERROR: Define this function for other models")
    return train, val


def set_trainable_layers(model, fz_layers_idx):
    """
    Sets the trainable state of the layers in a model based on specified indices.

    Parameters:
    - model (tf.keras.Model): The model whose layers will be configured.
    - fz_layers_idx (tuple): A tuple where the first element is the start index and the second element is the end index
                              for the range of layers to be frozen. If the tuple contains `None`, no layers will be frozen.

    Returns:
    - tf.keras.Model: The model with updated layer trainable states.
    """

    # Determine the range to freeze or unfreeze based on fz_layers_idx
    start_idx, end_idx = fz_layers_idx if fz_layers_idx[-1] is not None else (None, None)

    # Set trainable state based on the provided indices
    for i, layer in enumerate(model.layers):
        layer.trainable = True if start_idx is None else not (start_idx <= i < end_idx)
        logging.debug(f"{i}{layer.name} - {layer.trainable}")

    return model

############
# New 2024


# Define architecture-specific configurations
# NOTE: At this time, the v2.11+ optimizer `tf.keras.optimizers.Adam` runs slowly on M1/M2 Macs, please use the legacy Keras optimizer instead, located at `tf.keras.optimizers.legacy.Adam`.
architecture_configs = {
    utl.TCNV1: {
        'optimizer': lambda lr, clipnorm: tf.keras.optimizers.legacy.Adam(learning_rate=lr, clipnorm=clipnorm),
        'learning_rate': 0.002,
        'num_epochs': 50,
        'callbacks': lambda: [
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='loss', factor=0.5, patience=5,
                mode='auto', min_delta=1e-3, cooldown=0, min_lr=1e-8
            )
        ],
        'compile_args': {
            'loss': 'binary_crossentropy',
            'metrics': ['binary_accuracy']
        }
    },
    utl.TCNV2: {
        'optimizer': lambda lr, clipnorm: tfa.optimizers.Lookahead(
            tfa.optimizers.RectifiedAdam(learning_rate=lr, clipnorm=clipnorm),
            sync_period=6, slow_step_size=0.5),
        'learning_rate': 0.005,
        'num_epochs': 150,
        'callbacks': lambda: [
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='loss', factor=0.5, patience=1,
                mode='auto', min_delta=1e-3, cooldown=0, min_lr=1e-8
            )
        ],
        'compile_args': {
            'loss': [build_masked_loss(K.binary_crossentropy),
                     build_masked_loss(K.binary_crossentropy),
                     build_masked_loss(K.binary_crossentropy)],
            'metrics': ['binary_accuracy']
        }
    }
}


############
# Dataset-related
############


def setup_new_dataset(dataset, subdataset=None, daug_rate=1, start=None, stop=None, suffix=''):
    db = create_dataset(dataset, subdataset, daug_rate=daug_rate, start=start, stop=stop)
    paths = utl.set_paths_dataset(dataset, subdataset, suffix=suffix)
    return True


def create_dataset(dataset, subdataset=None, daug_rate=1, start=None, stop=None):
    """
    Create a dataset with specified preprocessing.

    Args:
        dataset (str): The name of the dataset.
        subdataset (str, optional): The name of the subdataset. Defaults to ''.
        daug_rate (int, optional): The daugmentation rate. Defaults to 1.
        start (int, optional): The start index. Defaults to None.
        stop (int, optional): The stop index. Defaults to None.

    Returns:
        Dataset: The created and preprocessed dataset.
    """
    audio_suffix = utl.AUDIO_SUFIX.get(dataset, '.flac')
    ds_path = os.path.join(utl.DATA_PATH, dataset)
    pkl_path = os.path.join(utl.PKL_PATH, dataset)
    if subdataset:
        ds_path = os.path.join(ds_path, subdataset)
        pkl_path = os.path.join(pkl_path, subdataset)
    pkl_path += '.pkl'
    logging.info("Creating dataset: %s, subdataset: %s", dataset, subdataset)

    db = Dataset(ds_path, audio_suffix=audio_suffix,
                 onset_suffix='.onsets', daug_rate=daug_rate, start=start, stop=stop)
    db.load_annotations()

    pp = PreProcessor(daug_rate=daug_rate, start=start, stop=stop)
    db.pre_process(pp)
    db.dump(pkl_path)
    logging.info("Dataset creation completed")
    return db


def create_retrain_dataset(dataset, subdataset=None, target_file=None, daug_rate=1.0, start=None, stop=None):
    audio_suffix = utl.AUDIO_SUFIX.get(dataset, '.flac')
    ds_path = os.path.join(utl.DATA_PATH, dataset)
    pkl_path = os.path.join(utl.PKL_PATH, dataset)
    if subdataset:
        ds_path = os.path.join(ds_path, subdataset)
        pkl_path = os.path.join(pkl_path, subdataset)
    pkl_path += '.pkl'
    logging.info("Creating retrain dataset: %s, subdataset: %s", dataset, subdataset)
    db = Dataset(ds_path, audio_suffix=target_file + audio_suffix,
                 onset_suffix=target_file + '.onsets', daug_rate=1.0, start=start, stop=stop)
    db.load_annotations()
    pp = PreProcessor(daug_rate=daug_rate, start=start, stop=stop)
    db.pre_process(pp)
    return db


def load_dataset(dataset, subdataset):
    pkl_path = '%s/%s/%s.pkl' % (utl.PKL_PATH, dataset, subdataset)
    with open(pkl_path, 'rb') as pkl:
        db = pickle.load(pkl)
    return db


# Old
def get_anns_from_idx(idx, dataset, suffix=''):
    db = pickle.load(open('%s/%s.pkl' % (utl.PKL_PATH, dataset + suffix), 'rb'))
    full_anns = db.annotations[idx]
    del db
    return full_anns


def get_dataset_from_paths(path):
    return str(path["analysis"].parents[1]).split('/')[-1]


def get_filename_from_idx(idx, dataset, suffix=''):

    db = pickle.load(open('%s/%s.pkl' % (utl.PKL_PATH, dataset + suffix), 'rb'))
    filename = db.files[idx]
    del db
    return filename
