import logging
import os
import pickle
import warnings

import keras.backend as K
import madmom
import modules.utils as utl
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from keras.layers import Activation, Conv1D, SpatialDropout1D
from keras.utils import Sequence
from madmom.audio.signal import FramedSignalProcessor, SignalProcessor
from madmom.audio.spectrogram import (
    FilteredSpectrogramProcessor,
    LogarithmicSpectrogramProcessor,
    SpectrogramDifferenceProcessor,
)
from madmom.audio.stft import ShortTimeFourierTransformProcessor
from madmom.processors import ParallelProcessor, SequentialProcessor
from scipy.ndimage import maximum_filter1d

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


class PreProcessor(SequentialProcessor):
    def __init__(
            self, frame_sizes=[2048],
            num_bands=[12],
            fps=FPS, log=np.log, add=1e-6, diff=None, start=None, stop=None, daug_rate=1.):
        sig = SignalProcessor(num_channels=1, sample_rate=44100, start=start, stop=stop)
        multi = ParallelProcessor([])
        for frame_size, num_bands in zip(frame_sizes, num_bands):
            frames = FramedSignalProcessor(frame_size=frame_size, fps=int(np.round(fps * daug_rate)))
            stft = ShortTimeFourierTransformProcessor()
            filt = FilteredSpectrogramProcessor(num_bands=num_bands)
            spec = LogarithmicSpectrogramProcessor(log=log, add=add)
            if diff:
                diff = SpectrogramDifferenceProcessor(positive_diffs=True,
                                                      stack_diffs=np.hstack)
            multi.append(SequentialProcessor((frames, stft, filt, spec, diff)))
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
        audio_files = madmom.utils.search_files(self.path + '/audio', audio_suffix)
        annotation_files = madmom.utils.search_files(self.path + '/annotations/onsets/', onset_suffix)
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
            self.x.append(pre_processor(f))

    def load_annotations(self, widen=None):
        self.annotations = []
        for f in self.annotation_files:
            if f is None:
                beats = np.array([])
            else:
                beats = madmom.io.load_beats(f)
                if beats.ndim > 1:
                    beats = beats[:, 0]

            if self.stop is not None:
                beats = beats[beats <= self.stop]

            if self.start is not None:
                beats = beats - self.start
                beats = beats[beats > 0]

            beats = beats * self.daug_rate
            self.annotations.append(beats)

    def add_dataset(self, dataset):
        self.files.extend(dataset.files)
        self.audio_files.extend(dataset.audio_files)
        self.annotation_files.extend(dataset.annotation_files)
        self.x.extend(dataset.x)
        self.annotations.extend(dataset.annotations)

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


class DataSequence_TCNv2(Sequence):

    def __init__(self, x, beats, fps=FPS, pad_frames=None):
        self.x = x
        self.pad_frames = pad_frames

        self.beats = [madmom.utils.quantize_events(o, fps=fps, length=len(d))
                      for o, d in zip(beats, self.x)]
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
            val = DataSequence_TCNv2(val_db.x, val_db.annotations, pad_frames=2)
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
