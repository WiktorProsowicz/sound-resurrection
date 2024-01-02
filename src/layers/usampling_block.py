# -*- coding: utf-8 -*-
"""Contains definition of the upsampling block for AudioResolutionEnhancer model."""
import dataclasses
from typing import Tuple

import keras
import tensorflow as tf

from layers import subpixel_shuffling_layer as subpixel


@dataclasses.dataclass
class UpSamplingBlockParams:
    """Contains configuration parameters for upsampling block."""

    filters_count: int
    filter_length: int
    dropout_rate: float
    subpixel_upsampling_factor: int


class UpSamplingBlock(keras.layers.Layer):
    """Upsampling block for AudioResolutionEnhancer bottleneck architecture.

    It consists of the following layers:
        - One-dimensional transposed convolutional layer.
        - Dropout layer.
        - RELU activation layer.
        - SubPixel shuffling layer.

    When used as a part of upsampling block sequence, the `filters_count` parameter
    should be set to a value n times smaller than the one from the previous layer.
    Thus the input tensor's dimension d x F is reduced to d x F/n, where d is the
    number of samples and F is the number of filters. Then the SubPixel shuffling
    layer should increase the number of samples by a factor of n, further reducing the
    filters count (d x F/n -> nd x F/2n). If a skip connection from the respective
    downsampling block is present, the input tensor is concatenated with the output
    of the upsampling block, resulting in xd x F/2x + 2d x F/2x = xd x F/x tensor.
    """

    def __init__(self, params: UpSamplingBlockParams,
                 *args, **kwargs):
        """Initializes the layer.

        Args:
            params: Parameters for the upsampling block.
        """

        super().__init__(*args, **kwargs)

        self._params = params

        self._conv = keras.layers.Conv1DTranspose(
            filters=self._params.filters_count,
            kernel_size=self._params.filter_length,
            padding='same',
            name=f'{self.name}/Conv1DTranspose'
        )

        self._batch_norm = keras.layers.BatchNormalization(axis=-2, name=f'{self.name}/BatchNorm')

        self._dropout = keras.layers.Dropout(
            rate=self._params.dropout_rate,
            name=f'{self.name}/Dropout')

        self._concatenate = keras.layers.Concatenate(axis=2)

        self._output_layer = subpixel.SubPixelShufflingLayer1D(
            self._params.subpixel_upsampling_factor, name=f'{self.name}/SubPixel1D')

    def build(self, input_shape: tf.TensorShape):
        """Overrides keras.Model build method.

        Raises:
            ValueError: If the input shape does not match the expected one.
        """

        if len(input_shape) < 2:
            raise ValueError(
                'Input shape of a sample must be (input_samples_len, channels)!')

        super().build(input_shape)

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor], *args, **kwargs) -> tf.Tensor:
        """Overrides keras.Model call method.

        Args:
            inputs: Tuple of tensors, where the first one is the input tensor and the
                second one is the respective downsampling layer's output.
        """

        layer_input, dsampling_features = inputs

        _ = self._conv(layer_input, *args, **kwargs)
        _ = self._batch_norm(_, *args, **kwargs)
        _ = self._dropout(_, *args, **kwargs)
        _ = self._concatenate([_, dsampling_features])

        return self._output_layer(_, *args, **kwargs)
