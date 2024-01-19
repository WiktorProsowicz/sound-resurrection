# -*- coding: utf-8 -*-
"""Module contains definition of custom block for GAN generator."""

from typing import Tuple

import keras
import tensorflow as tf


class PhaseConditioningBlock(keras.layers.Layer):
    """Performs convolutional filtering and conditioning.

    This is a custom block used by the generator of spectrogram completing GAN.
    It is designed to transform an input spectrogram using small convolutional
    filters to extract fine-grained features. Before the filtration an external
    conditioning information containing phase FFT coefficients is concatenated
    with the input.
    """

    def __init__(self, kernel_size: Tuple[int, int], n_filters: int, *args, **kwargs):
        """Initializes the block with a specific number of internal layers.

        Args:
            kernel_size: Size of the convolutional kernel.
            n_filters: Number of filters in internal convolutional layer.
        """

        super().__init__(*args, **kwargs)

        self._concat = keras.layers.Concatenate(axis=3, name=f'{self.name}/Concatenate')

        self._convolution = keras.layers.Conv2D(
            filters=n_filters,
            kernel_size=kernel_size,
            padding='same',
            name=f'{self.name}/Convolution')

    def build(self, input_shapes: Tuple[tf.TensorShape, tf.TensorShape]):
        """Overrides method of the base keras.Layer class.

        Args:
            input_shapes: Shapes of the input spectrogram and phase coefficients.
        """

        spectrogram_shape, phase_shape = input_shapes

        if len(spectrogram_shape) != 4:
            raise ValueError(
                f'Input spectrogram shape must be (batch_size, frequency, time, channels)!')

        if len(phase_shape) != 4 or phase_shape[-1] != 1:
            raise ValueError(f'Input phase shape must be (batch_size, frequency, time, 1)!')

        super().build(input_shapes)

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor], *args, **kwargs) -> tf.Tensor:
        """Overrides method of the base keras.Layer class.

        Args:
            inputs: Tuple containing spectrogram and phase coefficients.

        Returns:
            Filtered and conditioned spectrogram.
        """

        spectrogram, phase = inputs

        concatenated = self._concat((spectrogram, phase))

        return self._convolution(concatenated, *args, **kwargs)
