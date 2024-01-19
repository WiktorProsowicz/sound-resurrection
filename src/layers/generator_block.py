# -*- coding: utf-8 -*-
"""Module contains definition of custom block for GAN generator."""

from typing import List
import dataclasses

import keras
import tensorflow as tf


@dataclasses.dataclass
class BlockParams:
    """Contains parameters for GeneratorBlock."""

    # Number of internal convolutional layers. These are the convolutions in
    # the main signal path that does not have normalization. It means there is
    # one additional convolution at the end and one upscaling convolution at the
    # beginning.
    n_convolutions: int
    # Size of the convolutional kernel.
    kernel_size: int
    # Number of filters in internal convolutional layers.
    n_filters: int


class GeneratorBlock(keras.layers.Layer):
    """Custom upsampling block for sound completion GAN generator.

    It contains a series of processing convolutions interleaved with
    normalization and activations. The whole block is wrapped in semi-residual
    connection with one convolution and no normalization. The input spectrogram's
    size is doubled in freq and time dimensions. The number of channels is reduced 2 times.
    """

    def __init__(self, params: BlockParams, *args, **kwargs):
        """Initializes the block with a specific number of internal layers.

        Args:
            n_convolutions:
            kernel_size:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """

        super().__init__(*args, **kwargs)

        self._params = params

        self._residual_path: List[keras.layers.Layer] = None
        self._main_path: List[keras.layers.Layer] = None

        self._combine_layer = keras.layers.Add(name=f'{self.name}/ResidualAdd')

    def build(self, input_shape: tf.TensorShape):
        """Overrides method of the base keras.Layer class."""

        if len(input_shape) != 4:
            raise ValueError(
                f'Input spectrogram shape must be (batch_size, frequency, time, channels)!')

        self._main_path = self._make_main_path(input_shape)
        self._residual_path = self._make_residual_path(input_shape)

        super().build(input_shape)

    def call(self, inputs: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        """Overrides method of the base keras.Layer class."""

        main_path = inputs

        for layer in self._main_path:
            main_path = layer(main_path, *args, **kwargs)

        residual_path = inputs

        for layer in self._residual_path:
            residual_path = layer(residual_path, *args, **kwargs)

        return self._combine_layer((main_path, residual_path), *args, **kwargs)

    def _make_main_path(self, input_shape: tf.TensorShape) -> List[keras.layers.Layer]:
        """Creates the main signal processing path.

        Args:
            input_shape: Shape of the input spectrogram.
        """

        input_filters = input_shape[-1]

        normalization = keras.layers.BatchNormalization(name=f'{self.name}/MainBatchNorm')
        activation = keras.layers.LeakyReLU(name=f'{self.name}/MainLeakyReLU')

        transpose = keras.layers.Conv2DTranspose(input_filters,
                                                 self._params.kernel_size, padding='same',
                                                 name=f'{self.name}/MainTranspose')

        convolutions = [
            keras.layers.Conv2D(
                self._params.n_filters,
                self._params.kernel_size,
                padding='same',
                name=f'{self.name}/Conv2D/{idx}'
            )
            for idx in range(self._params.n_convolutions)
        ]

        last_norm = keras.layers.BatchNormalization(name=f'{self.name}/MainLastBatchNorm')
        last_activation = keras.layers.LeakyReLU(name=f'{self.name}/MainLastLeakyReLU')
        last_conv = keras.layers.Conv2D(
            self._params.n_filters,
            self._params.kernel_size,
            padding='same',
            name=f'{self.name}/LastConv2D'
        )

        return [normalization, activation, transpose,
                *convolutions, last_norm, last_activation, last_conv]

    def _make_residual_path(self, input_shape: tf.TensorShape) -> List[keras.layers.Layer]:
        """Creates the residual signal processing path.

        Args:
            input_shape: Shape of the input spectrogram.
        """

        input_filters = input_shape[-1]

        transpose = keras.layers.Conv2DTranspose(input_filters,
                                                 self._params.kernel_size,
                                                 padding='same',
                                                 name=f'{self.name}/ResidualTranspose')

        conv = keras.layers.Conv2D(
            self._params.n_filters,
            self._params.kernel_size,
            padding='same',
            name=f'{self.name}/ResidualConv2D'
        )

        return [transpose, conv]
