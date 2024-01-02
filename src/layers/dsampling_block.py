# -*- coding: utf-8 -*-
"""Contains custom downsampling layer related to AudioResolutionEnhancer model."""
import keras
import tensorflow as tf


class DownSamplingBlock(keras.layers.Layer):
    """Downsampling block for AudioResolutionEnhancer model.

    It consists of a one-dimensional convolutional layer, followed by a RELU activation layer.
    """

    def __init__(self, filters_count: int, filter_length: int, stride: int, *args, **kwargs):
        """Initializes the layer.

        Args:
            filters_count: Number of filters in the convolutional layer.
            filter_length: Length of the filters in the convolutional layer.
            stride: Stride of the convolutional layer.
        """

        super().__init__(*args, **kwargs)

        self._filters_count = filters_count
        self._filter_length = filter_length
        self._stride = stride

        self._conv = keras.layers.Conv1D(
            filters=self._filters_count,
            kernel_size=self._filter_length,
            strides=self._stride,
            padding='same',
            name=f'{self.name}_Conv1D',
        )

        self._batch_norm = keras.layers.BatchNormalization(axis=-2, name=f'{self.name}/BatchNorm')

        self._output_layer = keras.layers.ReLU(name=f'{self.name}/ReLU')

    @property
    def filters(self) -> int:
        return self._filters_count

    @property
    def filter_length(self) -> int:
        return self._filter_length

    def build(self, input_shape: tf.TensorShape):
        """Overrides keras.Model build method.

        Raises:
            ValueError: If the input shape does not match the expected one.
        """

        if input_shape.ndims < 2:
            raise ValueError('Input shape of a sample must be (input_samples_len, channels)!')

        super().build(input_shape)

    def call(self, inputs: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        """Overrides keras.Model call method."""

        _ = self._conv(inputs, *args, **kwargs)
        _ = self._batch_norm(_, *args, **kwargs)
        return self._output_layer(_, *args, **kwargs)
