# -*- coding: utf-8 -*-
"""Contains custom implementation of 1D subpixel shuffling layer."""
import keras
import tensorflow as tf


class SubPixelShufflingLayer1D(keras.layers.Layer):
    """Implementation of sub-pixel shuffling.

    It performs a rearrangement of the tensor's channels, squashing each (a, x) block
    and repositioning it to a (a * x, 1) block.
    """

    def __init__(self, upsampling_factor: int, *args, **kwargs):
        """Initializes the layer, setting the shuffling parameter.

        Args:
            upsampling_factor: Specifies the length of the side of the square block
                resulting from the shuffling.
        """

        super().__init__(*args, **kwargs)

        self._upsampling_factor = upsampling_factor

        self._output_layer: keras.layers.Layer = None

    def build(self, input_shape: tf.TensorShape):
        """Overrides keras.Model build method.

        Raises:
            ValueError: If the given input shape is ill-formed from the
            sub-pixel shuffling perspective.
        """
        if len(input_shape) < 2 or (input_shape[-1] % self._upsampling_factor) != 0:

            raise ValueError(
                'The input shape does not conform to the sub-pixel input expectations!' +
                ' Should be at least dwo dimensional and have the form' +
                '(size, n * `upsampling_factor``).')

        input_samples_length = input_shape[-2]
        input_channels = input_shape[-1]

        output_shape = (input_samples_length *
                        self._upsampling_factor,
                        input_channels //
                        self._upsampling_factor)

        self._output_layer = keras.layers.Reshape(output_shape, name=f'{self.name}/Reshape')

        super().build(input_shape)

    def call(self, inputs: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        """Overrides keras.Model call method."""

        return self._output_layer(inputs, *args, **kwargs)
