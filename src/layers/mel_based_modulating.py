# -*- coding: utf-8 -*-
"""Contains a custom layer for co-modulating GANs based on spectrograms."""

from typing import List, Tuple
import dataclasses
import itertools

import keras
import tensorflow as tf


@dataclasses.dataclass
class ModulatorParams:
    """Contains parameters for MelBasedCoModulator layer"""

    mapping_depth: int  # number of dense layers in latent mapping
    mapping_size: int  # size of each dense layer in latent mapping
    # shape of the output tensor (without batch dimension and channels)
    output_shape: Tuple[int, int]
    encoder_depth: int  # number of blocks in spectrogram encoder


class MelBasedCoModulator(keras.layers.Layer):
    """Layer providing co-modulation for spectrogram-spectrogram GANs.

    The layer takes in a MEL spectrogram containing missing regions, from which
    conditioning information shall be derived. The second input is a random latent
    vector providing unconditional modulation for the model.

    Internal image encoder is designed to extract conditioning information from the input
    spectrogram reducing its size to a specified output shape. The decoded spectrogram is
    flattened and combined with mapped latent vector in order to be processed by an internal
    dense layer performing affine transform.

    The input latent vector is mapped by a series of dense layers to the shape of the
    flattened output tensor. The mapping output is reproduced and added to the encoded spectrogram.
    """

    def __init__(self, params: ModulatorParams, *args, **kwargs):
        """Initializes MelBasedCoModulator layer.

        Args:
            params: ModulatorParams object containing parameters for the layer.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """

        super().__init__(*args, **kwargs)

        self._validate_params(params)

        self._params = params

        self._latent_mapping = [
            keras.layers.Dense(self._params.mapping_size, 'relu', name=f'{self.name}/Dense/{idx}')
            for idx in range(self._params.mapping_depth)
        ]

        self._spectrogram_encoder = self._create_encoder()

        self._combine_layer = keras.layers.Add(name=f'{self.name}/Combine')

        self._affine_dense = keras.layers.Dense(
            self._params.output_shape[0] * self._params.output_shape[1],
            name=f'{self.name}/AffineDense'
        )

    def build(self, input_shapes: Tuple[tf.TensorShape, tf.TensorShape]):
        """Overrides method of the base keras.Layer class.

        Args:
            input_shapes: Tuple of shapes for latent input and spectrogram input.
        """

        latent_shape, spectrogram_shape = input_shapes

        if len(latent_shape) != 2 or latent_shape[-1]:
            raise ValueError('Latent input must have shape (batch_size, latent_size)!')

        if len(spectrogram_shape) != 4 or spectrogram_shape[-1] != 1:
            raise ValueError('Spectrogram input must have shape (batch_size, freq, time, 1)!')

        spectrogram_freq_time = (spectrogram_shape[1], spectrogram_shape[2])

        upscaled_output_shape = (
            self._params.output_shape[0] * (2 ** self._params.encoder_depth),
            self._params.output_shape[1] * (2 ** self._params.encoder_depth)
        )

        for given_index, expected_index in zip(spectrogram_freq_time, upscaled_output_shape):
            if given_index < expected_index:
                raise ValueError(
                    'Spectrogram input shape must be reducable to output shape by' +
                    'the internal encoder!')

        super().build(input_shapes)

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor], *args, **kwargs) -> tf.Tensor:
        """Overrides method of the base keras.Layer class.

        Args:
            inputs: Tuple of latent input and spectrogram input.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.`
        """

        output_filters = 2 ** self._params.encoder_depth
        latent, spectrogram = inputs

        mapped_latent = self.map_latent(latent, *args, **kwargs)

        encoded_spectrogram = self.encode_spectrogram(spectrogram, *args, **kwargs)

        combined = self._combine_layer([mapped_latent, encoded_spectrogram])

        output = self._affine_dense(combined)

        return tf.reshape(output, (-1, *self._params.output_shape, output_filters))

    def map_latent(self, latent: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        """Maps latent vector to the shape of the flattened spectrogram."""

        for layer in self._latent_mapping:
            latent = layer(latent, *args, **kwargs)

        return tf.repeat(latent, 2 ** self._params.encoder_depth, axis=-1)

    def encode_spectrogram(self, spectrogram: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        """Encodes spectrogram to the shape of the flattened spectrogram."""

        output_filters = 2 ** self._params.encoder_depth
        flattened_output_size = self._params.output_shape[0] * self._params.output_shape[1]

        for layer in self._spectrogram_encoder:
            spectrogram = layer(spectrogram, *args, **kwargs)

        return tf.reshape(spectrogram, (-1, flattened_output_size * output_filters))

    def _validate_params(self, params: ModulatorParams):
        """Validates layer's parameters.

        Args:
            params: ModulatorParams object containing parameters for the layer.
        """

        if params.mapping_size != (params.output_shape[0] * params.output_shape[1]):
            raise ValueError('Mapping size must be equal to flattened output shape!')

    def _create_encoder(self) -> List[keras.layers.Layer]:
        """Creates a encoder for input spectrogram."""

        def create_encoder_block(filters: int, kernel_size: int, idx: int):
            return [
                keras.layers.Conv2D(filters, kernel_size, padding='same',
                                    name=f'{self.name}/Encoder/Block{idx}/Conv2D'),
                keras.layers.MaxPool2D((2, 2), name=f'{self.name}/Encoder/Block{idx}/MaxPool2D')
            ]

        blocks = [
            create_encoder_block(2 ** (idx + 1), 2 ** (idx + 1), idx)
            for idx in range(self._params.encoder_depth)
        ]

        flatten = [keras.layers.Flatten(name=f'{self.name}/Encoder/Flatten')]

        return list(itertools.chain(*blocks)) + flatten
