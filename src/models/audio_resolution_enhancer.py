# -*- coding: utf-8 -*-
"""Contains definitions related to AudioResolutionEnhancer.

It is a custom model class whose functionality enables extending
the frequency range for audio signals.
"""
import dataclasses
from typing import Iterator
from typing import List
from typing import Tuple

import keras
import tensorflow as tf

from layers import dsampling_block as d_block
from layers import subpixel_shuffling_layer as subpixel
from layers import usampling_block as u_block


@dataclasses.dataclass
class ModelConfig:
    """Stores model configuration."""

    source_sampling_rate: int
    target_sampling_rate: int
    n_internal_blocks: int  # Number of upsampling/downsampling blocks in the model.
    dropout_rate: float = .5
    leaky_relu_alpha: float = .2


class AudioResolutionEnhancer(keras.Model):
    """Model class for extending the frequency range for audio signals."""

    def __init__(self, config: ModelConfig, *args, **kwargs):
        """Initialize the model.

        Args:
            config: Model configuration.

        Raises:
            ValueError: If the configuration is invalid.
        """

        super().__init__(name='AudioResolutionEnhancer', *args, **kwargs)

        self._config = config

        self._validate_config()

        self._downsampling_blocks = self._make_downsampling_blocks()

        self._bottleneck = self._make_bottleneck_layers()

        self._upsampling_blocks = self._make_upsampling_blocks()

        self._upsampling_outputs_concat = keras.layers.Add(
            name=f'{self.name}/AdditiveResidual')

        self._final_conv = keras.layers.Conv1D(
            filters=2, kernel_size=9, padding='same',
            name=f'{self.name}/FinalConv1D')

        self._final_shuffling = subpixel.SubPixelShufflingLayer1D(
            2, name=f'{self.name}/FinalSubPixel1D')

    @property
    def upsampling_ratio(self) -> float:
        return self._config.target_sampling_rate / self._config.source_sampling_rate

    def build(self, input_shape: tf.TensorShape):
        """Overrides keras.Model build method.

        The `input_shape` is expected not to have the batch size dimension.

        Raises:
            ValueError: If the input shape does not match the expected one.
        """

        if input_shape.ndims != 2 or input_shape[-1] != 1:
            raise ValueError('Input shape must be (input_samples_len, 1).')

        super().build((None,) + input_shape)

    def call(self, inputs: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        """Overrides keras.Model call method."""

        dsampling_outputs, bottleneck_input = self._run_downsampling(inputs)

        bottleneck_output = self._run_bottleneck(bottleneck_input, *args, **kwargs)

        upsampling_output = self._run_upsampling(
            bottleneck_output, reversed(dsampling_outputs), *args, **kwargs)

        upsampling_concatenated = self._upsampling_outputs_concat([upsampling_output, inputs])
        upsampling_convoluted = self._final_conv(upsampling_concatenated, *args, **kwargs)
        return self._final_shuffling(upsampling_convoluted, *args, **kwargs)

    def _validate_config(self):
        """Validate model configuration.

        Args:
            config: Model configuration.

        Raises:
            ValueError: If the configuration is invalid.
        """

        if self._config.source_sampling_rate <= 0:
            raise ValueError('Source sampling rate must be positive!')

        if self._config.target_sampling_rate <= 0:
            raise ValueError('Target sampling rate must be positive!')

        if self._config.n_internal_blocks <= 0:
            raise ValueError('Number of internal blocks must be positive!')

        if self._config.dropout_rate < 0 or self._config.dropout_rate > 1:
            raise ValueError('Dropout rate must be in range [0, 1]!')

        if self._config.leaky_relu_alpha < 0:
            raise ValueError('Leaky ReLU alpha must be non-negative!')

    def _make_downsampling_blocks(self) -> List[d_block.DownSamplingBlock]:
        """Makes downsampling section of the model's architecture.

        Returns:
            List of created downsampling blocks.
        """

        created_blocks: List[d_block.DownSamplingBlock] = []

        for d_block_num in range(1, self._config.n_internal_blocks + 1):
            filters = max(2 ** (6 + d_block_num), 512)
            length = min(2 ** (7 - d_block_num) + 1, 9)

            dsampling_block = d_block.DownSamplingBlock(
                filters, length, 2, name=f'{self.name}/DownSamplingBlock/{d_block_num}')

            created_blocks.append(dsampling_block)

        return created_blocks

    def _run_downsampling(self,
                          inputs: tf.Tensor,
                          *args,
                          **kwargs) -> Tuple[List[tf.Tensor], tf.Tensor]:

        blocks_outputs: List[tf.Tensor] = []

        for block in self._downsampling_blocks:
            inputs = block(inputs, *args, **kwargs)
            blocks_outputs.append(inputs)

        return blocks_outputs, inputs

    def _make_upsampling_blocks(self) -> List[u_block.UpSamplingBlock]:
        """Makes upsampling section of the model's architecture.

        Returns:
            List of created blocks.
        """

        created_blocks: List[u_block.UpSamplingBlock] = []

        for block_num in range(self._config.n_internal_blocks):

            filters = max(2 ** (7 + (self._config.n_internal_blocks - block_num + 1)), 512)
            length = min(2 ** (7 - (self._config.n_internal_blocks - block_num + 1)) + 1, 9)

            params = u_block.UpSamplingBlockParams(filters, length, .5, 2)
            upsampling_block = u_block.UpSamplingBlock(
                params, name=f'{self.name}/UpSamplingBlock/{block_num}')

            created_blocks.append(upsampling_block)

        return created_blocks

    def _run_upsampling(self, inputs: tf.Tensor,
                        downsampling_outputs: Iterator[tf.Tensor], *args, **kwargs) -> tf.Tensor:

        for downsampling_output, block in zip(downsampling_outputs, self._upsampling_blocks):
            inputs = block((inputs, downsampling_output), *args, **kwargs)

        return inputs

    def _make_bottleneck_layers(self) -> List[keras.layers.Layer]:
        """Makes bottleneck part of the model's architecture.

        Returns:
            Created bottleneck layers.
        """

        conv = keras.layers.Conv1D(filters=self._downsampling_blocks[-1].filters,
                                   kernel_size=self._downsampling_blocks[-1].filter_length,
                                   padding='same', name=f'{self.name}/BottleneckConv1D')

        dropout = keras.layers.Dropout(
            rate=self._config.dropout_rate,
            name=f'{self.name}/BottleneckDropout')

        bottleneck = keras.layers.LeakyReLU(
            self._config.leaky_relu_alpha,
            name=f'{self.name}/BottleneckLeakyReLU')

        return [conv, dropout, bottleneck]

    def _run_bottleneck(self, inputs: tf.Tensor, *args, **kwargs) -> tf.Tensor:

        for layer in self._bottleneck:
            inputs = layer(inputs, *args, **kwargs)

        return inputs
