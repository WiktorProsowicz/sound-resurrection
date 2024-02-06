# -*- coding: utf-8 -*-
"""Contains generator part for GAN completing spectrograms."""

import dataclasses
from typing import List, Tuple

import keras
import tensorflow as tf

from layers import generator_block
from layers import mel_based_modulating
from layers import phase_conditioning


@dataclasses.dataclass
class GeneratorParams:
    """Contains parameters for CompletionGenerator layer."""

    n_generator_blocks: int
    n_convolutions_per_block: int
    # Number of internal blocks applying phase conditioning.
    n_conditioning_blocks: int


class CompletionGenerator(keras.layers.Layer):
    """Layer providing generator for completing spectrograms.

    The generator is designed to upsample input co-modulation info
    in shape (batch_size, frequency, time, filters). The output shall have a
    shape containing batch size, expected spectrogram size and two channels for
    frequency and phase coefficients.
    """

    def __init__(self, params: GeneratorParams,
                 modulator: mel_based_modulating.MelBasedCoModulator, *args, **kwargs):
        """Initializes the generator layer and its internal blocks.

        Args:
            params: Parameters of the generator.
            modulator: Layer providing co-modulation info from latent vector and input spectrogram.
        """

        super().__init__(*args, **kwargs)

        self._params = params

        self._co_modulator = modulator

        self._generator_blocks = self._make_generator_blocks()

        self._conditioning_blocks = self._make_conditioning_blocks()

    def build(self, input_shapes: Tuple[tf.TensorShape, tf.TensorShape, tf.TensorShape]):
        """Overrides method of the base keras.Layer class.

        Args:
            input_shapes: Shapes of the input latent vector, spectrogram
            and phase conditioning info.
        """

        latent_shape, spectrogram_shape, phase_shape = input_shapes

        if len(latent_shape) != 2:
            raise ValueError(f'The shape of the latent vector must be (batch_size, latent_size)!')

        if len(spectrogram_shape) != 4 or spectrogram_shape[-1] != 1:
            raise ValueError(
                f'The shape of the spectrogram must be (batch_size, frequency, time, 1)!')

        if len(phase_shape) != 4 or phase_shape[-1] != 1:
            raise ValueError(
                f'Phase conditioning info must have shape (batch_size, frequency, time, 1)!')

        super().build(input_shapes)

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor, tf.Tensor], *args, **kwargs) -> tf.Tensor:
        """Overrides method of the base keras.Layer class.

        Args:
            inputs: Tuple of latent vector, spectrogram and phase conditioning info.
        """

        latent_vector, spectrogram, phase_info = inputs

        co_modulation = self._co_modulator((latent_vector, spectrogram))

        filtered_spec = co_modulation
        for upsampling_block in self._generator_blocks:
            filtered_spec = upsampling_block(co_modulation, *args, **kwargs)

        for conditioning_block in self._conditioning_blocks:
            filtered_spec = conditioning_block(filtered_spec, phase_info, *args, **kwargs)

        return filtered_spec

    def _make_generator_blocks(self) -> List[generator_block.GeneratorBlock]:
        """Creates a list of upsampling generator blocks.

        Returns:
            List of generator blocks.
        """

        blocks = []

        for idx in range(self._params.n_generator_blocks):

            n_filters = min(10, 2 ** (self._params.n_generator_blocks - idx))

            params = generator_block.BlockParams(
                self._params.n_convolutions_per_block, (3, 3), n_filters)

            blocks.append(
                generator_block.GeneratorBlock(
                    params, name=f'{self.name}/GeneratorBlock{idx}'))

        return blocks

    def _make_conditioning_blocks(self) -> List[mel_based_modulating.MelBasedCoModulator]:
        """Creates a list of conditioning blocks.

        Returns:
            List of conditioning blocks.
        """

        blocks = []

        for idx in range(self._params.n_conditioning_blocks - 1):

            blocks.append(
                phase_conditioning.PhaseConditioningBlock(
                    (3, 3), 10, name=f'{self.name}/PhaseConditioningBlock{idx}')
            )

        blocks.append(
            phase_conditioning.PhaseConditioningBlock(
                (3, 3), 2, name=f'{self.name}/PhaseConditioningBlockLast')
        )

        return blocks
