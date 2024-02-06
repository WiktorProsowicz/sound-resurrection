"""Contains custom generator for spectrogram completion task."""


from typing import List
import dataclasses

import keras
import tensorflow as tf

from layers import discriminator_block


@dataclasses.dataclass
class DiscriminatorParams:
    """Contains parameters for CompletionDiscriminator layer."""

    n_discriminator_blocks: int
    n_convolutions_per_block: int


class CompletionDiscriminator(keras.layers.Layer):
    """Layer providing discriminator for spectrogram completion task.

    The discriminator is designed to take as input a spectrogram with two channels
    for frequency and phase coefficients. The output shall be a single value
    indicating whether the input is real or fake.
    """

    def __init__(self, params: DiscriminatorParams, *args, **kwargs):
        """Initializes the discriminator layer and its internal blocks.

        Args:
            params: Parameters of the discriminator.
        """

        super().__init__(*args, **kwargs)

        self._params = params

        self._discriminator_blocks = self._make_discriminator_blocks()

        self._final_dense = keras.layers.Dense(1, activation='sigmoid')

    def build(self, input_shape: tf.TensorShape):
        """Overrides method of the base keras.Layer class."""

        if len(input_shape) != 4:
            raise ValueError(f'Input shape must be (batch_size, frequency, time, channels)!')

        super().build(input_shape)

    def call(self, inputs: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        """Overrides method of the base keras.Layer class."""

        inp = inputs

        for block in self._discriminator_blocks:
            inp = block(inp, *args, **kwargs)

        reshaped = tf.reshape(inp, (-1, inp.shape[-1] * inp.shape[-2] * inp.shape[-3]))

        return self._final_dense(reshaped, *args, **kwargs)

    def _make_discriminator_blocks(self) -> List[keras.layers.Layer]:
        """Creates internal blocks of the discriminator.

        Returns:
            A model representing internal blocks of the discriminator.
        """

        blocks = []

        for idx in range(self._params.n_discriminator_blocks):

            params = discriminator_block.BlockParams(1, (3, 3), 2 ** (idx + 1))

            blocks.append(
                discriminator_block.DiscriminatorBlock(
                    params, name=f'{self.name}/Block_{idx}'))

        return blocks
