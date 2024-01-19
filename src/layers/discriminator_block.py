"""Contains custom reusable block fo the discriminator of spectrogram completion GAN."""

from typing import Tuple, List
import dataclasses


import keras
import tensorflow as tf


@dataclasses.dataclass
class BlockParams:
    """Contains parameters of a single block of the discriminator."""

    # Number of convolutional layers between the first layer and the last downsampling one.
    # This number relates to the main path of the signal, not the residual one.
    n_convolutional_layers: int
    kernel_size: Tuple[int, int]
    n_filters: int


class DiscriminatorBlock(keras.layers.Layer):
    """Contains duwnsampling, residual architecture of the discriminator."""

    def __init__(self, params: BlockParams, *args, **kwargs):
        """Initializes the block with a specific number of internal layers.

        Args:
            params: Parameters of the block.
        """

        super().__init__(*args, **kwargs)

        self._params = params

        self._main_path: List[keras.layers.Layer] = None
        self._residual_path = self._make_residual_path()

        self._combine_layer = keras.layers.Add(name=f'{self.name}/Add')

    def build(self, input_shape: tf.TensorShape):
        """Overrides method of the base keras.Layer class."""

        if len(input_shape) != 4:
            raise ValueError(
                f'Input spectrogram shape must be (batch_size, frequency, time, channels)!')

        self._main_path = self._make_main_path(input_shape)

        super().build(input_shape)

    def call(self, inputs: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        """Overrides method of the base keras.Layer class."""

        residual = inputs

        for layer in self._residual_path:
            residual = layer(residual, *args, **kwargs)

        main = inputs

        for layer in self._main_path:
            main = layer(main, *args, **kwargs)

        return self._combine_layer((residual, main), *args, **kwargs)

    def _make_residual_path(self) -> List[keras.layers.Layer]:
        """Creates internal blocks of the discriminator.
        """

        pooling = keras.layers.AveragePooling2D(
            pool_size=(2, 2),
            name=f'{self.name}/ResidualPooling')

        convolution = keras.layers.Conv2D(
            filters=self._params.n_filters,
            kernel_size=self._params.kernel_size,
            padding='same',
            name=f'{self.name}/ResidualConvolution')

        return [pooling, convolution]

    def _make_main_path(self, input_shape) -> List[keras.layers.Layer]:
        """Creates internal blocks of the discriminator.

        Args:
            input_shape: Shape of the input spectrogram.

        Returns:
            Created layers contained by the main signal processing path.
        """

        input_filters = input_shape[-1]

        normalization = keras.layers.BatchNormalization(name=f'{self.name}/MainNormalizationFirst')

        relu = keras.layers.ReLU(name=f'{self.name}/MainReLUFirst')

        conv = keras.layers.Conv2D(
            filters=input_filters,
            kernel_size=self._params.kernel_size,
            padding='same',
            name=f'{self.name}/MainConvolutionFirst')

        main_convolutions = []

        for i in range(self._params.n_convolutional_layers):

            main_convolutions.append(keras.layers.BatchNormalization(
                name=f'{self.name}/MainNormalization_{i}'))

            main_convolutions.append(keras.layers.ReLU(name=f'{self.name}/MainReLU_{i}'))

            main_convolutions.append(keras.layers.Conv2D(
                filters=self._params.n_filters,
                kernel_size=self._params.kernel_size,
                padding='same',
                name=f'{self.name}/MainConvolution_{i}'))

        last_conv = keras.layers.Conv2D(
            filters=self._params.n_filters,
            kernel_size=self._params.kernel_size,
            padding='same',
            strides=(2, 2),
            name=f'{self.name}/MainConvolutionLast')

        return [normalization, relu, conv] + main_convolutions + [last_conv]
