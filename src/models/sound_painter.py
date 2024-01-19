# -*- coding: utf-8 -*-
"""Contains definitions related to SoundPainter model."""

import dataclasses
from typing import Tuple, NamedTuple
import collections

import keras
import tensorflow as tf

from layers import mel_based_modulating
from layers import completion_generator
from layers import completion_discriminator


@dataclasses.dataclass
class SoundPainterParams:
    """Contains parameters for SoundPainter model."""

    latent_size: int
    n_generator_blocks: int
    n_generator_cond_blocks: int
    n_discriminator_blocks: int
    latent_mapping_depth: int

    # Specify the coefficients for generator loss function parts. These are respectively:
    # (Contextual loss, Gradient loss, Wasserstein loss)
    generator_loss_coeffs: Tuple[float, float, float]

    # Specify the coefficients for discriminator loss function parts. These are respectively:
    # (Gradient penalty, Wasserstein loss)
    discriminator_loss_coeffs: Tuple[float, float]


class SoundPainter(keras.Model):
    """Model class filling missing gaps in audio signals.

    It is a custom model based on Wasserstein GAN concept including gradient penalty
    and co-modulation. Its main role is to fill missing gaps in audio
    signals converted to MEL spectrograms. The model takes in multiple inputs:
        - input MEL spectrogram
        - matrix containing phase coefficients from FFT operation
        - random noise vector, on which the model bases its modelling of the real
            sounds distribution
    """

    _DiscriminatorLosses = collections.namedtuple(
        'DiscriminatorLosses',
        ['wasserstein_loss', 'gradient_penalty', 'total'])

    _GeneratorLosses = collections.namedtuple(
        'GeneratorLosses',
        ['contextual_loss', 'gradient_loss', 'wasserstein_loss', 'total'])

    def __init__(self, params: SoundPainterParams, *args, **kwargs):
        """Initializes SoundPainter model.

        Args:
            params: SoundPainterParams object containing parameters for the model.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """

        super().__init__(name='SoundPainter', *args, **kwargs)

        self._params = params

        self._co_modulation: mel_based_modulating.MelBasedCoModulator = None
        self._generator: completion_generator.CompletionGenerator = None
        self._discriminator: completion_discriminator.CompletionDiscriminator = None

        self._generator_optimizer: keras.optimizers.Optimizer = None
        self._discriminator_optimizer: keras.optimizers.Optimizer = None

    def build(self, input_shapes: Tuple[tf.TensorShape, tf.TensorShape]):
        """Overrides method of the base keras.Model class.

        Args:
            input_shapes: Shapes of input MEL spectrogram and phase coefficients.
        """

        spectrogram_shape, phase_shape = input_shapes

        if len(spectrogram_shape) != 4 or spectrogram_shape[-1] != 1:
            raise ValueError(
                f'Input spectrogram shape must be (batch_size, frequency, time, 1)!')

        if len(phase_shape) != 4 or phase_shape[-1] != 1:
            raise ValueError(
                f'Input phase shape must be (batch_size, frequency, time, 1)!')

        self._co_modulation = self._make_co_modulation(spectrogram_shape)
        self._generator = self._make_generator()
        self._discriminator = self._make_discriminator()

    def compile(self,
                gen_optimizer: keras.optimizers.Optimizer,
                dis_optimizer: keras.optimizers.Optimizer,
                *args, **kwargs):
        """Overrides method of the base keras.Model class."""

        super().compile(*args, **kwargs)

        self._generator_optimizer = gen_optimizer
        self._discriminator_optimizer = dis_optimizer

    def train_step(self, data: Tuple[Tuple[tf.Tensor, tf.Tensor], tf.Tensor]) -> dict:
        """Overrides method of the base keras.Model class.

        Args:
            data: Tuple containing two parts of input data:
                - Tuple containing MEL spectrogram and phase coefficients
                - Tensor with shape (batch_size, freq, time, 2) where the last dimension
                    is a concatenation of frequency coefficients and phase coefficients

        Returns:
            Dictionary containing named loss values.
        """

        (inp_spectrogram, inp_phase), real_spectrogram = data

        batch_size = tf.shape(inp_spectrogram)[0]

        random_vector = tf.random.normal((batch_size, self._params.latent_size))
        generated = self._generator((random_vector, inp_spectrogram, inp_phase))

        with tf.GradientTape() as tape:
            disc_losses = self._compute_discriminator_losses(real_spectrogram, generated)

        disc_gradients = tape.gradient(disc_losses.total, self._discriminator.trainable_variables)

        self._discriminator_optimizer.apply_gradients(
            zip(disc_gradients, self._discriminator.trainable_variables))

        random_vector = tf.random.normal((batch_size, self._params.latent_size))

        with tf.GradientTape() as tape:
            gen_losses = self._compute_generator_losses(
                random_vector, inp_spectrogram, inp_phase, inp_spectrogram != 0)

        gen_gradients = tape.gradient(gen_losses.total, self._generator.trainable_variables)

        self._generator_optimizer.apply_gradients(
            zip(gen_gradients, self._generator.trainable_variables))

    def test_step(self, data: Tuple[Tuple[tf.Tensor, tf.Tensor], tf.Tensor]) -> dict:
        """Overrides method of the base keras.Model class.

        Args:
            data: Tuple containing two parts of input data:
                - Tuple containing MEL spectrogram and phase coefficients
                - Tensor with shape (batch_size, freq, time, 2) where the last dimension
                    is a concatenation of frequency coefficients and phase coefficients

        Returns:
            Dictionary containing named loss values.
        """

        (inp_spectrogram, inp_phase), real_spectrogram = data

        batch_size = tf.shape(inp_spectrogram)[0]

        random_vector = tf.random.normal((batch_size, self._params.latent_size))
        generated = self._generator((random_vector, inp_spectrogram, inp_phase))

        disc_losses = self._compute_discriminator_losses(real_spectrogram, generated)
        gen_losses = self._compute_generator_losses(
            random_vector, inp_spectrogram, inp_phase, inp_spectrogram != 0)

        return {
            'discriminator_loss': disc_losses.total,
            'generator_loss': gen_losses.total,
            'discriminator_wasserstein_loss': disc_losses.wasserstein_loss,
            'discriminator_gradient_penalty': disc_losses.gradient_penalty,
            'generator_contextual_loss': gen_losses.contextual_loss,
            'generator_gradient_loss': gen_losses.gradient_loss,
            'generator_wasserstein_loss': gen_losses.wasserstein_loss
        }

    def _compute_discriminator_losses(self,
                                      real: tf.Tensor,
                                      generated: tf.Tensor) -> '_DiscriminatorLosses':
        """Computes discriminator's loss parts.

        Args:
            real: Spectrogram sampled from real data.
            generated: Spectrogram generated by the generator.

        Returns:
            Tuple containing discriminator's loss values.
        """

        gen_labels = self._discriminator(generated)
        real_labels = self._discriminator(real)

        wass_coeff, grad_penalty_coeff = self._params.discriminator_loss_coeffs

        wasserstein_loss = -(tf.reduce_mean(real_labels) - tf.reduce_mean(gen_labels))
        gradient_penalty = self._compute_gradient_penalty(real, generated)

        disc_loss = wass_coeff * wasserstein_loss + grad_penalty_coeff * gradient_penalty

        return self._DiscriminatorLosses(wasserstein_loss, gradient_penalty, disc_loss)

    def _compute_generator_losses(self,
                                  random_vec: tf.Tensor,
                                  inp_spectrogram: tf.Tensor,
                                  inp_phase: tf.Tensor,
                                  spec_mask: tf.Tensor) -> '_GeneratorLosses':
        """Computes generator's loss parts.

        Args:
            random_vec: Random noise vector.
            inp_spectrogram: Spectrogram with missing areas.
            inp_phase: Phase coefficients computed from `inp_spectrogram`.
            spec_mask: Boolean mask indicating which areas are missing.

        Returns:
            Tuple containing generator's loss values.
        """

        generated = self._generator((random_vec, inp_spectrogram, inp_phase))

        gen_labels = self._discriminator(generated)

        contextual_loss = tf.reduce_mean(tf.abs(spec_mask * (generated - inp_spectrogram)))
        gradient_loss = tf.reduce_mean(tf.abs(tf.image.image_gradients(generated)))
        wass_loss = -tf.reduce_mean(gen_labels)

        context_coeff, grad_coeff, wass_coeff = self._params.generator_loss_coeffs

        gen_loss = (wass_coeff * wass_loss +
                    grad_coeff * gradient_loss +
                    context_coeff * contextual_loss)

        return self._GeneratorLosses(contextual_loss, gradient_loss, wass_loss, gen_loss)

    def _compute_gradient_penalty(self, real_spectrogram: tf.Tensor,
                                  generated: tf.Tensor) -> tf.Tensor:
        """Computes gradient penalty for discriminator loss function.

        Args:
            real_spectrogram: Spectrogram sampled from real data.
            generated: Spectrogram generated by the generator.

        Returns:
            Gradient penalty value. It is not scaled by any coefficient.
        """

        batch_size = tf.shape(real_spectrogram)[0]

        alpha = tf.random.uniform(real_spectrogram.shape)
        interpolated = alpha * real_spectrogram + (1 - alpha) * generated

        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            dis_labels = self._discriminator(interpolated)

        gradients = tape.gradient(dis_labels, interpolated)
        gradients = tf.reshape(gradients, (batch_size, -1))

        gradient_norm = tf.norm(gradients, axis=-1)

        return tf.reduce_mean((gradient_norm - 1) ** 2)

    def _make_co_modulation(
            self, spectrogram_shape: tf.TensorShape) -> mel_based_modulating.MelBasedCoModulator:
        """Creates co-modulation layer. """

        downsampling_factor = 2 ** self._params.n_generator_blocks

        dsampled_freq = spectrogram_shape[1] // downsampling_factor
        dsampled_time = spectrogram_shape[2] // downsampling_factor

        params = mel_based_modulating.ModulatorParams(
            self._params.latent_mapping_depth,
            dsampled_freq * dsampled_time,
            (dsampled_freq, dsampled_time),
            self._params.n_generator_blocks)

        return mel_based_modulating.MelBasedCoModulator(params, name=f'{self.name}/CoModulation')

    def _make_generator(self) -> completion_generator.CompletionGenerator:
        """Creates generator model."""

        params = completion_generator.GeneratorParams(
            self._params.n_generator_blocks,
            1,
            self._params.n_generator_cond_blocks)

        return completion_generator.CompletionGenerator(
            params, self._co_modulation, name=f'{self.name}/Generator')

    def _make_discriminator(self) -> completion_discriminator.CompletionDiscriminator:
        """Creates discriminator model."""

        params = completion_discriminator.DiscriminatorParams(
            self._params.n_discriminator_blocks, 1)

        return completion_discriminator.CompletionDiscriminator(
            params, name=f'{self.name}/Discriminator')
