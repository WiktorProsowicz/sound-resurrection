# -*- coding: utf-8 -*-
"""Contains definition of a processing node performing noise addition to audio signal."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List
import random

import numpy as np

from preprocessing.processing_nodes import processing_node
from preprocessing import audio_signal


class NoiseProvider(ABC):
    """Provides the caller with noise to be added to audio signal.

    Properties of the generated noise depend on the functionality of concrete noise provider.
    """

    @abstractmethod
    def generate_noise(self, signal: audio_signal.AudioSignal) -> np.ndarray:
        """Generates noise to be added to the given signal.

        Args:
            signal: Audio signal to which the noise shall be added.

        Returns:
            Generated noise.
        """


class NoiseAddingProcessor(processing_node.ProcessingNode):
    """Adds noise to audio signals.

    The type and intensity of the noise is determined by the
    processor's configuration. It can be derived from the type
    of the concrete noise provider whether an artificial random
    noise, the one taken from the real worlds examples or any other
    possible. If initialized with multiple noise providers, the
    processor shall choose randomly at each processed signal.
    """

    def __init__(self, allow_backward_processing: bool, noise_providers: List[NoiseProvider]):
        """Initializes the base class and sets internal processor's config.

        Args:
            allow_backward_processing: Whether to allow reversal of the noise addition operation.
            noise_providers: Noise providers the processor shall get noise from.

        Raises:
            ValueError: If no noise providers were given.
        """

        if not noise_providers:
            raise ValueError('At least one noise provider must be given!')

        super().__init__(allow_backward_processing)

        self._noise_providers = noise_providers
        self._saved_noise: Dict[audio_signal.AudioSignal, np.ndarray] = {}

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'processing_node.ProcessingNode':
        """Overrides method of ProcessingNode class.

        Raises:
            ValueError: If there was specified unknown noise provider in the config.
        """

        known_providers = {
            "AWGNoiseProvider": AWGNoiseProvider
        }

        def parse_provider(provider_cfg: Dict[str, Any]) -> NoiseProvider:
            if len(provider_cfg) != 1:
                raise ValueError(
                    'Noise provider config must contain exactly one key being its type!')

            for name, params in provider_cfg.items():
                if name not in known_providers:
                    raise ValueError(f'Unknown noise provider: {name}')

                return known_providers[name](**params)

        providers = [parse_provider(provider_cfg) for provider_cfg in config['noise_providers']]

        return cls(config['allow_backward_processing'], providers)

    def process(
            self, signal: audio_signal.AudioSignal) -> audio_signal.AudioSignal:
        """Overrides method of ProcessingNode class."""

        chosen_provider = random.choice(self._noise_providers)
        noise = chosen_provider.generate_noise(signal)

        transformed = self._apply_transformations(signal, noise=noise)

        if self._allow_backward:
            self._store_processing_info(transformed, noise=noise)

        return transformed

    @property
    def signature(self) -> str:
        """Overrides method of ProcessingNode class."""

        return f'NoiseAddingProcessor(noise_providers={self._noise_providers})'

    def _apply_transformations(
            self, signal: audio_signal.AudioSignal, *args, **kwargs) -> audio_signal.AudioSignal:
        """Overrides method of ProcessingNode class.

        Args:
            **kwargs: Listed below

            Keyword arguments:
                noise: Noise to be added to the signal.
        """

        return audio_signal.AudioSignal(signal.data + kwargs['noise'], signal.meta)

    def _transform_backwards(
            self, signal: audio_signal.AudioSignal) -> audio_signal.AudioSignal:
        """Overrides method of ProcessingNode class."""

        if signal in self._saved_noise:
            return audio_signal.AudioSignal(signal.data - self._saved_noise[signal], signal.meta)

        raise processing_node.AudioProcessingError("No noise has been saved for the given signal!")

    def _store_processing_info(
            self, signal: audio_signal.AudioSignal, *args, **kwargs):
        """Overrides method of ProcessingNode class.

        Args:
            **kwargs: Listed below

            Keyword arguments:
                noise: Noise added to the signal.
        """

        self._saved_noise[signal] = kwargs['noise']


class AWGNoiseProvider(ABC):
    """Generates Additive White Gaussian noise."""

    def __init__(self, noise_std: float):
        """Initializes the noise provider with given standard deviation.

        Args:
            noise_std: Standard deviation of the generated noise.

        Raises:
            ValueError: If the given standard deviation is invalid.
        """

        if noise_std <= 0:
            raise ValueError('Standard deviation must be positive!')

        self._noise_std = noise_std

    def generate_noise(self, signal: audio_signal.AudioSignal) -> np.ndarray:
        """Overrides method of NoiseProvider class."""

        return np.random.normal(scale=self._noise_std, size=signal.data.shape)

    def __repr__(self) -> str:

        return f'AWGNoiseProvider(noise_std={self._noise_std})'
