# -*- coding: utf-8 -*-
"""Module contains simple processing node for resampling audio signals."""
from typing import Any
from typing import Dict

import librosa

from preprocessing import audio_signal
from preprocessing.processing_nodes import processing_node


class ResamplingProcessor(processing_node.ProcessingNode):
    """Simple processing node for resampling audio signals."""

    def __init__(self, allow_backward_processing, sampling_rate):
        """Initialize resampling processor.

        Args:
            sampling_rate: Target sampling rate for processed signals.
        """
        super().__init__(allow_backward_processing)

        self._sampling_rate = sampling_rate
        self._original_sampling_rates: Dict[audio_signal.AudioSignal, int] = {}

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> processing_node.ProcessingNode:
        """Overrides base class method."""

        return cls(config['allow_backward_processing'], config['sampling_rate'])

    def process(
            self, signal: audio_signal.AudioSignal) -> audio_signal.AudioSignal:
        """Overrides base class method."""

        transformed = self._apply_transformations(signal, sr=self._sampling_rate)

        if self._allow_backward:
            self._store_processing_info(transformed, sr=signal.meta.sampling_rate)

        return transformed

    @property
    def signature(self) -> str:
        """Overrides base class method."""

        return f'ResamplingProcessor(sampling_rate={self._sampling_rate})'

    def _apply_transformations(
            self, signal: audio_signal.AudioSignal, *args, **kwargs) -> audio_signal.AudioSignal:
        """Overrides base class method.

        Args:
            **kwargs: Keyword arguments, explained below.

            Keyword arguments:
                sr: Target sampling rate for the signal.
        """

        meta_data = audio_signal.AudioMeta(
            kwargs['sr'],
            signal.meta.channels,
            signal.meta.bits_per_sample)

        resampled_data = librosa.resample(
            signal.data,
            orig_sr=signal.meta.sampling_rate,
            target_sr=meta_data.sampling_rate, res_type='soxr_qq')

        return audio_signal.AudioSignal(resampled_data, meta_data)

    def _transform_backwards(
            self, signal: audio_signal.AudioSignal) -> audio_signal.AudioSignal:
        """Overrides base class method."""

        if signal not in self._original_sampling_rates:
            raise processing_node.AudioProcessingError(
                'No info about original sampling rate for the given signal!')

        return self._apply_transformations(signal, sr=self._original_sampling_rates[signal])

    def _store_processing_info(
            self, signal: audio_signal.AudioSignal, *args, **kwargs):
        """Overrides base class method.

        Args:
            **kwargs: Keyword arguments, explained below.

            Keyword arguments:
                sr: Sampling rate of the original signal.
        """

        self._original_sampling_rates[signal] = kwargs['sr']
