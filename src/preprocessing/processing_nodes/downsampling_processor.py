# -*- coding: utf-8 -*-
import dataclasses
from typing import Any
from typing import Dict

import librosa
import numpy as np
from scipy import signal as sp_signal

from preprocessing import audio_signal
from preprocessing.processing_nodes import processing_node


@dataclasses.dataclass(repr=True, frozen=True)
class ProcessorParams:
    """Contains configuration parameters for DownSamplingProcessor."""

    target_sampling_rate: int
    filter_poly_order: int = 5  # Used Chebyshev polynomial's order
    filter_ripple_extent: float = 1.5  # Maximal drop of frequency response gain in pass-band


class DownSamplingProcessor(processing_node.ProcessingNode):
    """Filters and downsamples an audio signal.

    The processor can be used to create an imitation of signal's distortion
    being a result of cutting of the higher frequencies. An example use case
    can be narrowing of the frequency range before passing the signal through
    a telephony line.

    The processor uses a low-pass Chebyshev Type I filter with adjustable ripple
    factor and used polynomial's order. Cut-off frequency is determined as a nyquist
    frequency of the signal after resampling. Signal produced after the filtering is
    resamples to target sampling rate.
    """

    def __init__(self, allow_backward_processing: bool, params: ProcessorParams):
        """
        Initializes the base class and sets internal processor's config.

        Args:
            allow_backward_processing: Argument for the base class.
            params: Processor's configuration.

        Raises:
            ValueError: If the given config contains invalid values.
        """

        super().__init__(allow_backward_processing)

        self._validate_params(params)

        self._params = params
        # Stores the data taken from the original signal before processing.
        self._original_data: Dict[audio_signal.AudioSignal, np.ndarray] = {}

    @property
    def signature(self) -> str:
        """Overrides method of ProcessingNode class."""

        return f'DownSamplingProcessor(config={self._params}))'

    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        """Overrides method of ProcessingNode class."""

        return cls(config['allow_backward_processing'], ProcessorParams(**config['params']))

    def process(self, signal: audio_signal.AudioSignal) -> audio_signal.AudioSignal:
        """Overrides method of ProcessingNode class."""

        transformed = self._apply_transformations(signal)

        if self._allow_backward:
            self._store_processing_info(transformed, original_data=signal.data.copy())

        return transformed

    def _apply_transformations(self, signal: audio_signal.AudioSignal,
                               *args, **kwargs) -> audio_signal.AudioSignal:
        """Overrides method of ProcessingNode class."""

        downsampled_data = self._downsample_and_filter(signal.data, signal.meta.sampling_rate)

        return audio_signal.AudioSignal(downsampled_data, signal.meta)

    def _transform_backwards(self, signal: audio_signal.AudioSignal) -> audio_signal.AudioSignal:
        """Overrides method of ProcessingNode class."""

        if signal in self._original_data:
            return audio_signal.AudioSignal(self._original_data[signal], signal.meta)

        raise processing_node.AudioProcessingError(
            f'No stored info for {self.signature}!')

    def _store_processing_info(self, signal: audio_signal.AudioSignal, *args, **kwargs):
        """Overrides method of ProcessingNode class.

        Args:
            **kwargs: Listed below

            Keyword arguments:
              original_data: Original signal's data before processing.
        """

        data = kwargs['original_data']

        self._original_data[signal] = data

    def _validate_params(self, params: ProcessorParams):
        """Validates the given configuration.

        Args:
            params: Processor's configuration.

        Raises:
            ValueError: If the given config contains invalid values.
        """

        if params.target_sampling_rate <= 0:
            raise ValueError('Target sampling rate must be positive!')

        if params.filter_poly_order <= 0:
            raise ValueError('Filter polynomial order must be positive!')

        if params.filter_ripple_extent <= 0:
            raise ValueError('Filter ripple extent must be positive!')

    def _downsample_and_filter(self, data: np.ndarray, sampling_rate: int) -> np.ndarray:
        """Filters and downsamples a given signal's data..

        Args:
            data: Signal's samples.
            sampling_rate: Original sampling rate of the signal.

        Returns:
            Downsampled and filtered signal.
        """

        critical_frequency = self._params.target_sampling_rate / 2

        second_order_sections = sp_signal.cheby1(
            self._params.filter_poly_order,
            self._params.filter_ripple_extent,
            critical_frequency,
            btype='lowpass',
            fs=sampling_rate)

        filtered_signal = sp_signal.sosfilt(second_order_sections, data)

        return librosa.resample(filtered_signal, orig_sr=sampling_rate,
                                target_sr=self._params.target_sampling_rate, res_type='soxr_qq')
