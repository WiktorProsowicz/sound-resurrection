from typing import Dict

import numpy as np
import librosa
from scipy.signal import cheby1, lfilter
from preprocessing import audio_signal
from preprocessing.processing_nodes import processing_node


class DownsamplingProcessor(processing_node.ProcessingNode):
    """
    Downsamples the signal to a lower sampling rate.
    """

    def __init__(self, allow_backward_processing: bool, 
                 filter_order: int, 
                 filter_rp: float,
                 critical_freq: float,
                 downsampling_ratio: float):
        
        """
        Initializes the base class and sets internal processor's config.

        Args:
            allow_backward_processing: Whether to allow reversal of the cut operation.
            filter_order: Order of the Chebyshev filter.
            filter_rp: Passband ripple of the Chebyshev filter.
            critical_freq: Critical frequency of the Chebyshev filter.
        """
        super().__init__(allow_backward_processing)
        self._filter_order = filter_order
        self._filter_rp = filter_rp
        self._critical_freq = critical_freq
        self._original_data = Dict[audio_signal.AudioSignal, np.ndarray] = {}
        self._downsampling_ratio = downsampling_ratio


    @property
    def signature(self) -> str:
        """Overrides method of ProcessingNode class."""

        return f'DownsamplingProcessor(filter_order={self._filter_order}, filter_rp={self._filter_rp}, critical_freq={self._critical_freq})'
    
    def process(self, signal: audio_signal.AudioSignal) -> audio_signal.AudioSignal:
        """Overrides method of ProcessingNode class."""

        transformed = self._apply_transformations(signal)

        if self._allow_backward:
            self._store_processing_info(transformed, original_data=signal.data.copy())

        return transformed
    
    def _apply_transformations(self, signal: audio_signal.AudioSignal) -> audio_signal.AudioSignal:
        """Overrides method of ProcessingNode class."""
        downsampled_data = self._downsample(signal.data, signal.meta.sampling_rate)

        return audio_signal.AudioSignal(downsampled_data, signal.meta)
    
    def _transform_backwards(self, signal: audio_signal.AudioSignal) -> audio_signal.AudioSignal:
        """Overrides method of ProcessingNode class."""

        if signal in self._original_data:
            return audio_signal.AudioSignal(self._original_data[signal], signal.meta)

        raise processing_node.AudioProcessingError(
            f'No stored info for {self.signature}!')


    def _store_processing_info(self, signal: audio_signal.AudioSignal, *args, **kwargs):
        """
        Overrides method of ProcessingNode class.
        """

        data = kwargs["original_data"]

        self._original_data[signal] = data

    def _downsample(self, data: np.ndarray, sampling_rate: int) -> np.ndarray:
        """
        Downsamples the signal to a lower sampling rate.


        Args:
            data: Signal's samples.
            sampling_rate: Sampling rate of the signal.

        Returns:
            Downsampled signal.
        """
        target_sampling = self._downsampling_ratio * sampling_rate

        b, a = cheby1(self._filter_order, self._filter_rp, self._critical_freq, btype='lowpass', fs=sampling_rate)
        return librosa.resample(lfilter(b, a, data, axis=1), sampling_rate, target_sampling)