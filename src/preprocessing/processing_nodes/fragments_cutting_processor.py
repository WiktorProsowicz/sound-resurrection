# -*- coding: utf-8 -*-
"""Contains definition of a class performing random cuts in audio signals."""

from typing import Dict, Tuple, List

import numpy as np

from preprocessing import audio_signal
from preprocessing.processing_nodes import processing_node


class FragmentsCuttingProcessor(processing_node.ProcessingNode):
    """Cuts portions of audio signals.

    The processor is used to create imitation of connection / recording flaws.
    """

    _N_CUT_RANGES_CLASSES = 10

    def __init__(self, allow_backward_processing: bool, cut_duration: float):
        """
        Initializes the base class and sets internal processor's config.

        Args:
            allow_backward_processing: Whether to allow reversal of the cut operation.
            cut_duration: Cumulative duration of the cut in seconds.
        """
        super().__init__(allow_backward_processing)
        self._cut_duration = cut_duration
        self._cut_ranges: Dict[audio_signal.AudioSignal, List[Tuple[int, np.ndarray]]] = {}

    @property
    def signature(self) -> str:
        """Overrides method of ProcessingNode class."""

        return f"CutSoundProcessor(duration={self._cut_duration})"

    def process(self, signal: audio_signal.AudioSignal) -> audio_signal.AudioSignal:
        """Overrides method of ProcessingNode class."""

        cut_data = self._random_cut(signal.data, signal.meta.sampling_rate)

        transformed = self._apply_transformations(signal, cut_ranges=cut_data)

        if self._allow_backward:
            self._store_processing_info(transformed, cut_ranges=cut_data)

        return transformed

    def _apply_transformations(self, signal: audio_signal.AudioSignal,
                               *args, **kwargs) -> audio_signal.AudioSignal:
        """Overrides method of ProcessingNode class.

        Args:
            **kwargs: Listed below

            Keyword arguments:
                cut_ranges: A list of cut data ranges to extract from the signal.
        """

        copied_data = signal.data.copy()

        for idx, cut_range in kwargs["cut_ranges"]:
            copied_data.data[:, idx:idx + len(cut_range)].fill(0)

        return audio_signal.AudioSignal(copied_data, signal.meta)

    def _transform_backwards(self, signal: audio_signal.AudioSignal) -> audio_signal.AudioSignal:
        """Overrides method of ProcessingNode class."""

        if signal in self._cut_ranges:
            copied_data = signal.data.copy()

            for idx, cut_range in self._cut_ranges[signal]:
                copied_data.data[:, idx:idx + len(cut_range)] = cut_range

            return audio_signal.AudioSignal(copied_data, signal.meta)

        raise processing_node.AudioProcessingError(
            "Backward processing for the given signal is not available!")

    def _store_processing_info(self, signal: audio_signal.AudioSignal, *args, **kwargs):
        """Overrides method of ProcessingNode class.

        Args:
            **kwargs: Listed below

            Keyword arguments:
                cut_ranges: A list of cut data ranges extracted from the signal.
        """

        self._cut_ranges[signal] = kwargs["cut_ranges"]

    def _random_cut(self, data: np.ndarray, sampling_rate: int) -> List[Tuple[int, np.ndarray]]:
        """Performs a random cut on the input audio data.

        The extracted data ranges sum up to the cumulative duration
        specified in the processor's config.

        Args:
            data: Signal's audio data to cut.
            sampling_rate: Sampling rate of the audio data.

        Returns:
            List containing positions of extracted data ranges and the ranges themselves.
        """

        cut_samples = int(self._cut_duration * sampling_rate)

        max_samples = data.shape[1]

        if max_samples < cut_samples:
            raise processing_node.AudioProcessingError(
                ("Requested to cut a number of samples that is bigger " +
                 "than the cumulative number of samples in the signal!"))

        idx = np.random.randint(0, max_samples - cut_samples + 1)

        return [(idx, data[:, idx:idx + cut_samples])]

        # TODO: Implement algorithm that would cut multiple data ranges, not just one.
