# -*- coding: utf-8 -*-
"""Contains definition of a class performing random cuts in audio signals."""
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

import numpy as np

from preprocessing import audio_signal
from preprocessing.processing_nodes import processing_node


class FragmentsCuttingProcessor(processing_node.ProcessingNode):
    """Cuts portions of audio signals.

    The processor is used to create imitation of connection / recording flaws.
    Cut part of the signal may not be equal to the one specified as it is currently
    only the expected sum of generated cut ranges. The ranges themselves can overlap.
    """

    _N_CUT_RANGES_CLASSES = 10  # Number of possible lengths of generated cut ranges.
    _MAX_CUT_RANGE_PART = .3  # Maximum contribution of a single cut range to the whole cut.

    def __init__(self, allow_backward_processing: bool, cut_part: float):
        """
        Initializes the base class and sets internal processor's config.

        Args:
            allow_backward_processing: Whether to allow reversal of the cut operation.
            cut_part: Part of the signal that shall be sum of determined cut ranges.
                Expected value range is [0, 1].

        Raises:
            ValueError: If the given cut part value is not in allowed range.
        """

        if 0.0 > cut_part or cut_part > 1.0:
            raise ValueError('Cut part value must be in range [0, 1]!')

        super().__init__(allow_backward_processing)
        self._cut_part = cut_part
        self._cut_ranges: Dict[audio_signal.AudioSignal, List[Tuple[int, np.ndarray]]] = {}

    @property
    def signature(self) -> str:
        """Overrides method of ProcessingNode class."""

        return f'CutSoundProcessor(cut_part={self._cut_part})'

    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        """Overrides method of ProcessingNode class."""

        return cls(config['allow_backward_processing'], config['cut_part'])

    def process(self, signal: audio_signal.AudioSignal) -> audio_signal.AudioSignal:
        """Overrides method of ProcessingNode class."""

        cut_data = self._random_cut(signal.data)

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

        for idx, cut_range in kwargs['cut_ranges']:
            copied_data[:, idx:idx + cut_range.shape[1]].fill(0)

        return audio_signal.AudioSignal(copied_data, signal.meta)

    def _transform_backwards(self, signal: audio_signal.AudioSignal) -> audio_signal.AudioSignal:
        """Overrides method of ProcessingNode class."""

        if signal in self._cut_ranges:
            copied_data = signal.data.copy()

            for idx, cut_range in self._cut_ranges[signal]:
                copied_data.data[:, idx:idx + len(cut_range)] = cut_range

            return audio_signal.AudioSignal(copied_data, signal.meta)

        raise processing_node.AudioProcessingError(
            'Backward processing for the given signal is not available!')

    def _store_processing_info(self, signal: audio_signal.AudioSignal, *args, **kwargs):
        """Overrides method of ProcessingNode class.

        Args:
            **kwargs: Listed below

            Keyword arguments:
                cut_ranges: A list of cut data ranges extracted from the signal.
        """

        self._cut_ranges[signal] = kwargs['cut_ranges']

    def _random_cut(self, data: np.ndarray) -> List[Tuple[int, np.ndarray]]:
        """Performs a random cut on the input audio data.

        The extracted data ranges sum up to the cumulative duration
        specified in the processor's config.

        Args:
            data: Signal's audio data to cut.
            sampling_rate: Sampling rate of the audio data.

        Returns:
            List containing positions of extracted data ranges and the ranges themselves.
        """

        max_samples = data.shape[1]

        cut_samples = int(self._cut_part * max_samples)

        cut_ranges_counts = self._generate_cut_ranges_counts(cut_samples)

        generated_ranges: List[Tuple[int, np.ndarray]] = []

        for range_length, range_count in cut_ranges_counts.items():
            for _ in range(range_count):
                pos = np.random.randint(0, max_samples - range_length + 1)
                generated_ranges.append((pos, data[:, pos:pos + range_length]))

        return generated_ranges

    def _generate_cut_ranges_counts(self, cut_samples: int) -> Dict[int, int]:
        """Generates a dictionary of cut ranges lengths and their counts.

        Cumulative length of the generated ranges shall be equal to the
        number of samples specified by the cut part specified for the processor.

        Args:
            cut_samples: Number of samples to cut.

        Returns:
            Dictionary containing cut ranges lengths and their counts.
        """

        max_cut_length = int(self._MAX_CUT_RANGE_PART * cut_samples)

        cut_ranges_lengths = np.random.choice(np.arange(2, max_cut_length),
                                              self._N_CUT_RANGES_CLASSES - 1)

        cut_ranges_lengths = np.concatenate([cut_ranges_lengths, np.array([1])])
        cut_ranges_lengths = sorted(cut_ranges_lengths, reverse=True)

        cut_ranges_counts = {length: 0 for length in cut_ranges_lengths}

        for range_length in cut_ranges_lengths:
            while sum(length * cnt for length, cnt in cut_ranges_counts.items()) < cut_samples:
                cut_ranges_counts[range_length] += 1

        return cut_ranges_counts
