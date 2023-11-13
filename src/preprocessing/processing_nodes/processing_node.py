# -*- coding: utf-8 -*-
"""Module contains declarations of interfaces for pre-processors."""
from abc import ABC
from abc import abstractmethod

from preprocessing import audio_signal


class AudioProcessingError(Exception):
    """Thrown by audio signal processors."""


class ProcessingNode(ABC):
    """Abstract class for audio signal processors.

    The class is designed to be inherited by classes performing
    various kinds of signal processing.
    """

    def __init__(self, allow_backward_processing: bool = True):
        """Initializes the common processor fields.

        Args:
            allow_backward_processing: Tells whether the processor should
            collects info needed for undoing the applied transformations.
        """

        self._allow_backward = allow_backward_processing

    def process(
            self, signal: audio_signal.AudioSignal) -> audio_signal.AudioSignal:
        """Processes a given audio signal.

        Applied transformations depend of the concrete subclass.

        Args:
            signal: Audio signal to be processed.

        Returns:
            A transformed signal.

        Raises:
            AudioProcessingError: If there processor encountered any problems
            while applying the transformations.
        """

        if self._allow_backward:
            self._store_processing_info(signal)

        return self._apply_transformations(signal)

    def process_backwards(
            self, signal: audio_signal.AudioSignal) -> audio_signal.AudioSignal:
        """Processes a given audio signal in reverse direction.

        The reverse processing shall be often possible only if the signal has been processed
        earlier and if the necessary info has been stored.

        Args:
            signal: Audio signal to be processed.

        Returns:
            A reversely transformed signal.

        Raises:
            AudioProcessingError: If there were encountered any problems while applying the
            transformation or if the stored after-processing info is corrupted.
        """

        if not self._allow_backward:
            raise AudioProcessingError(
                f'Reverse processing in this instance of {self.signature} not supported!')

        return self._transform_backwards(signal)

    @property
    @abstractmethod
    def signature(self) -> str:
        """Returns a string signature of the concrete instance of the abstract class."""

    @abstractmethod
    def _apply_transformations(
            self, signal: audio_signal.AudioSignal) -> audio_signal.AudioSignal:
        """Applies transformations to an audio signal.

        Args:
            signal: Audio signal to be processed.

        Returns:
            Processed signal.

        Raises:
            AudioProcessingError: If there were encountered any problems.
        """

    @abstractmethod
    def _transform_backwards(
            self, signal: audio_signal.AudioSignal) -> audio_signal.AudioSignal:
        """Applies reverse transformations to an audio signal.

        Args:
            signal: Audio signal to be processed.

        Returns:
            The signal transformed to its form before the processing.

        Raises:
            AudioProcessingError: If there were encountered any problems.
        """

    @abstractmethod
    def _store_processing_info(self, signal: audio_signal.AudioSignal):
        """Stores necessary info needed to perform a backward processing.

        Args:
            signal: Audio signal to store the associated info for.
        """
