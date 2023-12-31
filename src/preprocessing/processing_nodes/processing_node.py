# -*- coding: utf-8 -*-
"""Module contains declarations of interfaces for pre-processors."""
from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Dict

from preprocessing import audio_signal


class AudioProcessingError(Exception):
    """Thrown by audio signal processors."""


class ProcessingNode(ABC):
    """Abstract class for audio signal processors.

    The class is designed to be inherited by classes performing
    various kinds of signal processing.
    """

    def __init__(self, allow_backward_processing: bool):
        """Initializes the common processor fields.

        Args:
            allow_backward_processing: Tells whether the processor should
            collects info needed for undoing the applied transformations.
        """

        self._allow_backward = allow_backward_processing

    @classmethod
    @abstractmethod
    def from_config(cls, config: Dict[str, Any]) -> 'ProcessingNode':
        """Spawns a processing node from provided configuration.

        Interpretation of the configuration depends on the concrete subclass
        and the validation is implementation-defined.

        Args:
            config: Configuration dictionary.
                See: scripts/config/preprocessing_config.yaml for specification
                of allowed fields for concrete classes.

        Returns:
            Spawned processor with applied config.
        """

    @abstractmethod
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
            self, signal: audio_signal.AudioSignal, *args, **kwargs) -> audio_signal.AudioSignal:
        """Applies transformations to an audio signal.

        Args:
            signal: Audio signal to be processed.
            *args: Variable length arguments, specific to a concrete processor.
            **kwargs: Keyword arguments, specific to a concrete processor.

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
    def _store_processing_info(
            self, signal: audio_signal.AudioSignal, *args, **kwargs):
        """Stores necessary info needed to perform a backward processing.

        Args:
            signal: Audio signal to store the associated info for.
            *args: Variable length arguments, specific to a concrete processor.
            **kwargs: Keyword arguments, specific to a concrete processor.
        """
