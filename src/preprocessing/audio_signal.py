# -*- coding: utf-8 -*-
"""Module contains definition of class representing a single audio signal."""
import dataclasses

import numpy as np


@dataclasses.dataclass(eq=True)
class AudioMeta:
    """Contains information about an audio signal."""

    sampling_rate: int
    channels: int
    bits_per_sample: int


class AudioSignal:
    """Represents a single audio sample in time domain.

    AudioSample contains an array of the signal's data and a meta-data struct. It
    works as a plain-old-data container for audio signals, but provides additional
    validation to the passed data.

    Attributes:
        data: A numpy array containing the signal's samples.
        meta: A structure containing the info about the signal instance.
    """

    def __init__(self, data: np.ndarray, meta: AudioMeta):
        """Initializes and validates both the signal's data and meta data.

        Args:
            data: Signal's samples.
            meta: Signal description.
        """

        self._validate_data(data, meta)

        self._data = data
        self._meta = meta

    @property
    def data(self) -> np.ndarray:
        """Returns the read-only signal's data."""
        return self._data

    @property
    def meta(self) -> AudioMeta:
        """Returns the read-only signal's meta data."""
        return self._meta

    @property
    def length(self) -> float:
        """Returns the length of the signal in seconds."""
        return self._data.shape[1] / self._meta.sampling_rate

    def _validate_data(self, data: np.ndarray, meta: AudioMeta):
        """Validates the provided signal's data.

        Tells whether the provided data is compatible with the data description.

        Args:
            data: Signal's samples.
            meta: Signal description.

        Raises:
            ValueError: If the provided data is invalid.
        """

        if data.ndim != 2:
            raise ValueError(
                'The signal samples should be of the shape (channels, samples)!')

        if data.shape[0] != meta.channels:
            raise ValueError(
                'The number of signal channels does not equal the declared one!')

        if data.itemsize * 8 < meta.bits_per_sample:
            raise ValueError(
                "The declared number of bits per sample is higher than the data's  \
                underlying type's size!")

        if meta.sampling_rate <= 0:
            raise ValueError('The sampling rate should be positive!')
