"""Contains functions for reading audio signal files."""

import os
import wave
import dataclasses
from typing import Optional

import librosa

from preprocessing import audio_signal


class IncorrectAudioFileError(Exception):
    """File is not an audio signal container."""


@dataclasses.dataclass
class ReaderParams:
    """Contains configuration for SignalReader."""

    # If set, the reader will resample decoded signals to the given sampling rate.
    dest_sampling_rate: Optional[int] = None


class SignalReader:
    """Reads audio signal container files.

    The form of the result file depends on the internal reader's configuration.
    """

    def __init__(self, params: ReaderParams):
        """Initializes the reader with given parameters.

        Args:
            params: Reader's configuration.
        """

        self._validate_params(params)

        self._params = params

    def read(self, file_path: str) -> audio_signal.AudioSignal:
        """Reads a given file and decodes its contents.

        The used decoding tool depends on the container-file type and the
        codec used inside the container.

        Args:
            file_path: Path to the file to decode.

        Returns:
            Decoded file in AudioSignal format.

        Raises:
            FileNotFoundError: If the given file does not exist.
            IncorrectAudioFileError: If the given file is either invalid or
                not supported audio signal container.
        """

        if not os.path.exists(file_path) or not os.path.isfile(file_path):
            raise FileNotFoundError(f"File does not exist: '{file_path}'")

        _, file_extension = os.path.splitext(file_path)

        match file_extension:
            case ".wav" | ".wave":
                return self._read_wave(file_path)

        raise IncorrectAudioFileError(f"Unsupported file type: {file_extension}!")

    def _validate_params(self, params: ReaderParams):
        """Checks if the passed parameters are valid.

        Args:
            params: Parameters to check.

        Raises:
            ValueError: If the parameters are invalid.
        """

        if params.dest_sampling_rate is not None and params.dest_sampling_rate <= 0:
            raise ValueError("Destination sampling rate must be positive!")

    def _read_wave(self, file_path: str) -> audio_signal.AudioSignal:
        """Reads a file identified as a wave container."""

        with wave.open(file_path, "rb") as input_binary:

            meta_data = audio_signal.AudioMeta(input_binary.getframerate(),
                                               input_binary.getnchannels(),
                                               input_binary.getsampwidth() * 8)

            data, _ = librosa.load(file_path, sr=meta_data.sampling_rate)

            if data.ndim == 1:
                data = data.reshape((1, -1))

            return audio_signal.AudioSignal(data, meta_data)
