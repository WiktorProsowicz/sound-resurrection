# -*- coding: utf-8 -*-
"""Contains definition of class serializing audio signals."""
import dataclasses
import enum
import os
from typing import Optional

import numpy as np
import soundfile as sf

from preprocessing import audio_signal


class WriterFileExtension(enum.Enum):
    """Contains supported audio file extensions."""

    WAVE = 'wav'


@dataclasses.dataclass()
class WriterParams:
    """Contains internal SignalWriter's configuration."""

    # If true, writer shall generate a new file name for each signal.
    generate_names: bool
    overwrite_existing: bool
    file_extension: WriterFileExtension
    # Specifies the pattern for generated file names, if supported.
    output_name_pattern: Optional[str] = None


class SignalWriter:
    """Writes audio signals to audio container files.

    The form of the result file depends on the internal writer's configuration.
    """

    _DEFAULT_OUTPUT_NAME_PATTERN = 'audio_signal'

    def __init__(self, params: WriterParams):
        """Initializes the writer with given parameters.

        Args:
            params: Writer parameters.
        """

        self._params = params
        self._generated_names_count = 0

    def write(self, signal: audio_signal.AudioSignal, destination_dir: str, **kwargs):
        """Writes a given audio signal to a file.

        Args:
            signal: Audio signal to write.
            destination_dir: Directory in which the file shall be created.
            **kwargs: See below

            Keyword arguments:
                file_name (str): If given, the file will be created under this name and
                    a new one won't be generated.

        Raises:
            FIleExistsError: If the file already exists and overwriting mode is not set.
            ValueError: If the writer cannot find or create a name for the output file.
        """

        if 'file_name' not in kwargs:
            if not self._params.generate_names:
                raise ValueError('File name was not provided and generation is disabled!')

            file_name = self._generate_file_name(destination_dir)

        else:
            file_name = f"{kwargs['file_name']}.{self._params.file_extension.value}"

        file_path = os.path.join(destination_dir, file_name)

        if not self._params.overwrite_existing and os.path.exists(file_path):
            raise FileExistsError(f"File already exists: '{file_path}'")

        match self._params.file_extension:
            case WriterFileExtension.WAVE:
                self._write_wave(signal, file_path)
                return

        assert False, 'Unsupported file extension!'

    def _generate_file_name(self, destination_dir: str) -> str:
        """Generates a file name for a given signal.

        Generated name depends on the internal written files count, the presence of
        files in `destination_dir` and the internal configuration.

        Args:
            signal: Signal to generate a name for.
            destination_dir: Directory in which the file will be created.

        Returns:
            Generated file name.
        """

        if self._params.output_name_pattern is not None:
            chosen_pattern = self._params.output_name_pattern

        else:
            chosen_pattern = SignalWriter._DEFAULT_OUTPUT_NAME_PATTERN

        while True:
            file_name = (f'{chosen_pattern}_{self._generated_names_count}' +
                         f'.{self._params.file_extension.value}')
            file_path = os.path.join(destination_dir, file_name)

            self._generated_names_count += 1

            if not os.path.exists(file_path):
                return file_name

    def _write_wave(self, signal: audio_signal.AudioSignal, file_path: str):
        """Writes a given signal to a wave container file.

        Args:
            signal: Audio signal to write.
            file_path: Path to the file to create.
        """

        # pylint: disable=no-member

        sample_width = signal.meta.bits_per_sample // 8
        if sample_width * 8 < signal.meta.bits_per_sample:
            sample_width += 1

        data = np.swapaxes(signal.data, 0, 1)

        if f'PCM_{sample_width * 8}' not in sf.available_subtypes('WAV'):
            assert False, f'Unsupported data type size {sample_width} for WAV format!'

        sf.write(file_path, data, signal.meta.sampling_rate, f'PCM_{sample_width * 8}')

        # pylint: enable=no-member
