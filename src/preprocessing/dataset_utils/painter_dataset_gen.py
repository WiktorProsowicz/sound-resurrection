"""Contains definition of class providing dataset for training the SoundPainter model."""

import dataclasses
import os
from typing import Tuple, Callable, Any

import tensorflow as tf
import librosa
import numpy as np


@dataclasses.dataclass
class GeneratorParams:
    """Contains parameters for SoundPainterDatasetGenerator"""

    defective_path: str
    normal_path: str
    samples_per_file: int
    batch_size: int


class SoundPainterDatasetGenerator:
    """Generates training and testing datasets for SoundPainter model."""

    def __init__(self, params: GeneratorParams):
        """Initializes the generator with paths to the datasets.

        Args:
            params: Parameters of the generator.
        """

        self._validate_params(params)

        self._params = params

    def generate(self, train_size) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """Generates training and testing datasets.

        Args:
            train_size: Size of the training dataset in range (0, 1).

        Returns:
            Tuple containing training and testing datasets.
        """

        generator_input = self._map_directory(
            self._params.defective_path, self._read_generator_input)

        real_spectrograms = self._map_directory(
            self._params.normal_path, self._read_spectrogram_with_phase)

        dataset = tf.data.Dataset.zip((generator_input, real_spectrograms))

        dataset = dataset.shuffle(32).batch(self._params.batch_size)

        train_items = int(train_size * len(dataset))

        return dataset.take(train_items), dataset.skip(train_items)

    def _read_generator_input(self, path: str) -> Tuple[tf.Tensor, tf.Tensor]:
        """Reads a wave file and returns MEL spectrogram and phase coefficients.

        Args:
            path: Path to the file.

        Returns:
            Tuple with spectrogram and phase to be used as input for the generator.
        """

        signal, sr = librosa.load(path, sr=None, mono=True)
        signal = signal[:self._params.samples_per_file]

        stft = librosa.stft(signal, n_fft=1024, hop_length=512)

        spectrogram = tf.abs(stft)
        mel_spec = librosa.feature.melspectrogram(S=spectrogram, sr=sr)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        phase = tf.math.angle(stft)

        return tf.expand_dims(mel_spec_db, axis=2), tf.expand_dims(phase, axis=2)

    def _read_spectrogram_with_phase(self, path: str) -> tf.Tensor:
        """Reads a wave file and returns its spectrogram with phase coefficients.

        Args:
            path: Path to the file.

        Returns:
            Tensor in shape (frequency, time, 2) containing MEL spectrogram
            with phase coefficients.
        """

        spec, phase = self._read_generator_input(path)

        return tf.stack((spec, phase), axis=2)

    def _map_directory(self, path: str, mapping_func: Callable[[str], Any]) -> tf.data.Dataset:
        """Maps a directory containing wave files to a dataset.

        Args:
            path: Path to the directory.

        Returns:
            Dataset containing spectrograms from the directory.
        """

        files = tf.data.Dataset.list_files(os.path.join(path, '*.wav'), shuffle=False)

        return files.map(mapping_func, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    def _validate_params(self, params: GeneratorParams):
        """Validates the parameters of the generator.

        Args:
            params: Parameters of the generator.

        Raises:
            ValueError: If paths given in params are incompatible.
            FileNotFoundError: If directories given in params do not exist or are empty.
        """

        if not os.path.isdir(params.defective_path) or not os.listdir(params.defective_path):
            raise FileNotFoundError(f"No files found in '{params.defective_path}'!")

        if not os.path.isdir(params.defective_path) or not os.listdir(params.defective_path):
            raise FileNotFoundError(f"No files found in '{params.defective_path}'!")

        if os.listdir(params.defective_path) != os.listdir(params.normal_path):
            raise ValueError("Input directories must contain the same files!")
