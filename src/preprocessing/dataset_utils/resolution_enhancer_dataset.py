# -*- coding: utf-8 -*-
"""Module contains a helper class generating datasets for the resolution enhancer model."""
import dataclasses
import os
from typing import Tuple

import keras
import tensorflow as tf


@dataclasses.dataclass
class GeneratorParameters:
    """Parameters for the dataset generator."""
    high_quality_path: str
    low_quality_path: str
    inputs_length: int
    targets_length: int
    batch_size: int


class ResolutionEnhancerDatasetGenerator:
    """Generates training and testing datasets for the resolution enhancer model."""

    def __init__(self, params: GeneratorParameters):
        """Initializes the dataset generator with directories for high and low-quality audio files.
        Validates the existence of these directories before proceeding.

        Args:
            high_quality_path: Path to the directory with high-quality WAV files.
            low_quality_path: Path to the directory with low-quality WAV files.

        Raises:
            FileNotFoundError: If either directory is non-existent or empty.
        """

        self._validate_directory(params.high_quality_path)
        self._validate_directory(params.low_quality_path)

        self._params = params

        self._check_dataset_compatibility()

    def generate_dataset(self, train_size) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """Generates training and testing datasets from the high and low-quality audio files.

        Args:
            test_size: Fraction of the data to be used as the test set (default is 20%).

        Returns:
            A tuple of two tf.data.Dataset objects: (training dataset, testing dataset).
        """

        high_quality_dataset = self._load_wav_dataset(
            self._params.high_quality_path, self._params.targets_length)

        low_quality_dataset = self._load_wav_dataset(
            self._params.low_quality_path, self._params.inputs_length)

        dataset = tf.data.Dataset.zip((low_quality_dataset, high_quality_dataset))
        dataset = dataset.shuffle(32).batch(self._params.batch_size)

        train_items = int(train_size * len(dataset))

        return dataset.take(train_items), dataset.skip(train_items)

    def _check_dataset_compatibility(self):
        """Checks if the file sets in high and low-quality directories are compatible.

        Raises:
            ValueError: If the file lists in the directories are not identical.

        """
        high_quality_files = set(os.listdir(self._params.high_quality_path))
        low_quality_files = set(os.listdir(self._params.low_quality_path))

        if high_quality_files != low_quality_files:
            raise ValueError('Files in high and low-quality directories must match.')

    def _validate_directory(self, directory: str) -> None:
        """Checks if the specified directory exists and contains files.

        Args:
            directory: Directory path to validate.

        Raises:
            FileNotFoundError: If the directory is either empty or doesn't exist.
        """

        if not os.path.isdir(directory) or not os.listdir(directory):
            raise FileNotFoundError(f"No files found in '{directory}'!")

    def _decode_wav(self, file_path: tf.Tensor, n_samples: int) -> tf.Tensor:
        """Decodes a WAV file into a TensorFlow tensor.

        Args:
            file_path: A tf.Tensor containing the path to the WAV file.
            n_samples: Number of samples to load from the file.

        Returns:
            A 1-D tensor representing the audio signal from the WAV file.
        """
        audio_binary = tf.io.read_file(file_path)
        audio, _ = tf.audio.decode_wav(audio_binary, desired_samples=n_samples, desired_channels=1)
        return tf.reshape(audio, [-1, 1])

    def _load_wav_dataset(self, directory: str, n_samples: int) -> tf.data.Dataset:
        """Loads all WAV files from the specified directory into a TensorFlow dataset.

        Args:
            directory: Directory containing WAV files.
            n_samples: Number of samples to load from each file.

        Returns:
            A tf.data.Dataset object where each item is a decoded WAV file tensor.
        """

        files = tf.data.Dataset.list_files(os.path.join(directory, '*.wav'), shuffle=False)

        return files.map(lambda path: self._decode_wav(path, n_samples))
