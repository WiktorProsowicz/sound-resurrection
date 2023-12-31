# -*- coding: utf-8 -*-
import tempfile

import numpy as np
import pytest

from preprocessing import audio_signal
from preprocessing.processing_nodes import processing_node
from preprocessing.processing_nodes import resampling_processor


@pytest.fixture
def sample_signal():
    """Returns a sample audio signal."""

    return audio_signal.AudioSignal(
        data=np.random.rand(1, 8000),
        meta=audio_signal.AudioMeta(sampling_rate=8000, channels=1, bits_per_sample=16))


@pytest.fixture
def temp_output_dir():
    """Returns a temporary directory to write to."""

    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.mark.xfail(raises=processing_node.AudioProcessingError,
                   reason='No info about original sampling rate for the given signal!')
# pylint: disable=redefined-outer-name
def test_indicates_backward_processing_failure(sample_signal):
    """Tests whether the processor indicates failure of the backward processing."""

    processor = resampling_processor.ResamplingProcessor(
        allow_backward_processing=True, sampling_rate=16000)

    processor.process_backwards(sample_signal)


@pytest.mark.parametrize('target_sr', [8000, 4000, 2000, 1000])
# pylint: disable=redefined-outer-name
def test_properly_changes_sampling_rate(sample_signal, target_sr):
    """Tests whether the processor properly transforms a given signal."""

    processor = resampling_processor.ResamplingProcessor(
        allow_backward_processing=False, sampling_rate=target_sr)

    transformed_signal = processor.process(sample_signal)

    ratio = sample_signal.meta.sampling_rate // target_sr

    assert transformed_signal.meta.sampling_rate == target_sr
    assert transformed_signal.data.shape[1] == sample_signal.data.shape[1] // ratio
    assert transformed_signal.length == sample_signal.length
