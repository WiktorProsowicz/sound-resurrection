# -*- coding: utf-8 -*-
import numpy as np
import pytest

from preprocessing import audio_signal
from preprocessing.processing_nodes import downsampling_processor


@pytest.mark.xfail(raises=ValueError)
@pytest.mark.parametrize('target_sr,poly_order,ripple', [
    (-100, 2, 1.5), (0, 2, 1.5), (100, -1, 1.5), (100, 0, 1.5), (100, 2, -1.5), (100, 2, 0)])
def test_detects_illformed_params(target_sr, poly_order, ripple):
    """Tests that the processor detects ill-formed parameters."""

    params = downsampling_processor.ProcessorParams(target_sr, poly_order, ripple)
    downsampling_processor.DownSamplingProcessor(False, params)


@pytest.fixture
def sample_audio_signal():
    """A sample audio signal with 1 channel and 44100 sampling rate."""

    meta = audio_signal.AudioMeta(44100, 1, 16)
    return audio_signal.AudioSignal(np.ones((1, 44100)), meta)


# pylint: disable=redefined-outer-name
def test_resamples_signal(sample_audio_signal):
    """Tests that the processor correctly resamples the signal."""

    target_sr = 22050
    params = downsampling_processor.ProcessorParams(target_sr, 2, 1.5)
    processor = downsampling_processor.DownSamplingProcessor(False, params)

    processed_signal = processor.process(sample_audio_signal)

    assert processed_signal.meta.sampling_rate == target_sr
    assert processed_signal.meta.channels == 1
    assert processed_signal.meta.bits_per_sample == 16

    assert processed_signal.data.shape == (1, 22050)
