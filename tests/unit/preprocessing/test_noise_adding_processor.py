# -*- coding: utf-8 -*-
import numpy as np
import pytest

from preprocessing import audio_signal
from preprocessing.processing_nodes import noise_adding_processor


@pytest.fixture
def sample_audio_signal():
    """A sample audio signal with 1 channel and 44100 sampling rate."""

    meta = audio_signal.AudioMeta(44100, 1, 16)
    return audio_signal.AudioSignal(np.ones((1, 44100)), meta)


@pytest.mark.xfail(raises=ValueError, reason='Standard deviation must be positive!')
@pytest.mark.parametrize('noise_std', [-100, 0])
def test_awg_noise_provider_detects_illformed_params(noise_std):
    """Tests that the noise provider detects ill-formed parameters."""

    noise_adding_processor.AWGNoiseProvider(noise_std)


@pytest.mark.xfail(raises=ValueError, reason='At least one noise provider must be given!')
def test_detects_no_noise_providers():
    """Tests that the processor detects lack of noise providers."""

    noise_adding_processor.NoiseAddingProcessor(False, [])


@pytest.mark.parametrize('noise_providers', [[noise_adding_processor.AWGNoiseProvider(.5)]])
# pylint: disable=redefined-outer-name
def test_preserves_meta_data(sample_audio_signal, noise_providers):
    """Tests that the processor preserves meta data."""

    processor = noise_adding_processor.NoiseAddingProcessor(False, noise_providers)

    processed_signal = processor.process(sample_audio_signal)

    meta: audio_signal.AudioMeta = processed_signal.meta
    exp_meta = sample_audio_signal.meta

    assert meta.sampling_rate == exp_meta.sampling_rate
    assert meta.channels == exp_meta.channels
    assert meta.bits_per_sample == exp_meta.bits_per_sample

    assert processed_signal.data.shape == sample_audio_signal.data.shape
