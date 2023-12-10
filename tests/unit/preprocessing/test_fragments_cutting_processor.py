# -*- coding: utf-8 -*-

import pytest
import numpy as np

from preprocessing import audio_signal
from preprocessing.processing_nodes import fragments_cutting_processor


@pytest.mark.parametrize('cut_part', [-.1, 1.1,])
@pytest.mark.xfail(reason='Cut part value must be in range [0, 1]!', raises=ValueError)
def test_validates_wrong_cut_part(cut_part):
    """Test if processor detects ill-formed cur part."""

    fragments_cutting_processor.FragmentsCuttingProcessor(False, cut_part)


@pytest.fixture
def sample_processor():
    """Sample processor with cut part equal to 0.5."""

    return fragments_cutting_processor.FragmentsCuttingProcessor(False, .5)


# pylint: disable=redefined-outer-name
def test_preserves_meta_data(sample_processor, sample_audio_signal):
    """Test if processor preserves meta data of audio signal."""

    processed_signal = sample_processor.process(sample_audio_signal)

    meta: audio_signal.AudioMeta = processed_signal.meta
    exp_meta: audio_signal.AudioMeta = sample_audio_signal.meta

    assert meta.sampling_rate == exp_meta.sampling_rate
    assert meta.channels == exp_meta.channels
    assert meta.bits_per_sample == exp_meta.bits_per_sample


@pytest.fixture
def sample_audio_signal():
    """Sample audio signal with one channel, 44100 sampling rate, 1 s duration."""

    meta_data = audio_signal.AudioMeta(44100, 2, 16)
    return audio_signal.AudioSignal(np.ones((2, 44100)), meta_data)


@pytest.mark.parametrize('cut_part', [.1, .2, .5, .9])
# pylint: disable=redefined-outer-name
def test_processor_cuts_at_most_given_fragment(sample_audio_signal, cut_part):
    """Test if processor cuts at most given fragment from audio signal."""

    processor = fragments_cutting_processor.FragmentsCuttingProcessor(False, cut_part)
    processed_signal = processor.process(sample_audio_signal)

    assert processed_signal.data.size <= sample_audio_signal.data.size

    data_diff = sample_audio_signal.data.size - processed_signal.data.size

    assert np.count_nonzero(data_diff) <= (cut_part * sample_audio_signal.data.size)
