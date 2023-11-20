# -*- coding: utf-8 -*-
import numpy as np
import pytest
from preprocessing import audio_signal

VALID_AUDIO_RESOURCES = [
    (audio_signal.AudioMeta(200, 2, 24), np.random.rand(2, 450).astype(np.uint32)),
    (audio_signal.AudioMeta(4410, 1, 7), np.random.rand(1, 400).astype(np.uint8)),
    (audio_signal.AudioMeta(230, 6, 56), np.random.rand(6, 2222).astype(np.uint64)),
    (audio_signal.AudioMeta(111, 5, 20), np.random.rand(5, 20).astype(np.uint32))
]


@pytest.mark.dependency(name="test_making_a_valid_signal")
@pytest.mark.parametrize("meta_data,samples", VALID_AUDIO_RESOURCES)
def test_making_a_valid_signal(meta_data, samples):

    signal = audio_signal.AudioSignal(samples, meta_data)

    assert np.array_equal(signal.data, samples)
    assert signal.meta == meta_data


@pytest.mark.xfail(raises=ValueError)
def test_making_audio_signal_with_wrong_dimensionality():

    meta_data = audio_signal.AudioMeta(100, 1, 16)
    data = np.ndarray(shape=(2, 3, 4), dtype=np.uint16)

    audio_signal.AudioSignal(data, meta_data)


@pytest.mark.dependency(depends=["test_making_a_valid_signal"])
def test_length_property_of_audio_signal():

    meta_data = audio_signal.AudioMeta(100, 1, 8)
    data = np.ndarray(shape=(1, 200), dtype=np.uint8)

    signal = audio_signal.AudioSignal(data, meta_data)

    assert signal.length == pytest.approx(2.0)


@pytest.mark.xfail(raises=ValueError)
def test_making_signal_with_incompatible_channels_info():

    meta_data = audio_signal.AudioMeta(100, 5, 8)
    data = np.ndarray(shape=(1, 200), dtype=np.uint8)

    audio_signal.AudioSignal(data, meta_data)


@pytest.mark.xfail(raises=ValueError)
def test_making_signal_with_data_type_other_than_declared():

    meta_data = audio_signal.AudioMeta(100, 2, 16)
    data = np.ndarray(shape=(2, 200), dtype=np.uint8)

    audio_signal.AudioSignal(data, meta_data)


@pytest.mark.xfail(raises=ValueError)
def test_making_signal_with_invalid_sampling_rate():

    meta_data = audio_signal.AudioMeta(-50, 2, 8)
    data = np.ndarray(shape=(2, 200), dtype=np.uint8)

    audio_signal.AudioSignal(data, meta_data)
