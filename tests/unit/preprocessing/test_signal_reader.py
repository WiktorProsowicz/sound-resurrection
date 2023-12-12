# -*- coding: utf-8 -*-
import os
import tempfile

import librosa
import numpy as np
import pytest

from preprocessing import signal_reader

SIGNAL_READER_INPUTS_PATH = os.path.join(os.environ['TEST_RESOURCES'], 'signal_reader_inputs')


@pytest.mark.xfail(reason='File does not exist:', raises=FileNotFoundError)
def test_reporting_nonexisting_file():
    """Tests if the reader reports nonexisting file."""

    with tempfile.TemporaryDirectory() as temp_dir:
        input_path = os.path.join(temp_dir, 'nonexisting_file.wav')

        params = signal_reader.ReaderParams()
        reader = signal_reader.SignalReader(params)

        reader.read(input_path)


@pytest.mark.parametrize('dest_sr', [-1, 0, -44100])
@pytest.mark.xfail(reason='Destination sampling rate must be positive!', raises=ValueError)
def test_reader_creation_with_invalid_srate(dest_sr):
    """Tests if the reader creation fails when given invalid sampling rate."""

    params = signal_reader.ReaderParams(dest_sampling_rate=dest_sr)

    signal_reader.SignalReader(params)


@pytest.mark.parametrize('dest_channels', [-1, 0, -10])
@pytest.mark.xfail(reason='Destination number of channels must be positive!', raises=ValueError)
def test_reader_creation_with_invalid_channels(dest_channels):
    """Tests if the reader creation fails when given invalid number of channels."""

    params = signal_reader.ReaderParams(dest_channels=dest_channels)

    signal_reader.SignalReader(params)


@pytest.mark.xfail(reason='Unsupported file type: .txt!',
                   raises=signal_reader.IncorrectAudioFileError)
def test_decoding_unsupported_file_type():
    """Tests if the decoding fails when given unsupported file type."""

    input_path = os.path.join(SIGNAL_READER_INPUTS_PATH, 'unsupported_file.txt')

    params = signal_reader.ReaderParams()
    reader = signal_reader.SignalReader(params)

    reader.read(input_path)


@pytest.fixture
def wave_ground_truth():
    """Returns the actual data contained in expected wave signal."""

    time_vals = np.linspace(0, 1, 44100)
    time_vals = time_vals.reshape((1, -1))
    amplitude = .8
    frequency = 100

    return amplitude * np.sin(2 * np.pi * frequency * time_vals)


@pytest.mark.dependency(name='signal_reader_test', scope='session')
@pytest.mark.parametrize('dest_sr,dest_ch', [(None, None), (22050, None), (None, 2), (22050, 2)])
# pylint: disable=redefined-outer-name
def test_reader_wave_decoding(wave_ground_truth, dest_sr, dest_ch):
    """Tests if the reader decodes a wav file correctly."""

    input_path = os.path.join(SIGNAL_READER_INPUTS_PATH, 'signal.wav')

    params = signal_reader.ReaderParams(dest_sampling_rate=dest_sr, dest_channels=dest_ch)

    reader = signal_reader.SignalReader(params)

    signal = reader.read(input_path)

    if dest_sr is None:
        assert signal.meta.sampling_rate == 44100

    else:
        wave_ground_truth = librosa.resample(wave_ground_truth, orig_sr=44100, target_sr=dest_sr)

        assert signal.meta.sampling_rate == dest_sr

    if dest_ch is None:
        assert signal.meta.channels == 1

    else:
        assert signal.meta.channels == dest_ch

        for channel in range(dest_ch):
            assert np.allclose(signal.data[channel, :], wave_ground_truth[0, :], atol=1e-3)

    assert np.allclose(signal.data, wave_ground_truth, atol=1e-3)
