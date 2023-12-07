import os

import pytest
import numpy as np
import tempfile

from preprocessing import signal_writer
from preprocessing import audio_signal


SIGNAL_WRITER_OUTPUTS_PATH = os.path.join(os.environ['TEST_RESOURCES'], 'signal_writer_outputs')


@pytest.fixture
def example_signal():
    """Returns an audio signal to be written."""

    data = np.zeros((1, 100), dtype=np.float32)
    meta = audio_signal.AudioMeta(100, 1, 16)

    return audio_signal.AudioSignal(data, meta)


@pytest.fixture
def temp_output_dir():
    """Returns a temporary directory to write to."""

    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.mark.dependency(name='signal_writer_test', scope='session')
@pytest.mark.parametrize('generate_names,overwrite,pattern,given_names,expected', [
    (False, False, None, ['out_first', 'out_second', 'out_third'],
     {'out_first.wav', 'out_second.wav', 'out_third.wav'}),
    (True, False, None, None,
     {'audio_signal_0.wav', 'audio_signal_1.wav', 'audio_signal_2.wav'}),
    (False, True, None, ['out', 'out', 'out'],
     {'out.wav'}),
    (True, False, 'signal_pattern', None,
     {'signal_pattern_0.wav', 'signal_pattern_1.wav', 'signal_pattern_2.wav'}),
])
# pylint: disable=redefined-outer-name
def test_naming_files_with_given_names(example_signal, temp_output_dir,
                                       generate_names, overwrite,
                                       pattern, given_names, expected):
    """Tests if the writer names output files as expected."""

    params = signal_writer.WriterParams(
        generate_names=generate_names, overwrite_existing=overwrite,
        file_extension=signal_writer.WriterFileExtension.WAVE, output_name_pattern=pattern)

    writer = signal_writer.SignalWriter(params)

    if given_names is not None:
        for given_name in given_names:
            writer.write(example_signal, temp_output_dir, file_name=given_name)

    else:
        for _ in expected:
            writer.write(example_signal, temp_output_dir)

    assert set(os.listdir(temp_output_dir)) == expected


@pytest.mark.xfail(reason='File name was not provided and generation is disabled!',
                   raises=ValueError)
# pylint: disable=redefined-outer-name
def test_reporting_inability_to_generate_name(example_signal, temp_output_dir):
    """Tests if the writer reports inability to generate a file name."""

    params = signal_writer.WriterParams(
        generate_names=False,
        overwrite_existing=True,
        file_extension=signal_writer.WriterFileExtension.WAVE)

    writer = signal_writer.SignalWriter(params)

    writer.write(example_signal, temp_output_dir)


@pytest.mark.xfail(reason='File already exists:',
                   raises=FileExistsError)
# pylint: disable=redefined-outer-name
def test_reporting_existing_file(example_signal, temp_output_dir):
    """Tests if the writer reports existing file."""

    params = signal_writer.WriterParams(
        generate_names=False,
        overwrite_existing=False,
        file_extension=signal_writer.WriterFileExtension.WAVE)

    writer = signal_writer.SignalWriter(params)

    writer.write(example_signal, temp_output_dir, file_name='out')
    writer.write(example_signal, temp_output_dir, file_name='out')
