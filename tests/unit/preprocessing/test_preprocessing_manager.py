# -*- coding: utf-8 -*-
import os
import tempfile
import wave
from typing import Any
from typing import Dict

import pytest

from preprocessing import audio_signal
from preprocessing import preprocessing_manager
from preprocessing.processing_nodes import processing_node


class MockPreprocessor(processing_node.ProcessingNode):
    """Passes forward the input signals and records the processing info.

    The sampling rate of the output signal is twice as high as the input one.
    The number of processed signals is stored in the `number_of_processed_signals` property.
    """

    def __init__(self):
        super().__init__(False)

        self._number_of_processed_signals = 0

    @property
    def signature(self) -> str:
        return 'TestPreprocessor'

    @property
    def number_of_processed_signals(self) -> int:
        return self._number_of_processed_signals

    def process(
            self, signal: audio_signal.AudioSignal) -> audio_signal.AudioSignal:

        if self._allow_backward:
            self._store_processing_info(signal)

        return self._apply_transformations(signal)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'processing_node.ProcessingNode':
        return super().from_config(config)

    def _apply_transformations(
            self, signal: audio_signal.AudioSignal, *args, **kwargs) -> audio_signal.AudioSignal:

        new_signal_meta = signal.meta
        new_signal_meta.sampling_rate *= 2

        return audio_signal.AudioSignal(signal.data, new_signal_meta)

    def _transform_backwards(
            self, signal: audio_signal.AudioSignal) -> audio_signal.AudioSignal:
        return signal

    def _store_processing_info(
            self, signal: audio_signal.AudioSignal, *args, **kwargs):
        return


@pytest.fixture
def processing_setup():
    'Performs test directories cleaning and returns used paths.'

    preprocessing_inputs_path = os.path.join(
        os.environ['TEST_RESOURCES'], 'preprocessing_inputs')

    with tempfile.TemporaryDirectory() as preprocessing_outputs_path:
        yield preprocessing_outputs_path, preprocessing_inputs_path

# pylint: disable=redefined-outer-name


def test_detects_invalid_input_paths(processing_setup):
    """Checks if manager properly reacts to invalid input paths."""

    outputs_path, _ = processing_setup

    processor = MockPreprocessor()

    params = preprocessing_manager.ManagerParams(
        'nonexisting_path', outputs_path, [processor], False, r'.*\.wav')

    manager = preprocessing_manager.PreprocessingManager(params)

    manager.run_preprocessing()

    assert processor.number_of_processed_signals == 0


# pylint: disable=redefined-outer-name
def test_creates_output_directory_if_does_not_exist(processing_setup):
    """Checks if manager creates output directory if it does not exist."""

    outputs_path, inputs_path = processing_setup
    outputs_path = os.path.join(outputs_path, 'nonexisting_directory')

    params = preprocessing_manager.ManagerParams(
        inputs_path, outputs_path, [MockPreprocessor()], False, r'.*\.wav')

    manager = preprocessing_manager.PreprocessingManager(params)

    manager.run_preprocessing()

    assert os.path.exists(outputs_path)


@pytest.mark.parametrize('recursively,files_filter,expected', [
    (True, r'.*\.wav', {'file1.wav', 'file2.wav', 'file3.wav'}),
    (False, r'.*\.wav', {'file1.wav'}),
    (False, r'.*1\.wav', {'file1.wav'}),
    (True, r'.*1\.wav', {'file1.wav'})
])
@pytest.mark.dependency(depends=['signal_reader_test', 'signal_writer_test'], scope='session')
# pylint: disable=redefined-outer-name
def test_discovers_input_files_and_dumps_results(
        processing_setup, recursively, files_filter, expected):
    """Runs processing and checks the contents of the output directory."""

    outputs_path, inputs_path = processing_setup

    params = preprocessing_manager.ManagerParams(
        inputs_path, outputs_path, [MockPreprocessor()], recursively, files_filter)

    manager = preprocessing_manager.PreprocessingManager(params)

    manager.run_preprocessing()

    encountered_files = set(os.listdir(outputs_path))

    assert encountered_files == expected


@pytest.mark.parametrize('preprocessors,expected',
                         [([MockPreprocessor()], 88200), ([MockPreprocessor(), MockPreprocessor()], 176400)])
@pytest.mark.dependency(depends=['signal_reader_test', 'signal_writer_test'], scope='session')
# pylint: disable=redefined-outer-name
def test_properly_runs_processing(processing_setup, preprocessors, expected):
    """Checks whether the processors actually process the input signals."""

    outputs_path, inputs_path = processing_setup

    params = preprocessing_manager.ManagerParams(
        inputs_path, outputs_path, preprocessors, False, r'.*\.wav')
    manager = preprocessing_manager.PreprocessingManager(params)

    manager.run_preprocessing()

    assert len(os.listdir(outputs_path)) == 1

    with wave.open(os.path.join(outputs_path, 'file1.wav'), 'rb') as input_binary:
        target_sr = input_binary.getframerate()

        assert target_sr == expected
