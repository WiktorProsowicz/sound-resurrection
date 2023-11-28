# -*- coding: utf-8 -*-
import os
import shutil

import pytest

from preprocessing import audio_signal
from preprocessing import preprocessing_manager
from preprocessing.processing_nodes import processing_node


class MockPreprocessor(processing_node.ProcessingNode):
    """Mock preprocessor that passes input signals forward."""

    def __init__(self, *args, **kwargs):
        super().__init__(True)

    @property
    def signature(self) -> str:
        return 'TestPreprocessor'

    def process(
            self, signal: audio_signal.AudioSignal) -> audio_signal.AudioSignal:

        if self._allow_backward:
            self._store_processing_info(signal)

        return self._apply_transformations(signal)

    def _apply_transformations(
            self, signal: audio_signal.AudioSignal, *args, **kwargs) -> audio_signal.AudioSignal:
        return signal

    def _transform_backwards(
            self, signal: audio_signal.AudioSignal) -> audio_signal.AudioSignal:
        return signal

    def _store_processing_info(
            self, signal: audio_signal.AudioSignal, *args, **kwargs):
        return


PREPROCESSING_OUTPUTS_PATH = os.path.join(
    os.environ['TEST_RESULTS'], 'preprocessing_outputs')


def clean_path_recursively(path: str):
    """Removes content of a path recursively."""

    for root, dirs, files in os.walk(path):

        for file in files:
            os.remove(os.path.join(root, file))
        for directory in dirs:
            shutil.rmtree(os.path.join(root, directory))


@pytest.fixture(autouse=True)
def test_setup():
    'Returns path to nested directories containing audio files.'

    clean_path_recursively(PREPROCESSING_OUTPUTS_PATH)

    yield


def test_valid_processing_recursive():
    """Runs processing and checks the outputs."""

    input_path = os.path.join(
        os.environ['TEST_RESOURCES'], 'preprocessing_inputs')

    params = preprocessing_manager.ManagerParams(
        input_path, PREPROCESSING_OUTPUTS_PATH, [MockPreprocessor()], True, '.*\\.wav')

    manager = preprocessing_manager.PreprocessingManager(params)

    manager.run_preprocessing()

    encountered_files = set(os.listdir(PREPROCESSING_OUTPUTS_PATH))

    assert (encountered_files == {'file1.wav', 'file2.wav', 'file3.wav'})
