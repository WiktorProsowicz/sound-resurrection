# -*- coding: utf-8 -*-
import os

from preprocessing import audio_signal
from preprocessing import preprocessing_manager
from preprocessing.processing_nodes import processing_node


class MockPreprocessor(processing_node.ProcessingNode):
    """Mock preprocessor that passes input signals forward."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def signature(self) -> str:
        return 'TestPreprocessor'

    def _apply_transformations(
            self, signal: audio_signal.AudioSignal) -> audio_signal.AudioSignal:
        return signal

    def _transform_backwards(
            self, signal: audio_signal.AudioSignal) -> audio_signal.AudioSignal:
        return signal

    def _store_processing_info(self, signal: audio_signal.AudioSignal):
        return


def test_valid_processing():
    """Runs processing and checks the outputs."""

    input_path = os.path.join(
        os.environ['TEST_RESOURCES'], 'preprocessing_inputs')

    output_path = os.path.join(
        os.environ['TEST_RESULTS'], 'preprocessing_outputs')

    # params = preprocessing_manager.ManagerParams(
    #     input_path, output_path, [MockPreprocessor()], True, '*.wav')

    # TODO: Implement test.
