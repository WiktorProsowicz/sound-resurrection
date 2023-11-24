# -*- coding: utf-8 -*-
"""Contains a class responsible for managing the offline preprocessing."""
import dataclasses
import logging
import os
import pathlib
import re
from typing import Generator
from typing import List
from typing import Optional

from preprocessing import audio_signal
from preprocessing import signal_reader
from preprocessing import signal_writer
from preprocessing.processing_nodes import processing_node


@dataclasses.dataclass
class ManagerParams:
    """Contains parameters for the ProcessingManager."""

    input_path: str
    output_path: str
    processors: List[processing_node.ProcessingNode]
    search_recursively: bool = False
    filter: Optional[str] = None


class PreprocessingManager:
    """Manages the offline files preprocessing.

    Its responsibility is to collect and apply given files transformations, read
    prepared files and save processed ones to correct directory.
    """

    def __init__(self, params: ManagerParams):
        """Initializes the manager's parameters."""

        self._files_filter: re.Pattern = re.compile(params.filter)
        self._params: ManagerParams = params
        self._processed_files_names: List[str] = []

        reader_params = signal_reader.ReaderParams()
        self._reader = signal_reader.SignalReader(reader_params)

    def run_preprocessing(self):
        """Runs the preprocessing pipeline.

        At the beginning performs a validation based on the stored configuration.
        """

        if not self._is_path_valid(self._params.input_path):
            logging.error(
                "Input directory '%s' is not valid!", self._params.input_path)
            return

        if not self._is_path_valid(self._params.output_path):
            logging.debug(
                "Output directory '%s' does not exist. Creating...", self._params.output_path)

            os.makedirs(self._params.output_path)

        signals = self._run_processing_chain()

        writer_params = signal_writer.WriterParams(
            True, True, signal_writer.WriterFileExtension.WAVE)

        writer = signal_writer.SignalWriter(writer_params)

        for file_name, signal in zip(self._processed_files_names, signals):
            writer.write(signal, self._params.output_path, file_name=file_name)

        self._processed_files_names.clear()

    def _run_processing_chain(self) -> List[audio_signal.AudioSignal]:
        """Performs a processing over input files.

        Composes a chain of processing coroutines and collected processed signals.

        Returns:
            Processed audio signals.
        """

        collected_signals: List[audio_signal.AudioSignal] = []

        def ending_coroutine() -> Generator[None, audio_signal.AudioSignal, None]:

            while True:

                signal: audio_signal.AudioSignal = yield
                collected_signals.append(signal)

        chain_head = ending_coroutine()  # pylint: disable=assignment-from-no-return
        next(chain_head)

        temp_head = chain_head

        for processor in self._params.processors:
            temp_head = self._spawn_signal_processing_coroutine(
                processor, temp_head)
            next(temp_head)

        file_sender = self._spawn_file_sending_coroutine(temp_head)
        next(file_sender)

        return collected_signals

    def _spawn_file_sending_coroutine(
        self,
        first_processing_coroutine: Generator[None, audio_signal.AudioSignal, None]
    ) -> Generator[None, None, None]:
        """Spawns a coroutine sending audio signals to processors.

        The created coroutine iterates over input files and decodes them to audio
        signals. Then sends the created signals to first processor in line.

        Args:
            first_processing_coroutine: Next processor to send the audio signals to.
        """

        for path in self._get_input_paths():
            signal = self._reader.read(path)
            first_processing_coroutine.send(signal)

            self._processed_files_names.append(pathlib.Path(path).stem)

        first_processing_coroutine.close()
        yield

    def _spawn_signal_processing_coroutine(
        self,
        processor: processing_node.ProcessingNode,
        processing_coroutine: Generator[None, audio_signal.AudioSignal, None]
    ) -> Generator[None, audio_signal.AudioSignal, None]:
        """Spawns a coroutine performing signal processing.

        The spawned coroutine waits for an audio signal, processes it
        and passes the result to the next coroutine.

        Args:
            processor: Processing node which shall perform the processing.
            processing_coroutine: Next coroutine to send the result to.
        """

        try:
            while True:
                signal = yield
                processing_coroutine.send(processor.process(signal))

        except GeneratorExit as exc:
            processing_coroutine.close()
            raise exc

    def _get_input_paths(self) -> Generator[str, None, None]:
        """Yields paths to files to process."""

        for root, _, files in os.walk(self._params.input_path,):
            for file in files:
                if self._files_filter.match(file):
                    yield os.path.join(root, file)

            if not self._params.search_recursively:
                break

    def _is_path_valid(self, path: str) -> bool:
        return os.path.exists(path) and os.path.isdir(path)
