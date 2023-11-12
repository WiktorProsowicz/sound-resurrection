"""Contains a class responsible for managing the offline preprocessing."""

import dataclasses
import os
import logging
from typing import List, Optional, Generator

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

        self._params = params

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

        # TODO: Implement processing algorithm.

    def _is_path_valid(self, path: str) -> bool:
        return os.path.exists([path]) and os.path.isdir(path)
