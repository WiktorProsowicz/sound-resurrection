#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script runs preprocessing pipeline according to provided config."""
import argparse
from typing import Any
from typing import Dict

import yaml  # type: ignore

from preprocessing import preprocessing_manager
from preprocessing.processing_nodes import downsampling_processor
from preprocessing.processing_nodes import fragments_cutting_processor
from preprocessing.processing_nodes import noise_adding_processor
from preprocessing.processing_nodes import processing_node
from utilities import logging_utils

STR_TO_CLASS_DICT: Dict[str, type[processing_node.ProcessingNode]] = {
    'FragmentsCuttingProcessor': fragments_cutting_processor.FragmentsCuttingProcessor,
    'DownSamplingProcessor': downsampling_processor.DownSamplingProcessor,
    'NoiseAddingProcessor': noise_adding_processor.NoiseAddingProcessor,
}


def _spawn_processor(processor_spec: Dict[str, Any]) -> processing_node.ProcessingNode:
    """Creates a processing node according to provided specification.

    Args:
        processor_spec: Dictionary containing fields describing the
        processor and the specific processor's configuration.

    Returns:
        Spawned processor.

    Raises:
        ValueError: If the given processor type is unknown.
    """

    processor_type = processor_spec['type']
    processor_cfg = processor_spec['processor_cfg']

    if processor_type in STR_TO_CLASS_DICT:
        return STR_TO_CLASS_DICT[processor_type].from_config(processor_cfg)

    raise ValueError(f'Unknown processor type: {processor_type}')


def _load_config(path: str) -> Dict[str, Any]:
    """Loads yaml config from the given path."""

    with open(path, 'r', encoding='utf-8') as config_file:
        return yaml.safe_load(config_file)


def main(config: Dict[str, Any]):
    """Initializes preprocessing pipeline and runs it.

    Args:
        args: Script configuration.
    """

    m_params: Dict[str, Any] = config['manager_params']

    sorted_processor_specs = sorted(m_params['processors'], key=lambda spec: spec['index'])
    processors = [_spawn_processor(p_config) for p_config in sorted_processor_specs]

    manager_params = preprocessing_manager.ManagerParams(
        m_params['input_path'], m_params['output_path'], processors,
        m_params['search_recursively'], m_params['filter']
    )

    manager = preprocessing_manager.PreprocessingManager(manager_params)

    manager.run_preprocessing()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Runs preprocessing pipeline according to provided config.')

    parser.add_argument('config_path', help='path to the config file')

    args = parser.parse_args()

    logging_utils.setup_logging()

    main(_load_config(args.config_path))
