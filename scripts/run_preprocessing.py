#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script runs preprocessing pipeline according to provided config."""
import argparse
import io
import logging
from typing import Any
from typing import Dict

import yaml  # type: ignore

from preprocessing import preprocessing_manager
from preprocessing.processing_nodes import downsampling_processor
from preprocessing.processing_nodes import fragments_cutting_processor
from preprocessing.processing_nodes import noise_adding_processor
from preprocessing.processing_nodes import processing_node
from preprocessing.processing_nodes import resampling_processor
from utilities import logging_utils

STR_TO_CLASS_DICT: Dict[str, type[processing_node.ProcessingNode]] = {
    'FragmentsCuttingProcessor': fragments_cutting_processor.FragmentsCuttingProcessor,
    'DownSamplingProcessor': downsampling_processor.DownSamplingProcessor,
    'NoiseAddingProcessor': noise_adding_processor.NoiseAddingProcessor,
    'ResamplingProcessor': resampling_processor.ResamplingProcessor,
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


def _print_manager_params(manager_params: preprocessing_manager.ManagerParams):
    """Prints preprocessing configuration to the console."""

    stream = io.StringIO()

    stream.write('Preprocessing configuration:\n')

    stream.write(f'Input path: \'{manager_params.input_path}\'\n')
    stream.write(f'Output path: \'{manager_params.output_path}\'\n')
    stream.write(f'Search recursively: {manager_params.search_recursively}\n')
    stream.write(f'Filter: \'{manager_params.filter}\'\n')
    stream.write('Processors:\n\t')

    stream.write('\n\t'.join(processor.signature for processor in manager_params.processors))

    logging.info(stream.getvalue())


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

    _print_manager_params(manager_params)

    manager = preprocessing_manager.PreprocessingManager(manager_params)

    manager.run_preprocessing()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Runs preprocessing pipeline according to provided config.')

    parser.add_argument('config_path', help='path to the config file')

    args = parser.parse_args()

    logging_utils.setup_logging()
    logging.getLogger('numba').setLevel(logging.WARNING)

    main(_load_config(args.config_path))
