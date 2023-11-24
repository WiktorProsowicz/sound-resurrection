"""Script runs preprocessing pipeline according to provided config."""

import argparse
from typing import Dict
import yaml
import os
import librosa

from preprocessing.processing_nodes import fragments_cutting_processor, processing_node
from preprocessing import preprocessing_manager

STR_TO_CLASS_DICT = {
    'FragmentsCuttingProcessor': fragments_cutting_processor.FragmentsCuttingProcessor
}


def _spawn_processor(processor_spec: Dict[str, any]) -> processing_node.ProcessingNode:

    processor_type = processor_spec["type"]
    processor_cfg = processor_spec["processor_cfg"]

    if processor_type in STR_TO_CLASS_DICT:
        return STR_TO_CLASS_DICT[processor_type].from_config(processor_cfg)

    raise ValueError(f'Unknown processor type: {processor_type}')


def _load_config(path: str) -> Dict[str, any]:

    with open(path, 'r') as config_file:
        return yaml.safe_load(config_file)


def main(config: Dict[str, any]):
    """Initializes preprocessing pipeline and runs it.

    Args:
        args: Script configuration.
    """

    m_params = config["manager_params"]

    sorted_processor_specs = sorted(m_params["processors"], key=lambda spec: spec["index"])
    processors = [_spawn_processor(p_config) for p_config in sorted_processor_specs]

    manager_params = preprocessing_manager.ManagerParams(
        m_params["input_path"], m_params["output_path"], processors,
        m_params["search_recursively"], m_params["filter"]
    )

    manager = preprocessing_manager.PreprocessingManager(manager_params)

    manager.run_preprocessing()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Runs preprocessing pipeline according to provided config.")

    parser.add_argument("config_path", help="path to the config file")

    args = parser.parse_args()

    main(_load_config(args.config_path))
