"""Scripts containing the training loop for the audio res enhancer model."""

import argparse
import json
import logging
import os
from typing import Any, Dict, Optional

import keras
import tensorflow as tf
import yaml  # type: ignore

from models import audio_resolution_enhancer
from preprocessing.dataset_utils import resolution_enhancer_dataset
from utilities import logging_utils


def _logger() -> logging.Logger:
    """Returns a for the script."""

    return logging.getLogger('audio_res_enhancer')


def _make_parser():
    """Returns an arguments parser for the script."""

    parser = argparse.ArgumentParser(
        "run_audio_res_enhancer_training.py",
        description="Runs the training loop for the audio res enhancer model.")

    parser.add_argument("config_path", type=str, help="Path to the config file.")

    return parser


def _read_config(config_path: str) -> Dict[str, Any]:
    """Reads the config file and returns its contents."""

    with open(config_path, 'r', encoding='utf-8') as config_file:
        config_dict = yaml.safe_load(config_file.read())

    return config_dict


def _obtain_last_checkpoint(config: Dict[str, Any]) -> Optional[str]:
    """Returns name of the last checkpoint file."""

    checkpoints = [file
                   for file in os.listdir(config['model_checkpoints_path'])
                   if '.model.keras' in file]

    if checkpoints:
        return max(checkpoints)


def _obtain_interrupted_epoch(config: Dict[str, Any]) -> int:
    """Returns number of the epoch at which the last checkpoint was saved."""

    latest_checkpoint = _obtain_last_checkpoint(config)

    if latest_checkpoint is not None:
        return int(latest_checkpoint.split('-')[1])

    return 0


def _build_model(config: Dict[str, Any]) -> audio_resolution_enhancer.AudioResolutionEnhancer:
    """Builds the model based on the config."""

    latest_checkpoint = _obtain_last_checkpoint(config)

    if latest_checkpoint is not None:
        _logger().debug('Loading model from checkpoint %s.', latest_checkpoint)

        return keras.models.load_model(os.path.join(
            config['model_checkpoints_path'], latest_checkpoint))

    model_config = audio_resolution_enhancer.ModelConfig(**config['model']['config'])
    model = audio_resolution_enhancer.AudioResolutionEnhancer(model_config)

    input_shape = tf.TensorShape([None, config['dataset_generator']['params']['inputs_length'], 1])
    model.build(input_shape=input_shape)

    return model


def _save_history(history, config: Dict[str, Any]):
    """Saves the history of training to a file."""

    history_path = os.path.join(config['model_checkpoints_path'], 'history.json')

    if not os.path.exists(history_path):
        old_history = {'epochs': [], 'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}

    else:
        with open(history_path, 'r', encoding='utf-8') as history_file:
            old_history = json.load(history_file)

    old_history['epochs'].extend(history.epoch)

    for hist_key in history.history:
        if hist_key not in old_history:
            old_history[hist_key] = []

        old_history[hist_key].extend(history.history[hist_key])

    with open(history_path, 'w', encoding='utf-8') as history_file:
        json.dump(old_history, history_file)


def _build_dataset_generator(
        config: Dict[str, Any]) -> resolution_enhancer_dataset.ResolutionEnhancerDatasetGenerator:
    """Builds the dataset generator based on the config."""

    generator_params = resolution_enhancer_dataset.GeneratorParameters(
        **config['dataset_generator']['params'])
    dataset_gen = resolution_enhancer_dataset.ResolutionEnhancerDatasetGenerator(generator_params)

    return dataset_gen


def _compile_model(model: audio_resolution_enhancer.AudioResolutionEnhancer,
                   config: Dict[str, Any]):
    """Compiles the model based on the config."""

    optimizer = keras.optimizers.deserialize(config['compilation']['optimizer'])
    loss = keras.losses.deserialize(config['compilation']['loss'])
    metrics = [keras.metrics.deserialize(metric) for metric in config['compilation']['metrics']]

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics)


def main(arguments: argparse.Namespace):
    """Runs the training loop for the audio res enhancer model.

    Args:
        arguments: Command line arguments.
    """

    config = _read_config(arguments.config_path)

    model = _build_model(config)

    _compile_model(model, config)

    model.summary(line_length=120, print_fn=lambda x: _logger().debug(x))

    dataset_gen = _build_dataset_generator(config)

    train_ds, test_ds = dataset_gen.generate_dataset(
        config['dataset_generator']['train_size'])

    with tf.device('/device:GPU:0'):

        checkpoint_template = os.path.join(
            config['model_checkpoints_path'],
            "ckpt-{epoch:02d}-{loss:.10f}.model.keras")

        training_callbacks = [
            keras.callbacks.ModelCheckpoint(checkpoint_template, save_freq='epoch',
                                            period=config['checkpoint_period'])]

        left_epochs = config['training']['epochs'] - _obtain_interrupted_epoch(config)

        _logger().debug('Starting training for %d epochs.', left_epochs)

        history = model.fit(
            x=train_ds,
            epochs=config['training']['epochs'],
            validation_data=test_ds,
            callbacks=training_callbacks,
            initial_epoch=_obtain_interrupted_epoch(config))

        _save_history(history, config)


if __name__ == '__main__':

    args_parser = _make_parser()
    args = args_parser.parse_args()

    gpus = tf.config.experimental.list_physical_devices('GPU')

    for gpu in gpus:
        tf.config.set_logical_device_configuration(gpu, [
            tf.config.LogicalDeviceConfiguration(memory_limit=1024 * 3)
        ])

    logging_utils.setup_logging()

    main(args)
