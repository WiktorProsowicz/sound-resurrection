# -*- coding: utf-8 -*-
"""Contains script for training SoundPainter model."""

import argparse
import logging
from typing import Dict, Any, Optional
import yaml  # type: ignore
import os

import keras
import tensorflow as tf

from utilities import logging_utils
from models import sound_painter
from preprocessing.dataset_utils import painter_dataset_gen
from utilities import callbacks


def _parse_config(config_path: str) -> Dict[str, Any]:
    """Parses configuration file and returns its content as a dictionary."""

    with open(config_path, 'r') as config_file:
        parsed_config = yaml.safe_load(config_file)

    model_params = parsed_config['model']['params']
    g_coeffs = model_params['generator_loss_coeffs']

    model_params['generator_loss_coeffs'] = (g_coeffs['contextual'],
                                             g_coeffs['gradient'],
                                             g_coeffs['wasserstein'])

    d_coeffs = model_params['discriminator_loss_coeffs']

    model_params['discriminator_loss_coeffs'] = (d_coeffs['grad_penalty'],
                                                 d_coeffs['wasserstein'])

    return parsed_config


def _build_args_parser():
    """Creates parser for command-line arguments."""

    parser = argparse.ArgumentParser(
        description='Script for training SoundPainter model.'
    )

    parser.add_argument('config_path', help='Path to training configuration file.')

    return parser


def _logger():
    """Returns logger for this module."""

    return logging.getLogger('SoundPainter')


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


def _build_ds_generator(config: Dict[str, Any]) -> painter_dataset_gen.SoundPainterDatasetGenerator:
    """Builds dataset generator from provided config."""

    generator_params = painter_dataset_gen.GeneratorParams(**config['dataset_generator']['params'])

    return painter_dataset_gen.SoundPainterDatasetGenerator(generator_params)


def _compile_model(model: sound_painter.SoundPainter, config: Dict[str, Any]):
    """Compiles the model with provided optimizer and loss."""

    d_optimizer = keras.optimizers.deserialize(config['compilation']['discriminator_optimizer'])
    g_optimizer = keras.optimizers.deserialize(config['compilation']['generator_optimizer'])

    model.compile(g_optimizer, d_optimizer)


def _build_model(config: Dict[str, Any]) -> sound_painter.SoundPainter:
    """Builds the model form provided config."""

    latest_checkpoint = _obtain_last_checkpoint(config)

    if latest_checkpoint is not None:
        _logger().debug('Loading model from checkpoint %s.', latest_checkpoint)

        return keras.models.load_model(os.path.join(
            config['model_checkpoints_path'], latest_checkpoint))

    model_params = sound_painter.SoundPainterParams(**config['model']['params'])
    model = sound_painter.SoundPainter(model_params)

    input_shape = tf.TensorShape((config['dataset_generator']['params']['batch_size'],
                                  config['model']['shape_freq'],
                                  config['model']['shape_time'], 1))

    model.build((input_shape, input_shape))

    return model


def main(arguments: argparse.Namespace):
    """Runs training of SoundPainter model.

    Args:
        config: Configuration dictionary.
    """

    config = _parse_config(arguments.config_path)

    _logger().debug('Running training with config: %s', arguments.config_path)

    model = _build_model(config)

    _compile_model(model, config)

    ds_generator = _build_ds_generator(config)

    train_ds, test_ds = ds_generator.generate(config['dataset_generator']['train_size'])

    with tf.device('/device:GPU:0'):

        checkpoint_template = os.path.join(
            config['model_checkpoints_path'],
            "ckpt-{epoch:02d}-epoch.model.keras")

        training_callbacks = [
            keras.callbacks.ModelCheckpoint(checkpoint_template, save_freq='epoch',
                                            period=config['checkpoint_period']),
            callbacks.MetricsSaving(config['model_checkpoints_path'])
        ]

        left_epochs = config['training']['epochs'] - _obtain_interrupted_epoch(config)

        _logger().debug('Starting training for %d epochs.', left_epochs)

        history = model.fit(
            x=train_ds,
            epochs=config['training']['epochs'],
            validation_data=test_ds,
            callbacks=training_callbacks,
            initial_epoch=_obtain_interrupted_epoch(config))


if __name__ == "__main__":

    args = _build_args_parser().parse_args()

    logging_utils.setup_logging()

    gpus = tf.config.experimental.list_physical_devices('GPU')

    for gpu in gpus:
        tf.config.set_logical_device_configuration(gpu, [
            tf.config.LogicalDeviceConfiguration(memory_limit=1024 * 3)
        ])

    logging.getLogger('numba').setLevel(logging.WARNING)

    main(args)
