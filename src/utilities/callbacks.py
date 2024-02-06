"""Contains custom callbacks for models training."""

import keras


class MetricsSaving(keras.callbacks.Callback):
    """Saves metrics and losses to a file after each epoch."""

    def __init__(self, metrics_path: str, *args, **kwargs):
        """Initializes the callback.

        Args:
            metrics_path: Path to the file where metrics shall be saved.
        """

        super().__init__(*args, **kwargs)

        self._metrics_path = metrics_path

    def on_epoch_end(self, epoch, logs=None):
        """Overrides method of the base keras.callbacks.Callback class."""

        print(epoch, logs)

        # with open(self._metrics_path, 'a') as metrics_file:
        #     metrics_file.write(f'{epoch},{logs["loss"]},{logs["accuracy"]}\n')
