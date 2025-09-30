from typing import Optional, List
import tensorflow as tf


def get_early_stopping_callback() -> Optional[tf.keras.callbacks.Callback]:
    """Create an EarlyStopping callback using the Keras API.

    The callback monitors validation loss and stops training if no
    improvement is observed for a given patience period.

    Returns
    -------
    tf.keras.callbacks.Callback or None
        Configured EarlyStopping callback.
    """
    return tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0,
        patience=10,              # Minimum of 10 epochs without improvement
        verbose=0,
        mode="min",
        baseline=None,
        restore_best_weights=False,
        start_from_epoch=0,
    )


def get_callbacks() -> List[tf.keras.callbacks.Callback]:
    """Assemble and return all callbacks used during training.

    Returns
    -------
    list of tf.keras.callbacks.Callback
        List of configured callbacks (currently includes EarlyStopping).
    """
    callbacks: List[tf.keras.callbacks.Callback] = []
    early_stopping = get_early_stopping_callback()
    if early_stopping:
        callbacks.append(early_stopping)
    return callbacks
