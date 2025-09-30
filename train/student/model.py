import tensorflow as tf
import numpy as np


def create_micro_kws_model(
    model_settings: dict, model_name: str = "micro_kws_student"
) -> tf.keras.Model:
    """Build a compact CNN model for keyword spotting using the Keras API.

    Parameters
    ----------
    model_settings : dict
        Dictionary containing model configuration, including:
        - ``dct_coefficient_count`` (int): Feature dimension (frequency bins).
        - ``spectrogram_length`` (int): Number of frames (time steps).
        - ``fingerprint_size`` (int): Flattened input size (time × frequency).
        - ``label_count`` (int): Number of output classes.
    model_name : str, optional
        Name assigned to the model (default: "micro_kws_student").

    Returns
    -------
    tf.keras.Model
        Compiled Keras model.
    """
    input_frequency_size = model_settings["dct_coefficient_count"]
    input_time_size = model_settings["spectrogram_length"]

    # Input layer: flattened feature vector (e.g., MFCC or spectrogram features).
    inputs = tf.keras.Input(shape=(model_settings["fingerprint_size"]), name="input")

    # Reshape flattened input into [time, frequency, channels] for CNN processing.
    x = tf.reshape(inputs, shape=(-1, input_time_size, input_frequency_size, 1))

    # Convolution block 1: depthwise separable conv (efficient feature extraction).
    x = tf.keras.layers.DepthwiseConv2D(
        depth_multiplier=16,
        kernel_size=(5, 4),
        strides=(2, 2),
        padding="SAME",
        activation="relu",
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(filters=50, kernel_size=(1, 1), activation="relu")(x)

    # Convolution block 2: standard convolution for higher-level features.
    x = tf.keras.layers.Conv2D(
        filters=16,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="SAME",
        activation="relu",
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    # Dense block: compact fully connected layers with dropout regularization.
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(units=32, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    # Output layer: softmax classifier over labels.
    outputs = tf.keras.layers.Dense(
        units=model_settings["label_count"], activation="softmax"
    )(x)

    model = tf.keras.Model(inputs, outputs, name=model_name)
    return model
