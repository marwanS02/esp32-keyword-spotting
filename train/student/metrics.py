from typing import Dict
import tensorflow as tf


def recall(matrix: tf.Tensor, idx: int) -> tf.Tensor:
    """Compute recall for a given class from a confusion matrix.

    Parameters
    ----------
    matrix : tf.Tensor
        Confusion matrix with shape [num_classes, num_classes].
        Rows = true labels, columns = predicted labels.
    idx : int
        Class index for which recall is computed.

    Returns
    -------
    tf.Tensor
        Recall value for the class (scalar tensor between 0 and 1).
    """
    tp = matrix[idx, idx]
    fn = tf.reduce_sum(matrix[idx, :]) - tp
    return tp / (tp + fn + tf.keras.backend.epsilon())


def precision(matrix: tf.Tensor, idx: int) -> tf.Tensor:
    """Compute precision for a given class from a confusion matrix.

    Parameters
    ----------
    matrix : tf.Tensor
        Confusion matrix with shape [num_classes, num_classes].
        Rows = true labels, columns = predicted labels.
    idx : int
        Class index for which precision is computed.

    Returns
    -------
    tf.Tensor
        Precision value for the class (scalar tensor between 0 and 1).
    """
    tp = matrix[idx, idx]
    fp = tf.reduce_sum(matrix[:, idx]) - tp
    return tp / (tp + fp + tf.keras.backend.epsilon())


def f1_score(matrix: tf.Tensor, idx: int) -> tf.Tensor:
    """Compute F1 score for a given class from a confusion matrix.

    Parameters
    ----------
    matrix : tf.Tensor
        Confusion matrix with shape [num_classes, num_classes].
        Rows = true labels, columns = predicted labels.
    idx : int
        Class index for which F1 score is computed.

    Returns
    -------
    tf.Tensor
        F1 score value for the class (scalar tensor between 0 and 1).
    """
    r = recall(matrix, idx)
    p = precision(matrix, idx)
    return 2 * (p * r) / (p + r + tf.keras.backend.epsilon())


def get_metrics(matrix: tf.Tensor, idx: int) -> Dict[str, float]:
    """Return recall, precision, and F1 score for a given class.

    Parameters
    ----------
    matrix : tf.Tensor
        Confusion matrix with shape [num_classes, num_classes].
    idx : int
        Class index.

    Returns
    -------
    dict
        Dictionary with keys ``recall``, ``precision``, and ``f1_score``.
        Values are floats (converted from scalar tensors).
    """
    metrics = {
        "recall": recall(matrix, idx),
        "precision": precision(matrix, idx),
        "f1_score": f1_score(matrix, idx),
    }
    return {k: float(v.numpy()) for k, v in metrics.items()}
