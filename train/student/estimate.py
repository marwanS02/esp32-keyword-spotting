# Imports
from functools import reduce
import numpy as np
import collections
from typing import List
import tensorflow as tf

# Define custom types for tensors and layers
MyTensor = collections.namedtuple("MyTensor", ("idx", "name", "shape", "dtype", "is_const"))
MyLayer = collections.namedtuple("MyLayer", ("idx", "name", "inputs", "outputs"))


def estimate_conv2d_macs(in_shape: List[int], kernel_shape: List[int], out_shape: List[int]) -> int:
    """Estimate the number of multiply-accumulate operations (MACs) required for a Conv2D layer.

    Parameters
    ----------
    in_shape : list
        Input tensor shape [batch_size, input_h, input_w, input_c] (NHWC).
    kernel_shape : list
        Weight tensor shape [output_channels, kernel_h, kernel_w, input_c] (OHWI).
    out_shape : list
        Output tensor shape [batch_size, output_h, output_w, output_c] (NHWC).

    Returns
    -------
    int
        Estimated number of MAC operations.
    """
    input_n, input_h, input_w, input_c = in_shape
    kernel_oc, kernel_h, kernel_w, kernel_ic = kernel_shape
    output_n, output_h, output_w, output_c = out_shape

    assert input_n == output_n == 1  # Assumes inference mode (batch_size = 1)
    assert input_c == kernel_ic
    assert output_c == kernel_oc

    macs = input_n * output_h * output_w * output_c * kernel_h * kernel_w * input_c
    return macs


def estimate_depthwise_conv2d_macs(
    in_shape: List[int], kernel_shape: List[int], out_shape: List[int], channel_mult: int
) -> int:
    """Estimate the number of MACs required for a Depthwise Conv2D layer.

    Parameters
    ----------
    in_shape : list
        Input tensor shape [batch_size, input_h, input_w, input_c] (NHWC).
    kernel_shape : list
        Weight tensor shape [1, kernel_h, kernel_w, output_channels].
    out_shape : list
        Output tensor shape [batch_size, output_h, output_w, output_c] (NHWC).
    channel_mult : int
        Channel multiplier (see TensorFlow DepthwiseConv2D).

    Returns
    -------
    int
        Estimated number of MAC operations.
    """
    input_n, input_h, input_w, input_c = in_shape
    _, kernel_h, kernel_w, kernel_oc = kernel_shape
    output_n, output_h, output_w, output_c = out_shape

    assert input_n == output_n == 1
    assert output_c == kernel_oc == input_c * channel_mult

    macs = input_n * output_h * output_w * input_c * channel_mult * kernel_h * kernel_w
    return macs


def estimate_fully_connected_macs(in_shape: List[int], filter_shape: List[int], out_shape: List[int]) -> int:
    """Estimate the number of MACs required for a Fully Connected (Dense) layer.

    Parameters
    ----------
    in_shape : list
        Input tensor shape [input_h, input_w].
    filter_shape : list
        Weight tensor shape [filter_h, filter_w].
    out_shape : list
        Output tensor shape [output_h, output_w].

    Returns
    -------
    int
        Estimated number of MAC operations.
    """
    input_h, input_w = in_shape
    filter_h, filter_w = filter_shape
    output_h, output_w = out_shape

    assert input_w == filter_h
    assert output_w == filter_w
    assert input_h == output_h

    macs = input_h * input_w * output_w
    return macs


def estimate_rom(tensors: List[MyTensor]) -> int:
    """Estimate the ROM footprint (in bytes) required to store constant tensors (weights/biases).

    Parameters
    ----------
    tensors : list of MyTensor
        Model tensors.

    Returns
    -------
    int
        Estimated ROM usage in bytes.

    Notes
    -----
    - Only constant tensors contribute (is_const=True).
    - int8 weights consume 1 byte/element.
    - int32 or float32 biases consume 4 bytes/element.
    - Based on fully quantized TFLite models.
    """
    rom_bytes = 0
    for tensor in tensors:
        if tensor.is_const:
            nb_params = np.prod(tensor.shape)
            if tensor.dtype == "int8":
                rom_bytes += nb_params
            elif tensor.dtype in ("int32", "float32"):
                rom_bytes += nb_params * 4
    return rom_bytes


def estimate_ram(tensors: List[MyTensor], layers: List[MyLayer]) -> int:
    """Estimate peak RAM usage for intermediate tensors during inference.

    Parameters
    ----------
    tensors : list of MyTensor
        Model tensors.
    layers : list of MyLayer
        Model layers.

    Returns
    -------
    int
        Estimated RAM usage in bytes.

    Notes
    -----
    - Assumes sequential model execution (no branches).
    - Only intermediate activations are considered.
    - In-place operations are disallowed (input/output must be distinct).
    - Peak memory is estimated as the maximum sum of input/output tensor sizes
      across all layers.
    """

    def tensor_size(tensor: MyTensor) -> int:
        """Compute tensor size in bytes."""
        element_size = 1 if tensor.dtype == "int8" else 4
        return np.prod(tensor.shape) * element_size

    max_layer_ram = 0
    for layer in layers:
        input_tensors = [t for t in tensors if t.idx in layer.inputs and not t.is_const]
        output_tensors = [t for t in tensors if t.idx in layer.outputs and not t.is_const]
        layer_ram = sum(tensor_size(t) for t in input_tensors + output_tensors)
        max_layer_ram = max(max_layer_ram, layer_ram)

    return max_layer_ram
