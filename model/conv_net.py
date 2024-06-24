# adapted from https://github.com/facebookresearch/audiocraft/blob/main/audiocraft/modules/seanet.py
# using MLX built in convolutions instead of SEANet custom convs.
# ELU: https://github.com/ml-explore/mlx/blob/main/python/mlx/nn/layers/activations.py
"""
Encoder and Decoder 
"""

import math
from functools import partial
from typing import List, Dict, Any, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from model.conv_wrappers import SConvTranspose1d, SConv1d


@partial(mx.compile, shapeless=True)
def elu(x, alpha=1.0):
    return mx.where(x > 0, x, alpha * (mx.exp(x) - 1))

class ELU(nn.Module):
    r"""Applies the Exponential Linear Unit.
        Simply ``mx.where(x > 0, x, alpha * (mx.exp(x) - 1))``.

    See :func:`elu` for the functional equivalent.

    Args:
        alpha: the :math:`\alpha` value for the ELU formulation. Default: ``1.0``
    """
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def __call__(self, x):
        return elu(x, self.alpha)


class ConvBlock(nn.Module):
    """Convolution block adapted from SEANet model.

    Args:
        dim (int): Dimension of the input/output.
        kernel_sizes (list): List of kernel sizes for the convolutions.
        dilations (list): List of dilations for the convolutions.
        compress (int): Reduced dimensionality in residual branches (from Demucs v3).
        padding (bool): True, truth value of padding convolutions.
    """
    def __init__(self, dim: int, kernel_sizes: List[int] = [3, 1], dilations: List[int] = [1, 1],
                 elu_alpha: float = 1.0, compress: int = 2, padding: bool = True):
        super().__init__()
        assert len(kernel_sizes) == len(dilations), 'Number of kernel sizes should match number of dilations'
        act = ELU(elu_alpha)
        hidden = dim // compress
        block = []
        for i, (kernel_size, dilation) in enumerate(zip(kernel_sizes, dilations)):
            in_chs = dim if i == 0 else hidden
            out_chs = dim if i == len(kernel_sizes) - 1 else hidden
            block += [
                act,
                SConv1d(in_chs, out_chs, kernel_size=kernel_size, dilation=dilation, batch_norm_feat=out_chs),
            ]
        self.block = nn.Sequential(*block)

    def __call__(self, x):
        return x + self.block(x)


class ConvEncoder(nn.Module):
    """SEANet-inspired convolution encoder, no residual connections.

    Args:
        channels (int): Audio channels.
        dimension (int): Intermediate representation dimension.
        n_filters (int): Base width for the model.
        n_residual_layers (int): nb of residual layers.
        ratios (Sequence[int]): kernel size and stride ratios. The encoder uses downsampling ratios instead of
            upsampling ratios, hence it will use the ratios in the reverse order to the ones specified here
            that must match the decoder order. We use the decoder order as some models may only employ the decoder.
        kernel_size (int): Kernel size for the initial convolution.
        last_kernel_size (int): Kernel size for the initial convolution.
        residual_kernel_size (int): Kernel size for the residual layers.
        dilation_base (int): How much to increase the dilation with each layer.
        compress (int): Reduced dimensionality in residual branches (from Demucs v3).
        disable_norm_outer_blocks (int): Number of blocks for which we don't apply norm.
            For the encoder, it corresponds to the N first blocks.
    """
    def __init__(self, channels: int = 1, dimension: int = 128, n_filters: int = 32, n_residual_layers: int = 1,
                 ratios: List[int] = [8, 5, 4, 2], elu_alpha: float = 1.0, kernel_size: int = 7,
                 last_kernel_size: int = 7, residual_kernel_size: int = 3, dilation_base: int = 2,
                 compress: int = 2,
                 disable_norm_outer_blocks: int = 0):
        super().__init__()
        self.channels = channels
        self.dimension = dimension
        self.n_filters = n_filters
        self.ratios = list(reversed(ratios))
        del ratios
        self.n_residual_layers = n_residual_layers
        self.hop_length = np.prod(self.ratios)
        self.n_blocks = len(self.ratios) + 2  # first and last conv + residual blocks
        self.disable_norm_outer_blocks = disable_norm_outer_blocks
        assert self.disable_norm_outer_blocks >= 0 and self.disable_norm_outer_blocks <= self.n_blocks, \
            "Number of blocks for which to disable norm is invalid." \
            "It should be lower or equal to the actual number of blocks in the network and greater or equal to 0."

        act = ELU(alpha=elu_alpha)
        mult = 1
        model: List[nn.Module] = [
            SConv1d(channels, mult * n_filters, kernel_size, batch_norm_feat=mult * n_filters)
        ]
        # Downsample to raw audio scale
        for i, ratio in enumerate(self.ratios):
            # Add residual layers
            for j in range(n_residual_layers):
                model += [
                    ConvBlock(mult * n_filters, kernel_sizes=[residual_kernel_size, 1],
                                      dilations=[dilation_base ** j, 1],
                                      elu_alpha=elu_alpha, compress=compress)]

            # Add downsampling layers
            model += [
                act,
                SConv1d(mult * n_filters, mult * n_filters * 2,
                            kernel_size=ratio * 2, stride=ratio,batch_norm_feat=mult * n_filters * 2),
            ]
            mult *= 2

        model += [
            act,
            SConv1d(mult * n_filters, dimension, last_kernel_size, batch_norm_feat=dimension)
        ]

        self.model = nn.Sequential(*model)

    def __call__(self, x):
        return self.model(x)   


class ConvDecoder(nn.Module):
    """SEANet-inspired convolution encoder, no residual connections.
    Args:
        channels (int): Audio channels.
        dimension (int): Intermediate representation dimension.
        n_filters (int): Base width for the model.
        n_residual_layers (int): nb of residual layers.
        ratios (Sequence[int]): kernel size and stride ratios
        kernel_size (int): Kernel size for the initial convolution.
        last_kernel_size (int): Kernel size for the initial convolution.
        residual_kernel_size (int): Kernel size for the residual layers.
        dilation_base (int): How much to increase the dilation with each layer.
        compress (int): Reduced dimensionality in residual branches (from Demucs v3).
        trim_right_ratio (float): Ratio for trimming at the right of the transposed convolution under the causal setup.
            If equal to 1.0, it means that all the trimming is done at the right.
    """
    def __init__(self, channels: int = 1, dimension: int = 128, n_filters: int = 32, n_residual_layers: int = 1,
                 ratios: List[int] = [8, 5, 4, 2], elu_alpha: float = 1.0, kernel_size: int = 7,
                 last_kernel_size: int = 7, residual_kernel_size: int = 3, dilation_base: int = 2,
                 compress: int = 2):
        super().__init__()
        self.dimension = dimension
        self.channels = channels
        self.n_filters = n_filters
        self.ratios = ratios
        del ratios
        self.n_residual_layers = n_residual_layers
        self.hop_length = np.prod(self.ratios)

        act = ELU(alpha=elu_alpha)
        mult = int(2 ** len(self.ratios))
        model: List[nn.Module] = [
            SConv1d(dimension, mult * n_filters, kernel_size, batch_norm_feat=mult * n_filters)
        ]
        # Upsample to raw audio scale
        scales = [128, 40, 21, 1]
        for i, ratio in enumerate(self.ratios):
            # Add upsampling layer
            model += [
                act,
                SConvTranspose1d(mult * n_filters, mult * n_filters//2,
                                 kernel_size=ratio*2, stride=ratio, scale=scales[i], batch_norm_feat=mult * n_filters//2),
            ]
            # Add Conv layers
            for j in range(n_residual_layers):
                model += [
                    ConvBlock(mult * n_filters//2, kernel_sizes=[residual_kernel_size, 1],
                             compress=compress,dilations=[dilation_base ** j, 1], elu_alpha=elu_alpha, padding=False)]

            mult //= 2

        # Add final layers
        model += [
            act,
            SConv1d(n_filters, channels, last_kernel_size, batch_norm_feat=channels)
        ]
        
        model += [mx.tanh]
        self.model = nn.Sequential(*model)

    def __call__(self, z):
        y = self.model(z)
        return y
    