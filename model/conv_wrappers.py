# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# form: https://github.com/facebookresearch/encodec/blob/main/encodec/modules/conv.py

"""Convolutional layers wrappers and utilities."""

import math
import typing as tp
import warnings

import mlx.core as mx
import mlx.nn as nn



def get_extra_padding_for_conv1d(x: mx.array, kernel_size: int, stride: int,
                                 padding_total: int = 0) -> int:
    """See `pad_for_conv1d`.
    """
    length = x.shape[-1]
    n_frames = (length - kernel_size + padding_total) / stride + 1
    ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
    return ideal_length - length


def pad_for_conv1d(x: mx.array, kernel_size: int, stride: int, padding_total: int = 0):
    """Pad for a convolution to make sure that the last window is full.
    Extra padding is added at the end. This is required to ensure that we can rebuild
    an output of the same length, as otherwise, even with padding, some time steps
    might get removed.
    For instance, with total padding = 4, kernel size = 4, stride = 2:
        0 0 1 2 3 4 5 0 0   # (0s are padding)
        1   2   3           # (output frames of a convolution, last 0 is never used)
        0 0 1 2 3 4 5 0     # (output of tr. conv., but pos. 5 is going to get removed as padding)
            1 2 3 4         # once you removed padding, we are missing one time step !
    """
    extra_padding = get_extra_padding_for_conv1d(x, kernel_size, stride, padding_total)
    return mx.pad(x,((0,0), (0, extra_padding), (0,0)))



def pad1d(x: mx.array, paddings: tp.Tuple[int, int], mode: str = 'zero', value: float = 0.):
    """Tiny wrapper around F.pad, just to allow for reflect padding on small input.
    If this is the case, we insert extra 0 padding to the right before the reflection happen.
    """
    length = x.shape[1]
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0, (padding_left, padding_right)
    if mode == 'reflect':
        max_pad = max(padding_left, padding_right)
        extra_pad = 0
        if length <= max_pad:
            extra_pad = max_pad - length + 1
            x = mx.pad(x, ((0,0), (0, extra_pad), (0,0)))
        padded = mx.pad(x, ((0,0), (padding_left, padding_right), (0,0)))
        end = padded.shape[1] - extra_pad
        return padded[:,:end, :]
    else:
        return mx.pad(x, pad_with=((0,0), (padding_left, padding_right), (0,0)))


def unpad1d(x: mx.array, paddings: tp.Tuple[int, int]):
    """Remove padding from x, handling properly zero padding. Only for 1d!"""
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0, (padding_left, padding_right)
    assert (padding_left + padding_right) <= x.shape[1]
    end = x.shape[1] - padding_right
    return x[ :, padding_left:end, :]


#from: https://github.com/ml-explore/mlx-examples/blob/main/cvae/vae.py

def upsample_nearest(x, scale: int = 2):
    B, W, C = x.shape
    x = mx.broadcast_to(x[:, :, None, :], (B, W, scale, C))
    x = x.reshape(B, W * scale, C)
    return x


class UpsamplingConv1d(nn.Module):
    """
    A convolutional layer that upsamples the input by a factor of 2. MLX does
    not yet support transposed convolutions, so we approximate them with
    nearest neighbor upsampling followed by a convolution. This is similar to
    the approach used in the original U-Net.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, scale, dilation=1):
        super().__init__()
        self.conv = SConv1d(
            in_channels, out_channels, kernel_size, stride=stride
        )
        self.scale = scale

    def __call__(self, x):
        x = upsample_nearest(x, scale = self.scale)
        x = self.conv(x)
        return x
 

class NormConv1d(nn.Module):
    """Wrapper around Conv1d and normalization applied to this conv
    to provide a uniform interface across normalization approaches.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, batch_norm_feat, dilation=1):
        super().__init__()

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, dilation=dilation)
        self.norm = nn.BatchNorm(batch_norm_feat)
    
    def __call__(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return x


class NormConvTranspose1d(nn.Module):
    """Wrapper around ConvTranspose1d and normalization applied to this conv
    to provide a uniform interface across normalization approaches.
    """
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, stride: int = 1, dilation: int = 1, batch_norm_feat: int = 1, scale: int=2):
        super().__init__()
        self.convtr = UpsamplingConv1d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, scale=scale)
        self.norm = nn.BatchNorm(batch_norm_feat)

    def __call__(self, x):
        x = self.convtr(x)
        x = self.norm(x)
        return x


class SConv1d(nn.Module):
    """Conv1d with some builtin handling of asymmetric or causal padding
    and normalization.
    """
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, stride: int = 1, dilation: int = 1, batch_norm_feat: int = 1):
        super().__init__()
        # warn user on unusual setup between dilation and stride
        if stride > 1 and dilation > 1:
            warnings.warn('SConv1d has been initialized with stride > 1 and dilation > 1'
                          f' (kernel_size={kernel_size} stride={stride}, dilation={dilation}).')
        self.conv = NormConv1d(in_channels, out_channels, kernel_size, stride,
                               dilation=dilation, batch_norm_feat=batch_norm_feat)
        self.pad_mode = "reflect"
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation 

    def __call__(self, x):
        B, C, T = x.shape
        kernel_size = self.kernel_size
        stride = self.stride
        dilation = self.dilation
        kernel_size = (kernel_size - 1) * dilation + 1  # effective kernel size with dilations
        padding_total = kernel_size - stride
        extra_padding = get_extra_padding_for_conv1d(x, kernel_size, stride, padding_total)
        
        # Asymmetric padding required for odd strides
        padding_right = padding_total // 2
        padding_left = padding_total - padding_right
        x = pad1d(x, (padding_left, padding_right + extra_padding), mode=self.pad_mode)
        return self.conv(x)


class SConvTranspose1d(nn.Module):
    """ConvTranspose1d with some builtin handling of asymmetric padding
    and normalization.
    """
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, stride: int = 1, scale: int = 2, batch_norm_feat: int=1):
        super().__init__()
        self.convtr = NormConvTranspose1d(in_channels, out_channels, kernel_size, stride, scale=scale, batch_norm_feat=batch_norm_feat)
        self.trim_right_ratio = 1
        self.kernel_size = kernel_size
        self.stride = stride

    def __call__(self, x):
        kernel_size = self.kernel_size
        stride = self.stride
        padding_total = kernel_size - stride

        y = self.convtr(x)

        # We will only trim fixed padding. Extra padding from `pad_for_conv1d` would be
        # removed at the very end, when keeping only the right length for the output,
        # as removing it here would require also passing the length at the matching layer
        # in the encoder.

        # Asymmetric padding required for odd strides
        padding_right = padding_total // 2
        padding_left = padding_total - padding_right
        y = unpad1d(y, (padding_left, padding_right))
        return y