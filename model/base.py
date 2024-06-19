# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Base class for all quantizers.
Adapted from: https://github.com/facebookresearch/audiocraft/blob/main/audiocraft/quantization/base.py
"""

from dataclasses import dataclass, field
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

@dataclass
class QuantizedResult:
    x: mx.array
    codes: mx.array
    bandwith: mx.array
    penalty: Optional[mx.array] = None
    metrics: dict = field(default_factory=dict)

class BaseQuantizer(nn.Module):
    """
    Base quantizer class.
    """
    def __call__(self, x: mx.array, frame_rate: int) -> QuantizedResult:
        """
        Given input tensor x, returns first the quantized (or approximately quantized)
        representation along with quantized codes, bandwidth, and any penalty term for the loss.
        Finally, this returns a dict of metrics to update logging etc.
        Frame rate must be passed so that the bandwidth is properly computed.
        """
        raise NotImplementedError()

    def encode(self, x: mx.array) -> mx.array:
        """Encode a given input tensor with the specified sample rate at the given bandwidth."""
        raise NotImplementedError()

    def decode(self, codes: mx.array) -> mx.array:
        """Decode the given codes to the quantized representation."""
        raise NotImplementedError()

    @property
    def total_codebooks(self):
        """Total number of codebooks."""
        raise NotImplementedError()

    @property
    def num_codebooks(self):
        """Number of active codebooks."""
        raise NotImplementedError()

    def set_num_codebooks(self, n: int):
        """Set the number of active codebooks."""
        raise NotImplementedError()
