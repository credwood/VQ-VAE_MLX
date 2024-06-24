# adapted from https://github.com/facebookresearch/encodec/blob/main/encodec/model.py

import math
import time
from typing import List, Optional, Tuple
from functools import partial

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten
from mlx.nn import losses

from model.conv_net import ConvEncoder, ConvDecoder
from model.vq import ResidualVectorQuantizer


EncodedFrame = Tuple[mx.array, Optional[mx.array]]

class ConvVQ(nn.Module):
    """Loosely based on EnCodec model operating on the raw waveform.
    Args:
        target_bandwidths (list of float): Target bandwidths.
        encoder (nn.Module): Encoder network.
        decoder (nn.Module): Decoder network.
        quantizer (nn.Module): quantizer network.
        sample_rate (int): Audio sample rate.
        channels (int): Number of audio channels.
        normalize (bool): Whether to apply audio normalization.
        segment (float or None): segment duration in sec. when doing overlap-add.
        overlap (float): overlap between segment, given as a fraction of the segment duration.
        name (str): name of the model, used as metadata when compressing audio.
    """
    def __init__(self,
                 encoder = ConvEncoder(),
                 decoder = ConvDecoder(),
                 quantizer = ResidualVectorQuantizer(),
                 sample_rate: int = 44100,
                 channels: int = 1,
                 normalize: bool = False,
                 segment: Optional[float] = None,
                 overlap: float = 0.01,
                 name: str = 'convvq',):
        super().__init__()
        self.bandwidth: Optional[float] = None
        self.encoder = encoder
        self.quantizer = quantizer
        self.decoder = decoder
        self.sample_rate = sample_rate
        self.channels = channels
        self.normalize = normalize
        self.segment = segment
        self.overlap = overlap
        self.frame_rate = math.ceil(self.sample_rate / np.prod(self.encoder.ratios))
        self.name = name
        self._qt_vals = []
        self._qt_loss = mx.zeros((1))
        self._encoder_loss = mx.zeros((1))
        self._temporary_loss = []
        self._emb_for_loss = mx.zeros((1))

        self.bits_per_codebook = int(math.log2(self.quantizer.bins))
        assert 2 ** self.bits_per_codebook == self.quantizer.bins, \
            "quantizer bins must be a power of 2."

    @property
    def segment_length(self) -> Optional[int]:
        if self.segment is None:
            return None
        return int(self.segment * self.sample_rate)

    @property
    def segment_stride(self) -> Optional[int]:
        segment_length = self.segment_length
        if segment_length is None:
            return None
        return max(1, int((1 - self.overlap) * segment_length))

    def encode(self, x: mx.array) -> List[EncodedFrame]:
        """Given a tensor `x`, returns a list of frames containing
        the discrete encoded codes for `x`, along with rescaling factors
        for each segment, when `self.normalize` is True.

        Each frames is a tuple `(codebook, scale)`, with `codebook` of
        shape `[B, K, T]`, with `K` the number of codebooks.
        """
        assert len(x.shape) == 3
        _, length, channels  = x.shape
        assert channels > 0 and channels <= 2
  
        segment_length = self.segment_length
        if segment_length is None:
            segment_length = length
            stride = length
        else:
            stride = self.segment_stride  # type: ignore
            assert stride is not None

        encoded_frames: List[EncodedFrame] = []
        for offset in range(0, length, stride):
            frame = x[:, offset: offset + segment_length, :]
            encoded_frames.append(self._encode_frame(frame))
        if self.training:
            self._qt_loss = mx.array(sum(self._qt_vals)/len(self._qt_vals))
            
        return encoded_frames

    def _encode_frame(self, x: mx.array) -> EncodedFrame:
        length = x.shape[1]
        duration = length / self.sample_rate
        assert self.segment is None or duration <= 1e-5 + self.segment

        if self.normalize:
            mono = x.mean(axis=2, keepdims=True)
            volume = mx.power(mono, 2).mean(axis=1, keepdims=True).sqrt()
            scale = 1e-8 + volume
            x = x / scale
            scale = scale.reshape(-1, 1)
        else:
            scale = None

        emb = self.encoder(x)
        if self.training:
            quantized_results = self.quantizer(emb, frame_rate=self.frame_rate)
            codes = quantized_results.codes
            self._qt_vals.append(quantized_results.penalty)
            self._emb_for_loss = emb
            
        else:
            codes = self.quantizer.encode(emb)
        return codes, scale

    def decode(self, encoded_frames: List[EncodedFrame]) -> mx.array:
        """Decode the given frames into a waveform.
        Note that the output might be a bit bigger than the input. In that case,
        any extra steps at the end can be trimmed.
        """
        segment_length = self.segment_length
        if segment_length is None:
            assert len(encoded_frames) == 1
            return self._decode_frame(encoded_frames[0])

        frames = [self._decode_frame(frame) for frame in encoded_frames]
        if self.training:
            self._encoder_loss = mx.stop_gradient(mx.array(sum(self._temporary_loss)/len(self._temporary_loss)))
            self._temporary_loss = []
        return frames

    def _decode_frame(self, encoded_frame: EncodedFrame) -> mx.array:
        codes, scale = encoded_frame
        emb = self.quantizer.decode(codes)
        if self.training:
            emb = mx.stop_gradient(emb)
            emb_frame_loss = losses.mse_loss(emb, self._emb_for_loss)
            self._temporary_loss.append(emb_frame_loss)

        out = self.decoder(emb)
        if scale is not None:
            out = out * scale.reshape(-1, 1, 1)            

        return out

    def __call__(self, x: mx.array) -> mx.array:
        frames = self.encode(x)
        return self.decode(frames)[:, :x.shape[1], :]



class ConvDUMMY(nn.Module):
    """Loosely based on EnCodec model operating on the raw waveform.
    Args:
        target_bandwidths (list of float): Target bandwidths.
        encoder (nn.Module): Encoder network.
        decoder (nn.Module): Decoder network.
        sample_rate (int): Audio sample rate.
        channels (int): Number of audio channels.
        normalize (bool): Whether to apply audio normalization.
        segment (float or None): segment duration in sec. when doing overlap-add.
        overlap (float): overlap between segment, given as a fraction of the segment duration.
        name (str): name of the model, used as metadata when compressing audio.
    """
    def __init__(self,
                 encoder = ConvEncoder(),
                 decoder = ConvDecoder(),
                 sample_rate: int = 44100,
                 channels: int = 1,
                 normalize: bool = False,
                 segment: Optional[float] = None,
                 overlap: float = 0.01,
                 name: str = 'convvq',):
        super().__init__()
        self.bandwidth: Optional[float] = None
        self.encoder = encoder
        self.decoder = decoder
        self.sample_rate = sample_rate
        self.channels = channels
        self.normalize = normalize
        self.segment = segment
        self.overlap = overlap
        self.frame_rate = math.ceil(self.sample_rate / np.prod(self.encoder.ratios))
        self.name = name

    @property
    def segment_length(self) -> Optional[int]:
        if self.segment is None:
            return None
        return int(self.segment * self.sample_rate)

    @property
    def segment_stride(self) -> Optional[int]:
        segment_length = self.segment_length
        if segment_length is None:
            return None
        return max(1, int((1 - self.overlap) * segment_length))

    def encode(self, x: mx.array) -> List[EncodedFrame]:
        """Given a tensor `x`, returns a list of frames containing
        the discrete encoded codes for `x`, along with rescaling factors
        for each segment, when `self.normalize` is True.

        Each frames is a tuple `(codebook, scale)`, with `codebook` of
        shape `[B, K, T]`, with `K` the number of codebooks.
        """
        assert len(x.shape) == 3
        _, length, channels  = x.shape
        assert channels > 0 and channels <= 2
        segment_length = self.segment_length
        if segment_length is None:
            segment_length = length
            stride = length
        else:
            stride = self.segment_stride  # type: ignore
            assert stride is not None

        encoded_frames: List[EncodedFrame] = []
        for offset in range(0, length, stride):
            frame = x[:, offset: offset + segment_length, :]
            encoded_frames.append(self._encode_frame(frame))
        return encoded_frames

    def _encode_frame(self, x: mx.array) -> EncodedFrame:
        length = x.shape[1]
        duration = length / self.sample_rate
        assert self.segment is None or duration <= 1e-5 + self.segment

        if self.normalize:
            mono = x.mean(axis=2, keepdims=True)
            volume = mx.power(mono, 2).mean(axis=1, keepdims=True).sqrt()
            scale = 1e-8 + volume
            x = x / scale
            scale = scale.reshape(-1, 1)
        else:
            scale = None

        emb = self.encoder(x)
        return emb

    def decode(self, encoded_frames: List[EncodedFrame]) -> mx.array:
        """Decode the given frames into a waveform.
        Note that the output might be a bit bigger than the input. In that case,
        any extra steps at the end can be trimmed.
        """
        segment_length = self.segment_length
        if segment_length is None:
            assert len(encoded_frames) == 1
            return self._decode_frame(encoded_frames[0])

        frames = [self._decode_frame(frame) for frame in encoded_frames]
        return frames

    def _decode_frame(self, encoded_frame: EncodedFrame) -> mx.array:
        out = self.decoder(encoded_frame)
        return out

    def __call__(self, x: mx.array) -> mx.array:
        frames = self.encode(x)
        return self.decode(frames)[:, :x.shape[1], :]
