from collections import Counter
from typing import Optional, Any, Union, Callable
import math

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.losses import mse_loss
import numpy as np

import functional as F

from model.base import BaseQuantizer, QuantizedResult
from functional import l2normalize



def exists(val: Optional[Any]) -> bool:
    return val is not None


def default(val: Any, d: Any) -> Any:
    return val if exists(val) else d


def uniform_init(shape: list[int]):
    init = nn.init.he_uniform()
    return init(mx.zeros(shape))


def orthogonal_loss_fn(t: mx.array):
    # eq (2) from https://arxiv.org/abs/2112.00384
    n = t.shape[0]
    normed_codes = l2normalize(t)
    identity = mx.eye(n)
    cosine_sim = mx.matmul(normed_codes, normed_codes.transpose())
    return ((cosine_sim - identity) ** 2).sum() / (n ** 2)


class EuclideanCodebook(nn.Module):
    """Codebook with Euclidean distance.

    Args:
        dim (int): Dimension.
        codebook_size (int): Codebook size.
        kmeans_init (bool): Whether to use k-means to initialize the codebooks.
            If set to true, run the k-means algorithm on the first training batch and use
            the learned centroids as initialization.
        kmeans_iters (int): Number of iterations used for k-means algorithm at initialization.
        decay (float): Decay for exponential moving average over the codebooks.
        epsilon (float): Epsilon value for numerical stability.
        threshold_ema_dead_code (int): Threshold for dead code expiration. Replace any codes
            that have an exponential moving average cluster size less than the specified threshold with
            randomly selected vector from the current batch.
    """
    def __init__(
        self,
        dim: int,
        codebook_size: int,
        kmeans_init: int = False,
        kmeans_iters: int = 10,
        decay: float = 0.8,
        epsilon: float = 1e-5,
        threshold_ema_dead_code: int = 2,
    ):
        super().__init__()
        self.decay = decay
        init_fn: Union[Callable[..., mx.array], Any] = uniform_init
        embed = init_fn([codebook_size, dim])

        self.codebook_size = codebook_size

        self.epsilon = epsilon

        self.cluster_size = mx.zeros(codebook_size)
        self.embed = embed
        self.embed_avg = embed

    def preprocess(self, x):
        new_shape = 1
        for d in x.shape[:-1]:
            new_shape *= d
        x = x.reshape((new_shape, x.shape[-1]))
        return x

    def quantize(self, x):
        embed = self.embed.T
        dist = -(
            mx.power(x, 2).sum(1, keepdims=True)
            - 2 * mx.matmul(x, embed)
            + mx.power(embed, 2).sum(0, keepdims=True)
        )
        embed_ind = dist.argmax(axis=-1)
        return embed_ind

    def postprocess_emb(self, embed_ind, shape):
        return embed_ind.reshape(*shape[:-1])

    def dequantize(self, embed_ind):
        quantize = F.embedding(embed_ind, self.embed)
        return quantize

    def encode(self, x):
        shape = x.shape
        # pre-process
        x = self.preprocess(x)
        # quantize
        embed_ind = self.quantize(x)
        # post-process
        embed_ind = self.postprocess_emb(embed_ind, shape)
        return embed_ind

    def decode(self, embed_ind):
        quantize = self.dequantize(embed_ind)
        return quantize


    def __call__(self, x):
        shape = x.shape
        x = self.preprocess(x)
        embed_ind = self.quantize(x)
        embed_ind = self.postprocess_emb(embed_ind, shape)
        quantize = self.dequantize(embed_ind)
        return quantize, embed_ind

class VectorQuantization(nn.Module):
    """Vector quantization implementation.
    Currently supports only euclidean distance.

    Args:
        dim (int): Dimension
        codebook_size (int): Codebook size
        codebook_dim (int): Codebook dimension. If not defined, uses the specified dimension in dim.
        decay (float): Decay for exponential moving average over the codebooks.
        epsilon (float): Epsilon value for numerical stability.
        kmeans_init (bool): Whether to use kmeans to initialize the codebooks.
        kmeans_iters (int): Number of iterations used for kmeans initialization.
        threshold_ema_dead_code (int):
        channels_last (bool): Channels are the last dimension in the input tensors.
        commitment_weight (float): Weight for commitment loss.
        orthogonal_reg_weight (float): Orthogonal regularization weights.
        orthogonal_reg_active_codes_only (bool): Apply orthogonal regularization only on active codes.
        orthogonal_reg_max_codes (optional int): Maximum number of codes to consider
            for orthogonal regularization.
        threshold_ema_dead_code (int): Threshold for dead code expiration. Replace any codes
            that have an exponential moving average cluster size less than the specified threshold with
            randomly selected vector from the current batch.
    """
    def __init__(
        self,
        dim: int,
        codebook_size: int,
        codebook_dim: Optional[int] = None,
        decay: float = 0.8,
        epsilon: float = 1e-5,
        kmeans_init: bool = False,
        kmeans_iters: int = 10,
        threshold_ema_dead_code: int = 2,
        channels_last: bool = False,
        commitment_weight: float = 1.,
        orthogonal_reg_weight: float = 0.0,
        orthogonal_reg_active_codes_only: bool = False,
        orthogonal_reg_max_codes: Optional[int] = None,
    ):
        super().__init__()
        _codebook_dim: int = default(codebook_dim, dim)

        requires_projection = _codebook_dim != dim
        self.project_in = nn.Linear(dim, _codebook_dim) if requires_projection else None
        self.project_out = nn.Linear(_codebook_dim, dim) if requires_projection else None

        self.epsilon = epsilon
        self.commitment_weight = commitment_weight

        self.orthogonal_reg_weight = orthogonal_reg_weight
        self.orthogonal_reg_active_codes_only = orthogonal_reg_active_codes_only
        self.orthogonal_reg_max_codes = orthogonal_reg_max_codes

        self._codebook = EuclideanCodebook(dim=dim, codebook_size=codebook_size,
                                           kmeans_init=kmeans_init, kmeans_iters=kmeans_iters,
                                           decay=decay, epsilon=epsilon,
                                           threshold_ema_dead_code=threshold_ema_dead_code,
                                           )
        self.codebook_size = codebook_size

        self.channels_last = channels_last

    @property
    def codebook(self):
        return self._codebook.embed

    @property
    def inited(self):
        return self._codebook.inited

    def _preprocess(self, x):
        # TODO get rid
        if self.channels_last:
            return x.transpose(0,2,1)
        return x

    def _postprocess(self, quantize):
        if self.channels_last:
            quantize = quantize.transpose(0,2,1)
        return quantize

    def encode(self, x):
        x = self._preprocess(x)
        if self.project_in is not None:
            x = self.project_in(x)
        embed_in = self._codebook.encode(x)
        return embed_in

    def decode(self, embed_ind):
        quantize = self._codebook.decode(embed_ind)
        if self.project_out is not None:
            quantize = self.project_out(quantize)
        quantize = self._postprocess(quantize)
        return quantize

    def __call__(self, x):
        x = self._preprocess(x)

        if self.project_in is not None:
            x = self.project_in(x)

        quantize, embed_ind = self._codebook(x)
        
        if self.training:
            quantize = x + (quantize - x)

        loss = mx.array([0.0])

        if self.training:
            if self.commitment_weight > 0:
                commit_loss = mse_loss(quantize, x)
                loss = loss + commit_loss * self.commitment_weight

            if self.orthogonal_reg_weight > 0:
                codebook = self.codebook

                if self.orthogonal_reg_active_codes_only:
                    # only calculate orthogonal loss for the activated codes for this batch
                    unique_code_ids = mx.array(np.unique(np.array(embed_ind)))
                    codebook = codebook[unique_code_ids]

                num_codes = codebook.shape[0]
                if exists(self.orthogonal_reg_max_codes) and num_codes > self.orthogonal_reg_max_codes:
                    rand_ids = mx.array(np.random.permutation(num_codes))[:self.orthogonal_reg_max_codes]
                    codebook = codebook[rand_ids]

                orthogonal_reg_loss = orthogonal_loss_fn(codebook)
                loss = loss + orthogonal_reg_loss * self.orthogonal_reg_weight

        if self.project_out is not None:
            quantize = self.project_out(quantize)
        quantize = self._postprocess(quantize)

        return quantize, embed_ind, loss


class ResidualVectorQuantization(nn.Module):
    """Residual vector quantization implementation.

    Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf
    """
    def __init__(self, *, num_quantizers, **kwargs):
        super().__init__()
        self.layers = [
            VectorQuantization(**kwargs) for _ in range(num_quantizers)
        ]

    def __call__(self, x, n_q: Optional[int] = None):
        quantized_out = 0.0
        residual = x

        all_losses = []
        all_indices = []

        n_q = n_q or len(self.layers)

        for i, layer in enumerate(self.layers[:n_q]):
            quantized, indices, loss = layer(residual)
            residual = residual - quantized
            quantized_out = quantized_out + quantized
            all_indices.append(indices)
            all_losses.append(loss)

        if self.training:
            # Solving subtle bug with STE and RVQ: https://github.com/facebookresearch/encodec/issues/25
            quantized_out = x + (quantized_out - x)

        out_losses, out_indices = map(mx.stack, (all_losses, all_indices))
        return quantized_out, out_indices, out_losses

    def encode(self, x: mx.array, n_q: Optional[int] = None) -> mx.array:
        residual = x
        all_indices = []
        n_q = n_q or len(self.layers)
        for layer in self.layers[:n_q]:
            indices = layer.encode(residual)
            quantized = layer.decode(indices)
            residual = residual - quantized
            all_indices.append(indices)
        out_indices = mx.stack(all_indices)
        return out_indices

    def decode(self, q_indices: mx.array) -> mx.array:
        quantized_out = mx.array(0.0)
        for i, indices in enumerate(q_indices):
            layer = self.layers[i]
            quantized = layer.decode(indices)
            quantized_out = quantized_out + quantized
        return quantized_out
    

class ResidualVectorQuantizer(BaseQuantizer):
    """Residual Vector Quantizer.

    Args:
        dimension (int): Dimension of the codebooks.
        n_q (int): Number of residual vector quantizers used.
        q_dropout (bool): Random quantizer drop out at train time.
        bins (int): Codebook size.
        decay (float): Decay for exponential moving average over the codebooks.
        kmeans_init (bool): Whether to use kmeans to initialize the codebooks.
        kmeans_iters (int): Number of iterations used for kmeans initialization.
        threshold_ema_dead_code (int): Threshold for dead code expiration. Replace any codes
            that have an exponential moving average cluster size less than the specified threshold with
            randomly selected vector from the current batch.
        orthogonal_reg_weight (float): Orthogonal regularization weights.
        orthogonal_reg_active_codes_only (bool): Apply orthogonal regularization only on active codes.
        orthogonal_reg_max_codes (optional int): Maximum number of codes to consider.
            for orthogonal regularization.
    """
    def __init__(
        self,
        dimension: int = 128,
        n_q: int = 1,
        q_dropout: bool = False,
        bins: int = 1024,
        decay: float = 0.99,
        kmeans_init: bool = True,
        kmeans_iters: int = 10,
        threshold_ema_dead_code: int = 2,
        orthogonal_reg_weight: float = 0.0,
        orthogonal_reg_active_codes_only: bool = False,
        orthogonal_reg_max_codes: Optional[int] = None,
    ):
        super().__init__()
        self.max_n_q = n_q
        self.n_q = n_q
        self.q_dropout = q_dropout
        self.dimension = dimension
        self.bins = bins
        self.decay = decay
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.orthogonal_reg_weight = orthogonal_reg_weight
        self.orthogonal_reg_active_codes_only = orthogonal_reg_active_codes_only
        self.orthogonal_reg_max_codes = orthogonal_reg_max_codes

        self.vq = ResidualVectorQuantization(
            dim=self.dimension,
            codebook_size=self.bins,
            num_quantizers=self.n_q,
            decay=self.decay,
            kmeans_init=self.kmeans_init,
            kmeans_iters=self.kmeans_iters,
            threshold_ema_dead_code=self.threshold_ema_dead_code,
            orthogonal_reg_weight=self.orthogonal_reg_weight,
            orthogonal_reg_active_codes_only=self.orthogonal_reg_active_codes_only,
            orthogonal_reg_max_codes=self.orthogonal_reg_max_codes,
            channels_last=False,
        )

    def __call__(self, x: mx.array, frame_rate: int):
        n_q = self.n_q
        if self.training and self.q_dropout:
            n_q = int(mx.random.randint(1, self.n_q + 1, (1,)).item())
        bw_per_q = math.log2(self.bins) * frame_rate / 1000
        quantized, codes, commit_loss = self.vq(x, n_q=n_q)
        codes = codes.transpose(1, 0, 2)
        # codes is [B, K, T], with T frames, K nb of codebooks.
        bw = mx.array(n_q * bw_per_q)
        return QuantizedResult(quantized, codes, bw, penalty=mx.mean(commit_loss))

    def encode(self, x: mx.array) -> mx.array:
        """Encode a given input tensor with the specified frame rate at the given bandwidth.
        The RVQ encode method sets the appropriate number of quantizer to use
        and returns indices for each quantizer.
        """
        n_q = self.n_q
        codes = self.vq.encode(x, n_q=n_q)
        codes = codes.transpose(1, 0, 2)
        # codes is [B, K, T], with T frames, K nb of codebooks.
        return codes

    def decode(self, codes: mx.array) -> mx.array:
        """Decode the given codes to the quantized representation."""
        # codes is [B, K, T], with T frames, K nb of codebooks, vq.decode expects [K, B, T].
        codes = codes.transpose(1, 0, 2)
        quantized = self.vq.decode(codes)
        return quantized

    @property
    def total_codebooks(self):
        return self.max_n_q

    @property
    def num_codebooks(self):
        return self.n_q

    def set_num_codebooks(self, n: int):
        assert n > 0 and n <= self.max_n_q
        self.n_q = n
