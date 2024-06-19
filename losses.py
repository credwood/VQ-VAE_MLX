# Adapted from: https://github.com/facebookresearch/audiocraft/blob/main/audiocraft/losses/stftloss.py
# adapted from https://github.com/ml-explore/mlx-examples/blob/main/whisper/mlx_whisper/audio.py

import typing as tp

import numpy as np
import mlx.core as mx
import mlx.nn as nn

import functional as F
from model.conv_wrappers import pad_for_conv1d


import os
from functools import lru_cache
from subprocess import CalledProcessError, run
from typing import Union

import mlx.core as mx
import numpy as np



def load_audio(file: str, sr: int):
    """
    Open an audio file and read as mono waveform, resampling as necessary

    Parameters
    ----------
    file: str
        The audio file to open

    sr: int
        The sample rate to resample the audio if necessary

    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """

    # This launches a subprocess to decode audio while down-mixing
    # and resampling as necessary.  Requires the ffmpeg CLI in PATH.
    # fmt: off
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads", "0",
        "-i", file,
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(sr),
        "-"
    ]
    # fmt: on
    try:
        out = run(cmd, capture_output=True, check=True).stdout
    except CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return mx.array(np.frombuffer(out, np.int16)).flatten().astype(mx.float32) / 32768.0


def pad_or_trim(array, length: int, *, axis: int = -1):
    """
    Pad or trim the audio array to N_SAMPLES, as expected by the encoder.
    """
    if array.shape[axis] > length:
        sl = [slice(None)] * array.ndim
        sl[axis] = slice(0, length)
        array = array[tuple(sl)]

    if array.shape[axis] < length:
        pad_widths = [(0, 0)] * array.ndim
        pad_widths[axis] = (0, length - array.shape[axis])
        array = mx.pad(array, pad_widths)

    return array


def mel_filters(n_mels: int, n_fft: int) -> mx.array:
    """
    load the mel filterbank matrix for projecting STFT into a Mel spectrogram.
    Allows decoupling librosa dependency; saved using:

        np.savez_compressed(
            "mel_filters.npz",
            mel_80=librosa.filters.mel(sr=41000, n_fft={n_fft}, n_mels=80),
        )
    """
    assert n_mels in {80, 128}, f"Unsupported n_mels: {n_mels}"

    filename = os.path.join(os.path.dirname(__file__), "assets", f"mel_{n_fft}.npz")
    return mx.load(filename)[f"mel_{n_mels}"]


def hanning(size):
    return mx.array(np.hanning(size + 1)[:-1])

"""
def stft(x, window, nperseg=256, noverlap=None, nfft=None, axis=-1, pad_mode="reflect"):
    if nfft is None:
        nfft = nperseg
    if noverlap is None:
        noverlap = nfft // 4

    strides = [noverlap, 1]
    t = (x.shape[-1] - nperseg + noverlap) // noverlap
    shape = [x.shape[0], t, nfft]
    x = mx.as_strided(x, shape=shape, strides=strides)
    return mx.fft.rfft(x * window)
"""

def stft(x, window, nperseg=256, noverlap=None, nfft=None, axis=-1, pad_mode="reflect"):
    if nfft is None:
        nfft = nperseg
    if noverlap is None:
        noverlap = nfft // 4

    def _pad(x, padding, pad_mode="constant"):
        if pad_mode == "constant":
            return mx.pad(x, [(0, 0), (padding, padding)])
        elif pad_mode == "reflect":
            prefix = x[:, 1 : padding + 1][::-1]
            suffix = x[:, -(padding + 1) : -1][::-1]
            return mx.concatenate([prefix, x, suffix], axis=-1)
        else:
            raise ValueError(f"Invalid pad_mode {pad_mode}")

    padding = nperseg // 2
    x = _pad(x, padding, pad_mode)

    strides = [noverlap, 1]
    t = (x.shape[-1] - nperseg + noverlap) // noverlap
    shape = [x.shape[0], t, nfft]
    x = mx.as_strided(x, shape=shape, strides=strides)
    x = mx.fft.rfft(x * window)
    return x

def log_mel_spectrogram(
    audio: Union[str, np.ndarray, mx.array],
    n_mels: int = 80,
    n_fft: int = 1024,
    hop_length: int = 256,
    sample_rate: int = 44100,
):
    """
    Compute the log-Mel spectrogram of

    Parameters
    ----------
    audio: Union[str, np.ndarray, mx.array], shape = (*)
        The path to audio or either a NumPy or mlx array containing the audio waveform in 16 kHz

    n_mels: int
        The number of Mel-frequency filters, only 80 is supported

    padding: int
        Number of zero samples to pad to the right

    Returns
    -------
    mx.array, shape = (batch_size, num_channels, n_frames, 80)
        An  array that contains the Mel spectrogram
    """
    window = hanning(n_fft)
    B, C, T = audio.shape
    audio = audio.reshape(-1, T)
    with mx.stream(mx.cpu):
        freqs = stft(audio, window, nperseg=n_fft, noverlap=hop_length)
        magnitudes = freqs[:, :-1, :].abs().square()
        filters = mel_filters(n_mels, n_fft)
        mel_spec = magnitudes @ filters.T

    log_spec = mx.maximum(mel_spec, 1e-10).log10()
    log_spec = mx.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0

    log_spec = log_spec.reshape(B, C, log_spec.shape[-2], log_spec.shape[-1])
    return log_spec


class MelSpectrogramWrapper(nn.Module):
    """Wrapper around MelSpectrogram torchaudio transform providing proper padding
    and additional post-processing including log scaling.

    Args:
        n_mels (int): Number of mel bins.
        n_fft (int): Number of fft.
        hop_length (int): Hop size.
        win_length (int): Window length.
        n_mels (int): Number of mel bins.
        sample_rate (int): Sample rate.
        f_min (float or None): Minimum frequency.
        f_max (float or None): Maximum frequency.
        log (bool): Whether to scale with log.
        floor_level (float): Floor level based on human perception (default=1e-5).
    """
    def __init__(self, n_fft: int = 1024, hop_length: int = 256, win_length: tp.Optional[int] = None,
                 n_mels: int = 80, sample_rate: float = 22050, f_min: float = 0.0, f_max: tp.Optional[float] = None,
                 log: bool = True, floor_level: float = 1e-5):
        super().__init__()
        self.n_fft = n_fft
        hop_length = int(hop_length)
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length=win_length
        self.f_min=f_min
        self.f_max=f_max
        self.window="hann"
        self.center=False

        self.floor_level = floor_level
        self.log = log

    def __call__(self, x):
        p = int((self.n_fft - self.hop_length) // 2)
        if len(x.shape) == 2:
            x = mx.expand_dims(x, -1)
        #x = mx.pad(x, ((0,0), (p, p), (0,0)))
        # Make sure that all the frames are full.
        # The combination of `pad_for_conv1d` and the above padding
        # will make the output of size ceil(T / hop).
        #x = pad_for_conv1d(x, self.n_fft, self.hop_length)
        x = x.transpose(0, 2, 1)
        
        mel_spec = log_mel_spectrogram(x, n_mels=self.n_mels, sample_rate=self.sample_rate, 
                                    n_fft=self.n_fft, hop_length=self.hop_length)
        B , C, frame, freqs = mel_spec.shape
        return mel_spec.reshape(B, C * freqs, frame)


class MelSpectrogramL1Loss(nn.Module):
    """L1 Loss on MelSpectrogram.

    Args:
        sample_rate (int): Sample rate.
        n_fft (int): Number of fft.
        hop_length (int): Hop size.
        win_length (int): Window length.
        n_mels (int): Number of mel bins.
        f_min (float or None): Minimum frequency.
        f_max (float or None): Maximum frequency.
        log (bool): Whether to scale with log.
        floor_level (float): Floor level value based on human perception (default=1e-5).
    """
    def __init__(self, sample_rate: int, n_fft: int = 400, hop_length: int = 256, win_length: int = 1024,
                 n_mels: int = 80, f_min: float = 0.0, f_max: tp.Optional[float] = None,
                 log: bool = True, floor_level: float = 1e-5):
        super().__init__()
    
        self.melspec = MelSpectrogramWrapper(n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                                             n_mels=n_mels, sample_rate=sample_rate, f_min=f_min, f_max=f_max,
                                             log=log, floor_level=floor_level)

    def loss(self, x, y):
        s_x = self.melspec(x)
        s_y = self.melspec(y)
        return nn.losses.l1_loss(s_x, s_y)


class MultiScaleMelSpectrogramLoss(nn.Module):
    """Multi-Scale spectrogram loss (msspec).

    Args:
        sample_rate (int): Sample rate.
        range_start (int): Power of 2 to use for the first scale.
        range_stop (int): Power of 2 to use for the last scale.
        n_mels (int): Number of mel bins.
        f_min (float): Minimum frequency.
        f_max (float or None): Maximum frequency.
        alphas (bool): Whether to use alphas as coefficients or not.
        floor_level (float): Floor level value based on human perception (default=1e-5).
    """
    def __init__(self, sample_rate: int, range_start: int = 6, range_end: int = 11,
                 n_mels: int = 80, f_min: float = 0.0, f_max: tp.Optional[float] = None,
                 alphas: bool = True, floor_level: float = 1e-5):
        super().__init__()
        l1s = list()
        l2s = list()
        self.alphas = list()
        self.total = 0
        for i in range(range_start, range_end):
            l1s.append(
                MelSpectrogramWrapper(n_fft=2 ** i, hop_length=(2 ** i) / 4, win_length=2 ** i,
                                      n_mels=n_mels, sample_rate=sample_rate, f_min=f_min, f_max=f_max,
                                      log=False, floor_level=floor_level))
            l2s.append(
                MelSpectrogramWrapper(n_fft=2 ** i, hop_length=(2 ** i) / 4, win_length=2 ** i,
                                      n_mels=n_mels, sample_rate=sample_rate, f_min=f_min, f_max=f_max,
                                      log=True, floor_level=floor_level))
            if alphas:
                self.alphas.append(np.sqrt(2 ** i - 1))
            else:
                self.alphas.append(1)
            self.total += self.alphas[-1] + 1

        self.l1s = l1s
        self.l2s = l2s

    def __call__(self, x, y):
        loss = 0.0
        for i in range(len(self.alphas)):
            s_x_1 = self.l1s[i](x)
            s_y_1 = self.l1s[i](y)
            s_x_2 = self.l2s[i](x)
            s_y_2 = self.l2s[i](y)
            mse = nn.losses.mse_loss(s_x_2, s_y_2)
            l1_err = nn.losses.l1_loss(s_x_1, s_y_1)
            loss += l1_err + mse #getting a compilation error multiplying mse by self.alphas[i], taking it out for now
        return loss
