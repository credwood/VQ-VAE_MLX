# adapted from: https://github.com/facebookresearch/encodec/blob/main/encodec/utils.py
"""Audio utilities."""

import os
from pathlib import Path
import typing as tp

import librosa
import mlx.core as mx
import numpy as np
from scipy.io import wavfile

def convert_audio(wav: mx.array, sr: int, target_sr: int, target_channels: int):
    assert len(wav.shape) >= 2, "Audio tensor must have at least 2 dimensions"
    assert wav.shape[-2] in [1, 2], "Audio must be mono or stereo."
    *shape, channels, length = wav.shape
    if target_channels == 1:
        wav = wav.mean(-2, keepdims=True)
    elif target_channels == 2:
        wav = np.broadcast_to(wav, (*shape, target_channels, length))
    elif channels == 1:
        wav = np.broadcast_to(wav, (target_channels, -1))
    else:
        raise RuntimeError(f"Impossible to convert from {channels} to {target_channels}")
    wav = librosa.resample(wav, sr, target_sr)
    return wav


def save_audio(wav: tp.Union[mx.array, np.array], path: tp.Union[Path, str],
               sample_rate: int, rescale: bool = False):
    wav = np.array(wav)
    limit = 0.99
    mx = np.max(np.abs(wav))
    if rescale:
        wav = wav * min(limit / mx, 1)
    else:
        wav = np.clip(wav, a_min=-limit, a_max=limit)
    
    wavfile.write(str(path), rate=sample_rate, data=wav)