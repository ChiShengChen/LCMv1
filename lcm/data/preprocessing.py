from __future__ import annotations
"""EEG preprocessing: resample, filter, segment, re-reference."""

import numpy as np
from scipy.signal import butter, filtfilt, resample_poly
from math import gcd


def resample_signal(
    data: np.ndarray, orig_sr: int, target_sr: int
) -> np.ndarray:
    """Resample EEG data to target sample rate.

    Args:
        data: [C, T] — EEG data
        orig_sr: original sample rate in Hz
        target_sr: target sample rate in Hz

    Returns:
        resampled: [C, T'] — resampled EEG data
    """
    if orig_sr == target_sr:
        return data
    g = gcd(orig_sr, target_sr)
    up = target_sr // g
    down = orig_sr // g
    return resample_poly(data, up, down, axis=-1)


def average_rereference(data: np.ndarray) -> np.ndarray:
    """Apply average re-referencing.

    Args:
        data: [C, T] — EEG data

    Returns:
        rereferenced: [C, T]
    """
    mean = data.mean(axis=0, keepdims=True)
    return data - mean


def bandpass_filter(
    data: np.ndarray,
    low_freq: float,
    high_freq: float,
    sample_rate: int,
    order: int = 5,
) -> np.ndarray:
    """Apply bandpass filter to EEG data using Butterworth filter.

    Args:
        data: [C, T] — EEG data
        low_freq: low cutoff frequency in Hz
        high_freq: high cutoff frequency in Hz
        sample_rate: sampling rate in Hz
        order: filter order

    Returns:
        filtered: [C, T]
    """
    nyquist = sample_rate / 2.0

    if low_freq <= 0:
        # Lowpass only
        b, a = butter(order, high_freq / nyquist, btype="low")
    elif high_freq >= nyquist:
        # Highpass only
        b, a = butter(order, low_freq / nyquist, btype="high")
    else:
        b, a = butter(order, [low_freq / nyquist, high_freq / nyquist], btype="band")

    return filtfilt(b, a, data, axis=-1).astype(data.dtype)


def segment_signal(
    data: np.ndarray, segment_length: int, overlap: int = 0
) -> list[np.ndarray]:
    """Segment EEG into fixed-length windows.

    Args:
        data: [C, T] — EEG data
        segment_length: number of samples per segment
        overlap: number of overlapping samples between segments

    Returns:
        segments: list of [C, segment_length] arrays
    """
    _, T = data.shape
    step = segment_length - overlap
    segments = []
    for start in range(0, T - segment_length + 1, step):
        segments.append(data[:, start : start + segment_length])
    return segments


def scale_to_millivolts(data: np.ndarray, unit: str = "uV") -> np.ndarray:
    """Scale EEG data to millivolts.

    Args:
        data: EEG data
        unit: current unit ('uV' for microvolts, 'V' for volts)

    Returns:
        scaled: data in millivolts
    """
    if unit == "uV":
        return data * 1e-3  # μV → mV
    elif unit == "V":
        return data * 1e3   # V → mV
    return data


def preprocess_raw(
    data: np.ndarray,
    orig_sr: int,
    target_sr: int = 256,
    segment_length: int = 1024,
    apply_bandpass: bool = False,
    low_freq: float = 0.0,
    high_freq: float = 38.0,
    unit: str = "uV",
) -> list[np.ndarray]:
    """Full preprocessing pipeline for raw EEG data.

    1. Resample to target_sr
    2. Average re-reference
    3. Optional bandpass filter
    4. Scale to millivolts
    5. Segment into fixed-length windows

    Args:
        data: [C, T] — raw EEG data
        orig_sr: original sample rate
        target_sr: target sample rate (default 256 Hz)
        segment_length: samples per segment (default 1024 = 4s at 256Hz)
        apply_bandpass: whether to apply bandpass filter
        low_freq: low cutoff for bandpass
        high_freq: high cutoff for bandpass
        unit: unit of input data

    Returns:
        segments: list of [C, segment_length] arrays
    """
    # 1. Resample
    data = resample_signal(data, orig_sr, target_sr)

    # 2. Average re-reference
    data = average_rereference(data)

    # 3. Bandpass filter (optional)
    if apply_bandpass:
        data = bandpass_filter(data, low_freq, high_freq, target_sr)

    # 4. Scale to millivolts
    data = scale_to_millivolts(data, unit)

    # 5. Segment
    segments = segment_signal(data, segment_length)

    return segments
