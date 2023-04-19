import numpy as np


def normalize(signal: np.ndarray) -> np.ndarray:
    for channel_no, channel in enumerate(signal):
        max_ = np.max(channel)
        min_ = np.min(channel)

        signal[channel_no] = (channel - min_) / (max_ - min_)

    return signal


def standardize(signal: np.ndarray) -> np.ndarray:
    for channel_no, channel in enumerate(signal):
        mean = np.mean(channel)
        std = np.std(channel)

        signal[channel_no] = (channel - mean) / std

    return signal
