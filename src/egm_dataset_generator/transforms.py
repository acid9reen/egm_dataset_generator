import numpy as np


def scale(signal: np.ndarray) -> np.ndarray:
    min_ = signal.min()
    max_ = signal.max()
    scale = max(abs(min_), abs(max_))
    scaled = signal / scale

    return scaled
