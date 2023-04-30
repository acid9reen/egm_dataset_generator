import json
import multiprocessing as mp
from pathlib import Path
from typing import NamedTuple

import numpy as np
from scipy.interpolate import CubicSpline
from tqdm import tqdm


def find_relative_minimum_derivative_index(signal_cutout: np.ndarray) -> int | None:
    if np.std(signal_cutout) < 0.01:
        return None

    indexes = range(len(signal_cutout))
    cs = CubicSpline(indexes, signal_cutout)

    high_res_indexes = np.linspace(
        0,
        len(signal_cutout) - 1,
        len(signal_cutout) - 1,
    )

    abscis_min_der_index = int(np.argmin(cs(high_res_indexes, 1)))

    return abscis_min_der_index


class Peak(NamedTuple):
    index: int
    channel: int


def read_label_file(filepath: Path) -> list[list[int]]:
    with open(filepath, "r") as in_:
        labels = json.load(in_)

    return labels


def labels_to_pairs(labels: list[list[int]]) -> tuple[list[Peak], set[int]]:
    peaks = []
    empty_channels = set()

    for channel_index, channel in enumerate(labels):
        if channel:
            transformed_channel = map(lambda index: Peak(index, channel_index), channel)
            peaks.extend(transformed_channel)
            continue

        empty_channels.add(channel_index)

    return sorted(peaks, key=lambda peak: peak.index), empty_channels


def fix_peaks(
    peaks: list[Peak],
    signal: np.ndarray,
    empty_channels: set[int],
    window_size: int = 20,
) -> list[list[int]]:
    result: list[list[int]] = [[] for __ in range(64)]
    threshold = (64 - len(empty_channels)) // 2

    pivot = peaks[0].index
    window_frame: list[Peak] = []

    for peak in peaks:
        if (index := peak.index) < pivot + window_size:
            window_frame.append(peak)
            continue

        if len(window_frame) > threshold:
            channel_index = {peak.channel: peak.index for peak in window_frame}

            for i in range(len(result)):
                if i in empty_channels:
                    continue

                if (found_peak := channel_index.get(i)) is not None:
                    result[i].append(found_peak)
                elif (
                    rel_der_ind := find_relative_minimum_derivative_index(
                        signal[i][pivot - window_size : pivot + window_size],
                    )
                ) is not None:
                    result[i].append(pivot - window_size + rel_der_ind)

        pivot = index
        window_frame: list[Peak] = []

    return result


def save_processed_labels(processed_labels: list[list[int]], filepath: Path) -> None:
    with open(filepath, "w") as out:
        json.dump(processed_labels, out)


class LabelProcessor(object):
    def __init__(
        self,
        output_folder: Path,
        num_workers: int = mp.cpu_count(),
    ) -> None:
        self.output_folder = output_folder
        self.num_workers = num_workers

    def preprocess_label(self, label_signal_paths: tuple[Path, Path]) -> None:
        label_path, signal_path = label_signal_paths

        fixed_label_path = self.output_folder / label_path.name
        if fixed_label_path.exists():
            return

        label = read_label_file(label_path)
        signal = np.load(signal_path, "r")

        peaks, empty_channels = labels_to_pairs(label)
        fixed_label = fix_peaks(peaks, signal, empty_channels, 20)

        save_processed_labels(fixed_label, fixed_label_path)

    def process_labels(self, label_signal: dict[Path, Path]) -> None:
        with mp.Pool(self.num_workers) as pool:
            for __ in tqdm(
                pool.imap_unordered(
                    self.preprocess_label,
                    label_signal.items(),
                    chunksize=len(label_signal) // self.num_workers or 1,
                ),
                desc="Preprocessing label files",
                colour="green",
                total=len(label_signal),
            ):
                ...
