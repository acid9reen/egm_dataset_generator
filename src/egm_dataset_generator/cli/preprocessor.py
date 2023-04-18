import argparse
from pathlib import Path
from typing import Callable
from typing import Iterable

import numpy as np


class PreprocessorNamespace(argparse.ArgumentParser):
    signals_folder: Path
    labels_folder: Path
    output_folder_name: Path


def parse_args() -> PreprocessorNamespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "signals_folder",
        type=Path,
        help="Path to .npy signals files",
    )
    parser.add_argument(
        "-y",
        "--labels_folder",
        type=Path,
        help="Path to .json labels files",
    )
    parser.add_argument(
        "-o",
        "--output_folder_name",
        type=Path,
        help="Output folder name",
        default="interim",
    )

    return parser.parse_args(namespace=PreprocessorNamespace())


class SignalProcessor(object):
    def __init__(
        self,
        transformations: Iterable[Callable[[np.ndarray, np.ndarray], None]],
        output_folder: Path,
    ) -> None:
        self.transformations = transformations
        self.output_folder = output_folder

    def preprocess_signal(self, signal_path: Path) -> None:
        interim_signal_path = self.output_folder / signal_path.name

        if interim_signal_path.exists():
            return

        raw_signal = np.load(signal_path, "r")
        interim_signal = np.memmap(
            interim_signal_path,
            dtype=np.float32,
            mode="w+",
            shape=raw_signal.shape,
        )

        for transform in self.transformations:
            transform(raw_signal, interim_signal)

    def preprocess_signals(self, signals_paths: Iterable[Path]) -> None:
        for signal_path in signals_paths:
            self.preprocess_signal(signal_path)


def main() -> int:
    args = parse_args()
    signals = args.signals_folder.glob("*.npy")
    # labels = args.labels_folder.glob('*.json')

    output_folder = args.signals_folder.parent.parent / args.output_folder_name
    output_folder.mkdir(exist_ok=True)

    signal_preprocessor = SignalProcessor([], output_folder)
    signal_preprocessor.preprocess_signals(signals)

    return 0
