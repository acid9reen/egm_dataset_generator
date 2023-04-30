import argparse
import multiprocessing as mp
from pathlib import Path
from typing import Callable
from typing import Iterable

import numpy as np
from tqdm import tqdm

from egm_dataset_generator.peaks_fixer import LabelProcessor
from egm_dataset_generator.tranformations import standardize


class PreprocessorNamespace(argparse.ArgumentParser):
    signals_folder: Path
    labels_folder: Path | None
    output_folder_name: Path
    num_workers: int


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
        default=None,
    )
    parser.add_argument(
        "-o",
        "--output_folder_name",
        type=Path,
        help="Output folder name",
        default="interim",
    )
    parser.add_argument(
        "-w",
        "--num_workers",
        type=int,
        help="Number of workers for multiprocessing",
        default=1,
    )

    return parser.parse_args(namespace=PreprocessorNamespace())


class SignalProcessor(object):
    def __init__(
        self,
        transformations: Iterable[Callable[[np.ndarray], np.ndarray]],
        output_folder: Path,
        num_workers: int,
    ) -> None:
        self.transformations = transformations
        self.output_folder = output_folder
        self.num_workers = num_workers

    def preprocess_signal(self, signal_path: Path) -> None:
        interim_signal_path = self.output_folder / signal_path.name

        if interim_signal_path.exists():
            return

        signal = np.load(signal_path)

        for transform in self.transformations:
            signal = transform(signal)

        np.save(interim_signal_path, signal)

    def preprocess_signals(self, signals_paths: list[Path]) -> None:
        with mp.Pool(self.num_workers) as pool:
            for __ in tqdm(
                pool.imap_unordered(
                    self.preprocess_signal,
                    signals_paths,
                    chunksize=len(signals_paths) // self.num_workers,
                ),
                desc="Processing signal files",
                colour="green",
                total=len(signals_paths),
            ):
                ...


def main() -> int:
    args = parse_args()
    signals = list(args.signals_folder.glob("*.npy"))

    if args.labels_folder is not None:
        labels = args.labels_folder.glob("*.json")

        labels_output_folder = (
            args.labels_folder.parent.parent
            / args.output_folder_name
            / args.labels_folder.name
        )
        labels_output_folder.mkdir(exist_ok=True)

        label_signal = {
            label_path: args.signals_folder / f"X{label_path.stem[1:]}.npy"
            for label_path in labels
        }
        LabelProcessor(labels_output_folder).process_labels(label_signal)

    signals_output_folder = (
        args.signals_folder.parent.parent
        / args.output_folder_name
        / args.signals_folder.name
    )
    signals_output_folder.mkdir(exist_ok=True)

    transformations = [standardize]

    signal_preprocessor = SignalProcessor(
        transformations,
        signals_output_folder,
        args.num_workers,
    )
    signal_preprocessor.preprocess_signals(signals)

    return 0
