import os
from pathlib import Path
from typing import NamedTuple

import numpy as np

from egm_dataset_generator.label_transformers import IdentityTransformer
from egm_dataset_generator.label_transformers import LabelTransformer
from egm_dataset_generator.transforms import scale


class SelectInstruction(NamedTuple):
    label_path: Path
    channel_index: int
    peak_index: int
    shift: int


class SelectResult(NamedTuple):
    processed_signal_file_path: str
    processed_label_file_path: str
    peak_quantity: int
    channel: int


class Selector:
    def __init__(
        self,
        sample_length: int,
        out_folder_path: str = "generated_data",
        processed_xs_out_folder: str = "x",
        processed_ys_out_folder: str = "y",
        label_transformer: LabelTransformer = IdentityTransformer(),
    ) -> None:
        self.sample_length = sample_length
        self.label_transformer = label_transformer

        self.processed_xs_path = os.path.join(out_folder_path, processed_xs_out_folder)
        self.processed_ys_path = os.path.join(out_folder_path, processed_ys_out_folder)

        os.makedirs(self.processed_xs_path, exist_ok=True)
        os.makedirs(self.processed_ys_path, exist_ok=True)

    def _save(self, x: np.ndarray, y: np.ndarray, filename: str) -> None:
        with open(os.path.join(self.processed_xs_path, filename), "wb") as xs_out:
            np.save(xs_out, scale(x))

        with open(os.path.join(self.processed_ys_path, filename), "wb") as ys_out:
            np.save(ys_out, y)

    def select(
        self,
        signal: np.ndarray,
        label: list[int],
        select_instruction: SelectInstruction,
    ) -> SelectResult:
        peak = label[select_instruction.peak_index]

        max_index = len(signal) - 1

        left_index = 0
        right_index = left_index + self.sample_length

        if peak + self.sample_length > max_index:
            right_index = max_index
            left_index = right_index - self.sample_length + 1
        elif peak - select_instruction.shift < 0:
            left_index = 0
            right_index = self.sample_length - 1
        else:
            right_index = peak + self.sample_length - select_instruction.shift - 1
            left_index = peak - select_instruction.shift

        selected_signal = signal[left_index : right_index + 1]
        selected_peaks = list(filter(lambda x: left_index <= x <= right_index, label))

        selected_label = np.zeros(len(selected_signal))

        for selected_peak in selected_peaks:
            selected_label[selected_peak - left_index] = 1

        transformed_label = self.label_transformer.transform(selected_label)

        sample_file_name = (
            f"{select_instruction.label_path.stem}"
            f"_{select_instruction.channel_index}"
            f"_{left_index}"
            f"_{right_index}"
            f".npy"
        )

        self._save(selected_signal, transformed_label, sample_file_name)

        return SelectResult(
            os.path.join(self.processed_xs_path, sample_file_name),
            os.path.join(self.processed_ys_path, sample_file_name),
            len(selected_peaks),
            select_instruction.channel_index,
        )
