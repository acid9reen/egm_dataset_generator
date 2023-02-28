import json
import multiprocessing as mp
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
from tqdm import tqdm

from egm_dataset_generator.selector import SelectInstruction
from egm_dataset_generator.selector import Selector
from egm_dataset_generator.selector import SelectResult


class RawDataProcessor:
    def __init__(
            self,
            signal_filename_from_label_getter: Callable[[Path], Path],
            select_instructions: list[SelectInstruction],
            selector: Selector,
    ) -> None:
        self.signal_filename_from_label_getter = signal_filename_from_label_getter
        self.selector = selector
        self.file_select_instructions: dict[Path, list[SelectInstruction]] = defaultdict(
            list,
        )

        for select_instruction in select_instructions:
            self.file_select_instructions[select_instruction.label_path].append(
                select_instruction,
            )

    def _process_label_file(
            self,
            label_file_path_select_instructions: tuple[Path, list[SelectInstruction]],
    ) -> list[SelectResult]:
        label_file_path, select_instructions = label_file_path_select_instructions
        signal_file_path = self.signal_filename_from_label_getter(label_file_path)
        signals = np.load(str(signal_file_path), mmap_mode='r')

        select_results: list[SelectResult] = []

        with open(str(label_file_path), 'r') as labels_file:
            labels = json.load(labels_file)

            for select_instruction in select_instructions:
                select_results.append(
                    self.selector.select(
                        signals[select_instruction.channel_index],
                        labels[select_instruction.channel_index],
                        select_instruction,
                    ),
                )

        return select_results

    def process(self, num_processes: int | None = None) -> list[SelectResult]:
        num_processes = num_processes or mp.cpu_count()
        select_results: list[SelectResult] = []

        with mp.Pool(min(6, num_processes)) as pool:
            for select_result in tqdm(
                pool.imap_unordered(
                    self._process_label_file,
                    self.file_select_instructions.items(),
                ), total=len(self.file_select_instructions), desc='Processing label files',
            ):
                select_results.extend(select_result)

        return select_results
