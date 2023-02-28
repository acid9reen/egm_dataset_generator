import argparse
import logging
import os
import random
from typing import get_args

import numpy as np

from egm_dataset_generator.generate_dataset import DatasetGenerator
from egm_dataset_generator.label_transformers import AVAILABLE_TRANSFORMS
from egm_dataset_generator.label_transformers import dispatch_transform
from egm_dataset_generator.label_transformers import LabelTransformer


LOG_FILE = '.log'


# Logger configuration
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
    ],
)

# Now you can use same config for other files triggered by this module
# Just paste the line below into desired module
logger = logging.getLogger(__name__)


seconds = int


class EGMDatasetGeneratorNamespace(argparse.Namespace):
    random_seed: int
    trim_by: seconds
    limit: int
    label_transform: LabelTransformer
    raw_data_path: str
    output_folder: str


def parse_args() -> EGMDatasetGeneratorNamespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--random_seed', help='Fixed random seed', default=420,
    )

    parser.add_argument(
        '--trim_by',
        type=seconds,
        help='Desired length of a signal to generate in seconds',
        default=2,
    )

    parser.add_argument(
        '-l', '--limit',
        type=int,
        help='Number of signals to generate (size of desired dataset)',
        default=10,
    )

    parser.add_argument(
        '--label_transform',
        type=dispatch_transform,
        help='Type of label transform',
        default='sin',
    )

    parser.add_argument(
        'raw_data_root', help='Path to files to generate dataset from',
    )

    parser.add_argument(
        '-o', '--output_folder',
        help='Location to store generated dataset',
        default='./egm_dataset',
    )

    return parser.parse_args(namespace=EGMDatasetGeneratorNamespace())


def seed_everything(seed: int) -> None:
    """
    Fix seed for random generators
    """

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    logger.info(f'Fix random seed with value: {seed}!')


def main() -> int:
    args = parse_args()
    seed_everything(args.random_seed)

    dataset_generator = DatasetGenerator(
        args.raw_data_root,
        args.trim_by,
        args.limit,
        args.output_folder,
    )

    dataset_generator.generate(args.label_transform)

    return 0
