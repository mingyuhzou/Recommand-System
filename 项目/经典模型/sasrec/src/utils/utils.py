import argparse
import os

import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm


InputSequences = torch.Tensor
PositiveSamples = torch.Tensor
NegativeSamples = torch.Tensor

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # 基础参数
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
    )
    parser.add_argument(
        "--device",
        default="",
        type=str,
    )

    # 路径参数
    parser.add_argument(
        "--data_root",
        default="../data",
        type=str,
        help="数据根目录",
    )
    parser.add_argument(
        "--data_filename",
        default="movie-lens_1m.txt",
        type=str,
        choices=[
            "amazon_beauty.txt",
            "amazon_games.txt",
            "steam.txt",
            "movie-lens_1m.txt",
        ],
        help="数据文件名",
    )
    parser.add_argument(
        "--log_dir",
        default="../logs",
        type=str,
    )
    parser.add_argument(
        "--save_dir",
        default="../outputs",
        type=str,
    )

    # 数据参数
    parser.add_argument(
        "--max_seq_len",
        default=50,
        type=int,
    )
    parser.add_argument(
        "--batch_size",
        default=128,
        type=int,
    )

    # 模型参数
    parser.add_argument(
        "--hidden_dim",
        default=50,
        type=int,
    )
    parser.add_argument(
        "--num_blocks",
        default=2,
        type=int,
    )
    parser.add_argument(
        "--dropout",
        default=0.5,
        type=float,
    )
    parser.add_argument(
        "--share_item_emb",
        action="store_true",
        default=False,
    )

    # 优化器参数
    parser.add_argument(
        "--lr",
        default=0.001,
        type=float,
    )
    parser.add_argument(
        "--beta1",
        default=0.9,
        type=float,
    )
    parser.add_argument(
        "--beta2",
        default=0.999,
        type=float,
    )
    parser.add_argument(
        "--eps",
        default=1e-8,
        type=float,
    )
    parser.add_argument(
        "--weight_decay",
        default=0.0,
        type=float,
    )

    # 训练参数
    parser.add_argument(
        "--topk",
        default=10,
        type=int,
    )
    parser.add_argument(
        "--epochs",
        default=2000,
        type=int,
    )
    parser.add_argument(
        "--patience",
        default=20,
        type=int,
    )

    args = parser.parse_args()

    if not args.device:
        args.device = get_device()

    args.data_path = os.path.join(args.data_root, args.data_filename)

    return args


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"

    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def get_positive2negatives(num_items: int, num_samples: int = 100) -> list[int]:
    """
    Creates a dictionary that maps an integer to an array of
      negative integers. This dictionary will be used later
      when we create negative samples for each positive sample.
    """
    all_samples = np.arange(1, num_items + 1)
    positive2negatives = {}
    pbar = tqdm(
        iterable=all_samples,
        desc="Creating positive2negatives",
        total=all_samples.shape[0],
    )
    for positive_sample in pbar:
        candidates = np.concatenate(
            (np.arange(positive_sample), np.arange(positive_sample + 1, num_items + 1)),
            axis=0,
        )
        negative_samples = np.random.choice(
            candidates, size=(num_samples,), replace=False
        )

        positive2negatives[positive_sample] = negative_samples.tolist()

    return positive2negatives


def get_negative_samples(
    positive2negatives: dict[int, list[int]],
    positive_seqs: torch.Tensor,
    num_samples=1,
) -> torch.Tensor:
    negative_seqs = torch.zeros(size=positive_seqs.shape, dtype=torch.long)
    for row_idx in range(positive_seqs.shape[0]):
        for col_idx in range(positive_seqs[row_idx].shape[0]):
            positive_sample = positive_seqs[row_idx][col_idx].item()

            if positive_sample == 0:
                continue

            negative_samples = positive2negatives[positive_sample]
            negative_sample = np.random.choice(
                a=negative_samples, size=(num_samples,), replace=False
            )
            negative_seqs[row_idx][col_idx] = negative_sample[0]

    return negative_seqs


def pad_or_truncate_seq(
    sequence: list[int],
    max_seq_len: int,
) -> InputSequences:
    """Pads or truncates sequences depending on max_seq_len."""
    if isinstance(sequence, list):
        sequence = torch.tensor(sequence)

    if len(sequence) > max_seq_len:
        sequence = sequence[-max_seq_len:]
    else:
        diff = max_seq_len - len(sequence)
        sequence = F.pad(sequence, pad=(diff, 0))

    return sequence


def get_output_name(args: argparse.Namespace, timestamp: str) -> str:
    data_name, _ = os.path.splitext(args.data_filename)

    output_name = (
        f"sasrec-{data_name}_"
        f"lr-{args.lr}_"
        f"batch-size-{args.batch_size}_"
        f"early-stop-{args.early_stop_epoch}"
        f"num-epochs-{args.num_epochs}_"
        f"seed-{args.random_seed}_"
        f"{timestamp}"
    )

    return output_name
