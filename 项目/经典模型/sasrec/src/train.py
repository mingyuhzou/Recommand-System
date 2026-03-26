import os
import math
import time
import random
import logging
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.optimizer import StateDict
from tqdm import tqdm

from src.dataset import Dataset
from src.model.sasrec import SASRec
from src.utils.utils import get_args, get_negative_samples


def build_logger(log_dir: str, log_name: str = "sasrec_train.log"):
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger("sasrec")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )

    file_handler = logging.FileHandler(
        os.path.join(log_dir, log_name),
        encoding="utf-8"
    )
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


def sasrec_loss(positive_logits, negative_logits, positive_idxs, negative_idxs):
    loss_func = nn.BCEWithLogitsLoss()

    positive_logits = positive_logits[positive_idxs]
    positive_labels = torch.ones(size=positive_logits.shape)

    negative_logits = negative_logits[negative_idxs]
    negative_labels = torch.zeros(size=negative_logits.shape)

    positive_loss = loss_func(positive_logits, positive_labels)
    negative_loss = loss_func(negative_logits, negative_labels)

    return positive_loss + negative_loss

@torch.no_grad()
def evaluate_topk(
        mode: str = "valid",
        valid_dataloader=None,
        test_dataloader=None,
        evaluate_k=10,
        model: SASRec = None,
) -> tuple[float, float]:
    dataloader = valid_dataloader if mode == "valid" else test_dataloader

    if model is not None:
        model = model

    model.eval()

    ndcg = 0.0
    hit = 0.0
    num_users = 0

    with torch.no_grad():
        for batch in dataloader:
            input_seqs, item_idxs = batch
            num_users += input_seqs.shape[0]

            outputs = model(
                input_seqs=input_seqs,
                item_idxs=item_idxs,
            )

            logits = -outputs[0]

            if logits.device.type == "mps":
                logits = logits.detach().cpu()

            ranks = logits.argsort().argsort()
            ranks = [r[0].item() for r in ranks]

            for rank in ranks:
                if rank < evaluate_k:
                    ndcg += 1 / np.log2(rank + 2)
                    hit += 1

    ndcg /= num_users
    hit /= num_users

    return ndcg, hit

def main():
    args = get_args()
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )

    logger = build_logger(args.log_dir)

    logger.info("start training")
    logger.info(
        f"args: "
        f"data_path={args.data_path}, "
        f"batch_size={args.batch_size}, "
        f"max_seq_len={args.max_seq_len}, "
        f"hidden_dim={args.hidden_dim}, "
        f"num_blocks={args.num_blocks}, "
        f"dropout={args.dropout}, "
        f"lr={args.lr}, "
        f"epochs={args.epochs}, "
        f"patience={args.patience}, "
        f"topk={args.topk}, "
        f"seed={args.seed}, "
        f"debug={args.debug}, "
        f"device={device}"
    )
    evaluate_k = args.topk

    dataset=dataset_obj = Dataset(
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        data_filepath=args.data_path,
        debug=args.debug,
    )

    train_loader = dataset_obj.get_dataloader(
        dataset_obj.user2items_train,
        split="train"
    )
    valid_dataloader= dataset_obj.get_dataloader(
        dataset_obj.user2items_valid,
        split="valid"
    )
    test_dataloader= dataset_obj.get_dataloader(
        dataset_obj.user2items_test,
        split="test"
    )

    num_items = dataset_obj.num_items

    logger.info(
        f"dataset loaded: "
        f"num_users={dataset_obj.num_users}, "
        f"num_items={num_items}, "
        f"train_users={len(dataset_obj.user2items_train)}, "
        f"valid_users={len(dataset_obj.user2items_valid)}, "
        f"test_users={len(dataset_obj.user2items_test)}"
    )

    model = SASRec(
        num_items=dataset.num_items + 1,
        num_blocks=args.num_blocks,
        hidden_dim=args.hidden_dim,
        max_seq_len=args.max_seq_len,
        dropout_p=args.dropout,
        share_item_emb=True,
        device=device,
    ).to(device)

    for param in model.parameters():
        if param.dim() >= 2:
            nn.init.xavier_uniform_(param)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_valid_recall = -1.0
    best_valid_ndcg = 0.0
    best_state_dict = None
    wait = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_steps = 0
        start_time = time.time()

        for batch in train_loader:
            batch = batch.to(device)  # [B, L]

            optimizer.zero_grad()

            positive_seqs = batch.clone()  # [B, L]
            positive_idxs = torch.where(positive_seqs != 0)

            batch[:, -1] = 0
            input_seqs = batch.roll(shifts=1)  # [B, L]

            negative_seqs = get_negative_samples(
                dataset_obj.positive2negatives,
                positive_seqs
            ).to(device)
            negative_idxs = torch.where(negative_seqs != 0)

            positive_logits, negative_logits = model(
                input_seqs=input_seqs,
                positive_seqs=positive_seqs,
                negative_seqs=negative_seqs
            )

            loss = sasrec_loss(
                positive_logits=positive_logits,
                negative_logits=negative_logits,
                positive_idxs=positive_idxs,
                negative_idxs=negative_idxs
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_steps += 1

        avg_loss = total_loss / max(total_steps, 1)

        ndcg, hit = evaluate_topk("valid",valid_dataloader,test_dataloader, evaluate_k,model)

        elapsed = time.time() - start_time


        print(
            f"epoch={epoch} "
            f"train_loss={avg_loss:.6f} "
            f"valid_recall@{args.topk}={hit:.6f} "
            f"valid_ndcg@{args.topk}={ndcg:.6f} "
            f"time={elapsed:.2f}s"
        )

        if ndcg > best_valid_ndcg:
            best_valid_recall = hit
            best_valid_ndcg = ndcg
            wait = 0

            best_state_dict = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

            if args.save_dir:
                os.makedirs(args.save_dir, exist_ok=True)
                best_model_path = os.path.join(args.save_dir, "best_sasrec.pt")
                torch.save(best_state_dict, best_model_path)
                logger.info(f"best model saved to {best_model_path}")
        else:
            wait += 1
            logger.info(f"no improvement, wait={wait}/{args.patience}")

            if wait >= args.patience:
                logger.info("early stopping triggered")
                print("Early stopping")
                break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        logger.info(
            f"load best model: "
            f"valid_recall@{args.topk}={best_valid_recall:.6f}, "
            f"valid_ndcg@{args.topk}={best_valid_ndcg:.6f}"
        )

    test_recall, test_ndcg = evaluate_topk("test",valid_dataloader,test_dataloader, evaluate_k,model)

    logger.info(
        f"test_recall@{args.topk}={test_recall:.6f} "
        f"test_ndcg@{args.topk}={test_ndcg:.6f}"
    )
    print(
        f"test_recall@{args.topk}={test_recall:.6f} "
        f"test_ndcg@{args.topk}={test_ndcg:.6f}"
    )

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        final_model_path = os.path.join(args.save_dir, "sasrec.pt")
        torch.save(model.state_dict(), final_model_path)
        logger.info(f"final model saved to {final_model_path}")
        print(f"model saved to {final_model_path}")

    logger.info("training finished")


if __name__ == "__main__":
    main()