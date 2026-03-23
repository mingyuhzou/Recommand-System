import math
import pandas as pd
import torch
from torch import optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

from config import data_cfg, model_cfg
from model.two_towers import TwoTowers
from Data.Movielens import Movielens,Movies

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


train_data = Movielens(
    data_path=data_cfg["train"],
    max_title_len=model_cfg["max_title_len"],
    max_genre_len=model_cfg["max_genre_len"],
)

test_data = Movielens(
    data_path=data_cfg["test"],
    max_title_len=model_cfg["max_title_len"],
    max_genre_len=model_cfg["max_genre_len"],
)

movie_data = Movies(
    data_path=data_cfg["movies"],
    max_title_len=model_cfg["max_title_len"],
    max_genre_len=model_cfg["max_genre_len"],
)

train_loader = DataLoader(
    train_data,
    batch_size=model_cfg["batch_size"],
    shuffle=True,
)

test_loader = DataLoader(
    test_data,
    batch_size=model_cfg["batch_size"],
    shuffle=False,
    drop_last=False
)

movie_loader = DataLoader(
    movie_data,
    batch_size=model_cfg["batch_size"],
    shuffle=False,
    drop_last=False
)

model = TwoTowers(
    embed_dim=model_cfg["embed_dim"],
    mlp_hidden_units=model_cfg["mlp_hidden_units"],
).to(device)

optimizer = optim.Adam(model.parameters(), lr=model_cfg["lr"])


def move_batch_to_device(batch, device):
    return {
        k: v.to(device) if torch.is_tensor(v) else v
        for k, v in batch.items()
    }


def sampled_softmax(user_embedding, item_embedding, temperature=0.05):
    user_embedding = F.normalize(user_embedding, p=2, dim=-1)
    item_embedding = F.normalize(item_embedding, p=2, dim=-1)

    logits = torch.matmul(user_embedding, item_embedding.t()) / temperature
    labels = torch.arange(logits.size(0), device=logits.device)

    loss_u2i = F.cross_entropy(logits, labels)
    loss_i2u = F.cross_entropy(logits.t(), labels)

    return 0.5 * (loss_u2i + loss_i2u)


@torch.no_grad()
def build_all_item_embeddings():
    model.eval()

    all_item_embs = []
    all_movie_ids = []

    for batch in movie_loader:
        batch = move_batch_to_device(batch, device)

        item_output = model.get_item_embedding(batch)
        item_output = F.normalize(item_output, p=2, dim=-1)

        all_item_embs.append(item_output.cpu())
        all_movie_ids.append(batch["movieId"].cpu())

    all_item_embs = torch.cat(all_item_embs, dim=0)   # [N, D]
    all_movie_ids = torch.cat(all_movie_ids, dim=0)   # [N]

    return all_movie_ids, all_item_embs


def build_test_gt_dict():
    df = pd.read_parquet(data_cfg["test"], columns=["userId", "movieId"])
    gt_dict = df.groupby("userId")["movieId"].apply(set).to_dict()
    return gt_dict


@torch.no_grad()
def evaluate_recall_ndcg(k=50):
    model.eval()

    all_movie_ids, all_item_embs = build_all_item_embeddings()
    all_item_embs = all_item_embs.to(device)
    all_movie_ids_device = all_movie_ids.to(device)

    gt_dict = build_test_gt_dict()
    user_emb_dict = {}

    for batch in test_loader:
        batch = move_batch_to_device(batch, device)

        user_output = model.get_user_embedding(batch)
        user_output = F.normalize(user_output, p=2, dim=-1)

        user_ids = batch["userId"].detach().cpu().tolist()
        user_output = user_output.detach().cpu()

        for i, user_id in enumerate(user_ids):
            if user_id not in user_emb_dict:
                user_emb_dict[user_id] = user_output[i]

    recall_list = []
    ndcg_list = []

    for user_id, user_emb in user_emb_dict.items():
        gt_items = gt_dict.get(user_id, None)
        if gt_items is None or len(gt_items) == 0:
            continue

        user_emb = user_emb.to(device).unsqueeze(0)
        scores = torch.matmul(user_emb, all_item_embs.t()).squeeze(0)

        topk = min(k, scores.size(0))
        _, topk_indices = torch.topk(scores, topk, dim=0)
        pred_movie_ids = all_movie_ids_device[topk_indices].detach().cpu().tolist()

        pred_set = set(pred_movie_ids)

        hit_cnt = len(pred_set & gt_items)
        recall = hit_cnt / len(gt_items)
        recall_list.append(recall)

        dcg = 0.0
        for rank, movie_id in enumerate(pred_movie_ids):
            if movie_id in gt_items:
                dcg += 1.0 / math.log2(rank + 2)

        ideal_hits = min(len(gt_items), topk)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcg_list.append(ndcg)

    mean_recall = sum(recall_list) / len(recall_list) if recall_list else 0.0
    mean_ndcg = sum(ndcg_list) / len(ndcg_list) if ndcg_list else 0.0

    return mean_recall, mean_ndcg

def train():
    best_recall = 0.0

    for epoch in range(model_cfg["epochs"]):
        model.train()
        total_loss = 0.0
        total_step = 0

        for batch in train_loader:
            batch = move_batch_to_device(batch, device)

            optimizer.zero_grad()

            user_output, item_output = model(batch)
            loss = sampled_softmax(user_output, item_output)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_step += 1

        train_loss = total_loss / max(total_step, 1)
        val_recall, val_ndcg = evaluate_recall_ndcg(k=50)

        if epoch%10==0:
            print(
                f"epoch={epoch + 1} "
                f"train_loss={train_loss:.6f} "
                f"recall@50={val_recall:.6f} "
                f"ndcg@50={val_ndcg:.6f}"
            )
        if val_recall > best_recall:
            best_recall = val_recall

    print(f"best_recall@50={best_recall:.6f}")


if __name__ == "__main__":
    train()