import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import time
from config import cfg
from model.Din import DIN
from Data.movielens import Movielens


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# 1. 数据
# =========================
train_dataset = Movielens(cfg["mini_train"])
test_dataset = Movielens(cfg["mini_train"])

train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=True)


# =========================
# 2. 模型
# =========================
model = DIN(cfg["embed_dim"]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.BCEWithLogitsLoss()


# =========================
# 3. 训练函数
# =========================
def train_one_epoch(model, dataloader, epoch):
    model.train()
    total_loss = 0.0
    start_time = time.time()

    for step, batch in enumerate(dataloader):
        for k in batch:
            batch[k] = batch[k].to(device)

        optimizer.zero_grad()

        output = model(batch)
        logit = output["logit"]
        label = batch["label"].float()

        loss = criterion(logit, label)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # ⭐ 每100步打印一次
        if step % 100 == 0:
            elapsed = time.time() - start_time
            print(
                f"[Train] Epoch={epoch} Step={step}/{len(dataloader)} "
                f"Loss={loss.item():.4f} "
                f"AvgLoss={total_loss/(step+1):.4f} "
                f"Time={elapsed:.1f}s"
            )

    return total_loss / len(dataloader)


# =========================
# 4. AUC评估 与论文中指标一致
# =========================
def evaluate(model, dataloader):
    model.eval()

    all_pred = []
    all_label = []

    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            for k in batch:
                batch[k] = batch[k].to(device)

            output = model(batch)
            logit = output["logit"]
            prob = torch.sigmoid(logit)

            all_pred.extend(prob.cpu().numpy().tolist())
            all_label.extend(batch["label"].float().cpu().numpy().tolist())

            #  每100步打印
            if step % 100 == 0:
                print(f"[Eval] Step={step}/{len(dataloader)}")

    auc = roc_auc_score(all_label, all_pred)
    return auc


# =========================
# 5. 训练主流程
# =========================
epochs = 10

for epoch in range(epochs):
    print(f"\n========== Epoch {epoch} ==========")

    train_loss = train_one_epoch(model, train_loader, epoch)
    val_auc = evaluate(model, test_loader)

    print(
        f"[Epoch {epoch} Done] "
        f"TrainLoss={train_loss:.4f} "
        f"ValAUC={val_auc:.4f}"
    )