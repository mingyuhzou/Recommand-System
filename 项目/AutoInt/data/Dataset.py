import torch
from torch.utils.data import Dataset
import polars as pl

class MovieDataset(Dataset):
    def __init__(self, data_path, max_genre_len=6):
        """
        df: 你之前构建好的 samples（polars DataFrame）
        """
        df=pl.read_parquet(data_path)

        self.max_genre_len = max_genre_len

        # 转成 python list（避免每次getitem慢）
        self.user_id = df["user_id"].to_list()
        self.movie_id = df["movie_id"].to_list()
        self.gender = df["gender_id"].to_list()
        self.age = df["age_id"].to_list()
        self.occupation = df["occupation_id"].to_list()
        self.zipcode = df["zipcode_id"].to_list()
        self.year = df["year_id"].to_list()
        self.genre = df["genre_ids"].to_list()
        self.timestamp = df["timestamp_norm"].to_list()
        self.label = df["label"].to_list()

    def pad_genre(self, g):
        if g is None:
            g = []
        g = g[:self.max_genre_len]  # 截断
        pad_len = self.max_genre_len - len(g)
        return g + [0] * pad_len

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        genre_ids = self.pad_genre(self.genre[idx])

        return {
            "user_id": torch.tensor(self.user_id[idx], dtype=torch.long),
            "movie_id": torch.tensor(self.movie_id[idx], dtype=torch.long),
            "gender": torch.tensor(self.gender[idx], dtype=torch.long),
            "age": torch.tensor(self.age[idx], dtype=torch.long),
            "occupation": torch.tensor(self.occupation[idx], dtype=torch.long),
            "zipcode": torch.tensor(self.zipcode[idx], dtype=torch.long),
            "year": torch.tensor(self.year[idx], dtype=torch.long),
            "genre": torch.tensor(genre_ids, dtype=torch.long),   # ← 已补齐
            "timestamp": torch.tensor(self.timestamp[idx], dtype=torch.float32),
            "label": torch.tensor(self.label[idx], dtype=torch.float32),
        }
