from torch.utils.data import Dataset
import polars as pl
import torch
import pickle
from config import cfg

class Movielens(Dataset):
    def __init__(self, data_path,max_cate_len=5, max_hist_len=15):
        df = pl.read_parquet(data_path)

        self.max_cate_len=max_cate_len
        self.max_hist_len = max_hist_len

        with open(cfg["user_mapping_path"], "rb") as f:
            self.user_id_to_idx = pickle.load(f)
        self.user = [self.user_id_to_idx[x] for x in df["userId"].to_list()]

        with open(cfg["movie_mapping_path"], "rb") as f:
            self.movie_id_to_idx = pickle.load(f)
        self.movie=[self.movie_id_to_idx[x] for x in df["movie_id"].to_list()]

        self.movie_cate = df["movie_cate_id_list"].to_list()
        self.hist_movies = [
            [self.movie_id_to_idx[m] for m in seq] if seq is not None else []
            for seq in df["rated_movie_id_list"].to_list()
        ]
        self.hist_cate = df["rated_movie_cate_id_list"].to_list()
        self.label = df["label"].to_list()

    def __getitem__(self, idx):
        movie_cate = self._process_cate_data(self.movie_cate[idx], is_hist=False)
        hist_movies = self._process_hist_data(self.hist_movies[idx])
        hist_cate = self._process_cate_data(self.hist_cate[idx], is_hist=True)

        return {
            "userId": torch.tensor(self.user[idx], dtype=torch.long),
            "movie_id": torch.tensor(self.movie[idx], dtype=torch.long),
            "movie_cate_id_list": torch.tensor(movie_cate, dtype=torch.long),
            "hist_movie_id_list": torch.tensor(hist_movies, dtype=torch.long),
            "hist_movie_cate_id_list": torch.tensor(hist_cate, dtype=torch.long),
            "label": torch.tensor(self.label[idx], dtype=torch.float32),
        }

    def _process_hist_data(self, seq):
        """
        截断/补充hist
        """
        if len(seq) >= self.max_hist_len:
            seq = seq[-self.max_hist_len:]
        else:
            if len(seq) > 0 and isinstance(seq[0], list): # 二维
                seq = [[]  for _ in range(self.max_hist_len - len(seq))] + seq
            else:
                seq = [0] * (self.max_hist_len - len(seq)) + seq
        return seq

    def _process_cate_data(self, seq, is_hist=False):
        if seq is None:
            seq = []

        # 历史类别序列: 必须返回 [T, C]
        if is_hist:
            if len(seq) >= self.max_hist_len:
                seq = seq[-self.max_hist_len:]
            else:
                seq = [[] for _ in range(self.max_hist_len - len(seq))] + seq

            new_seq = []
            for ls in seq:
                if ls is None:
                    ls = []
                if len(ls) >= self.max_cate_len:
                    ls = ls[-self.max_cate_len:]
                else:
                    ls = [0] * (self.max_cate_len - len(ls)) + ls
                new_seq.append(ls)
            return new_seq

        # 当前item类别: 返回 [C]
        else:
            if len(seq) >= self.max_cate_len:
                seq = seq[-self.max_cate_len:]
            else:
                seq = [0] * (self.max_cate_len - len(seq)) + seq
            return seq

    def __len__(self):
        return len(self.user)