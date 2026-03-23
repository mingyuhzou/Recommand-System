import pickle
import torch
from torch.utils.data import Dataset
import polars as pl

from config import data_cfg,model_cfg
from utils import ages_split_buckets


class Movielens(Dataset):
    def __init__(self, data_path, max_title_len=model_cfg['max_title_len'], max_genre_len=model_cfg['max_genre_len']):
        df = pl.read_parquet(data_path)

        self.df = df
        self.max_title_len = max_title_len
        self.max_genre_len = max_genre_len

        with open(data_cfg["uid2idx"], "rb") as f:
            self.uid2idx = pickle.load(f)
        self.users = [self.uid2idx[x] for x in df["userId"].to_list()]

        with open(data_cfg["mid2idx"], "rb") as f:
            self.mid2idx = pickle.load(f)
        self.movies = [self.mid2idx[x] for x in df["movieId"].to_list()]

        with open(data_cfg["genre2idx"], "rb") as f:
            self.genre2idx = pickle.load(f)
        self.genres = [
            self._pad_seq(
                [self.genre2idx[g] for g in str(x).split("|") if g in self.genre2idx],
                0,
                self.max_genre_len
            )
            for x in df["genres"].to_list()
        ]

        with open(data_cfg["zip2idx"], "rb") as f:
            self.zipcode2idx = pickle.load(f)
        self.zipcodes = [self.zipcode2idx[str(x)] for x in df["zipCode"].to_list()]

        with open(data_cfg["word2idx"], "rb") as f:
            self.word2idx = pickle.load(f)
        self.titles = [
            self._pad_seq(
                [self.word2idx.get(w, self.word2idx["<UNK>"]) for w in str(x).split() if w],
                0,
                self.max_title_len
            )
            for x in df["title"].to_list()
        ]

        self.genders = [self._gender_to_idx(x) for x in df["gender"].to_list()]
        self.ages = ages_split_buckets(df["age"].to_list())
        self.occupations = [x + 1 for x in df["occupation"].to_list()]
        self.labels = [1 if x >= 4 else 0 for x in df["rating"].to_list()]

    def _gender_to_idx(self, g):
        if g == "M":
            return 1
        elif g == "F":
            return 2
        return 0

    def _pad_seq(self, seq, pad_val, max_len):
        if len(seq) > max_len:
            return seq[:max_len]
        return seq + [pad_val] * (max_len - len(seq))

    def __getitem__(self, idx):
        return {
            "userId": torch.tensor(self.df["userId"][idx], dtype=torch.long),
            "movieId": torch.tensor(self.df["movieId"][idx], dtype=torch.long),

            "uid": torch.tensor(self.users[idx], dtype=torch.long),
            "mid": torch.tensor(self.movies[idx], dtype=torch.long),
            "genres": torch.tensor(self.genres[idx], dtype=torch.long),
            "zipcode": torch.tensor(self.zipcodes[idx], dtype=torch.long),
            "title": torch.tensor(self.titles[idx], dtype=torch.long),
            "gender": torch.tensor(self.genders[idx], dtype=torch.long),
            "age": torch.tensor(self.ages[idx], dtype=torch.long),
            "occupation": torch.tensor(self.occupations[idx], dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

    def __len__(self):
        return len(self.df)


class Movies(Dataset):
    def __init__(self, data_path, max_title_len, max_genre_len):
        df = pl.read_parquet(data_path)

        self.df = df
        self.max_title_len = max_title_len
        self.max_genre_len = max_genre_len

        with open(data_cfg["mid2idx"], "rb") as f:
            self.mid2idx = pickle.load(f)
        self.movies = [self.mid2idx[x] for x in df["movieId"].to_list()]

        with open(data_cfg["genre2idx"], "rb") as f:
            self.genre2idx = pickle.load(f)

        self.genres = [
            self._pad_seq(
                [self.genre2idx[g] for g in str(x).split("|") if g in self.genre2idx],
                0,
                self.max_genre_len
            )
            for x in df["genres"].to_list()
        ]

        with open(data_cfg["word2idx"], "rb") as f:
            self.word2idx = pickle.load(f)
        self.titles = [
            self._pad_seq(
                [self.word2idx.get(w, self.word2idx["<UNK>"]) for w in str(x).split() if w],
                0,
                self.max_title_len
            )
            for x in df["title"].to_list()
        ]

    def __len__(self):
        return len(self.df)

    def _pad_seq(self, seq, pad_idx, max_len):
        if seq is None:
            seq = []
        if len(seq) >= max_len:
            return seq[:max_len]
        return seq + [pad_idx] * (max_len - len(seq))

    def __getitem__(self, idx):
        return {
            "movieId": torch.tensor(self.df["movieId"][idx], dtype=torch.long),
            "mid": torch.tensor(self.movies[idx], dtype=torch.long),
            "genres": torch.tensor(self.genres[idx], dtype=torch.long),
            "title": torch.tensor(self.titles[idx], dtype=torch.long),
        }
