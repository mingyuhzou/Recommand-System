import os
import pickle
import pandas as pd

from config import data_cfg


def save_pickle(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_movies_genres(movies_path: str):
    movies_df = pd.read_parquet(movies_path)
    if "movieId" not in movies_df.columns:
        raise ValueError("movies.parquet 缺少 movieId 字段")
    if "genres" not in movies_df.columns:
        raise ValueError("movies.parquet 缺少 genres 字段")
    return movies_df


def build_uid2idx(train_path: str, test_path: str):
    train_df = pd.read_parquet(train_path, columns=["userId"])
    test_df = pd.read_parquet(test_path, columns=["userId"])

    user_ids = pd.concat([train_df["userId"], test_df["userId"]], axis=0)
    user_ids = pd.Series(user_ids.unique()).sort_values().tolist()

    uid2idx = {uid: idx for idx, uid in enumerate(user_ids,start=1)}
    idx2uid = {idx: uid for uid, idx in uid2idx.items()}

    return uid2idx, idx2uid


def build_mid2idx(train_path: str, test_path: str, movies_path: str):
    train_df = pd.read_parquet(train_path, columns=["movieId"])
    test_df = pd.read_parquet(test_path, columns=["movieId"])
    movies_df = pd.read_parquet(movies_path, columns=["movieId"])

    movie_ids = pd.concat(
        [train_df["movieId"], test_df["movieId"], movies_df["movieId"]],
        axis=0
    )
    movie_ids = pd.Series(movie_ids.unique()).sort_values().tolist()

    mid2idx = {mid: idx for idx, mid in enumerate(movie_ids,start=1)}
    idx2mid = {idx: mid for mid, idx in mid2idx.items()}

    return mid2idx, idx2mid


def build_genre2idx(movies_path: str):
    movies_df = load_movies_genres(movies_path)

    all_genres = set()

    for genres in movies_df["genres"].fillna(""):
        if not isinstance(genres, str):
            continue
        for g in genres.split("|"):
            g = g.strip()
            if g:
                all_genres.add(g)

    all_genres = sorted(all_genres)

    genre2idx = {genre: idx for idx, genre in enumerate(all_genres,start=1)}
    idx2genre = {idx: genre for genre, idx in genre2idx.items()}

    return genre2idx, idx2genre


def build_id2idx(data_cfg: dict):
    if not os.path.exists(data_cfg["train"]):
        raise FileNotFoundError(f'train.parquet 不存在: {data_cfg["train"]}')
    if not os.path.exists(data_cfg["test"]):
        raise FileNotFoundError(f'test.parquet 不存在: {data_cfg["test"]}')
    if not os.path.exists(data_cfg["movies"]):
        raise FileNotFoundError(f'movies.parquet 不存在: {data_cfg["movies"]}')

    uid2idx, idx2uid = build_uid2idx(data_cfg["train"], data_cfg["test"])
    mid2idx, idx2mid = build_mid2idx(data_cfg["train"], data_cfg["test"], data_cfg["movies"])
    genre2idx, idx2genre = build_genre2idx(data_cfg["movies"])

    save_pickle(uid2idx, data_cfg["uid2idx"])

    save_pickle(mid2idx, data_cfg["mid2idx"])

    save_pickle(genre2idx, data_cfg["genre2idx"])

    print("uid count   =", len(uid2idx))
    print("movie count =", len(mid2idx))
    print("genre count =", len(genre2idx))

    print("\nuid2idx sample:")
    print(list(uid2idx.items())[:10])

    print("\nmid2idx sample:")
    print(list(mid2idx.items())[:10])

    print("\ngenre2idx sample:")
    print(list(genre2idx.items())[:10])

    return {
        "uid2idx": uid2idx,
        "idx2uid": idx2uid,
        "mid2idx": mid2idx,
        "idx2mid": idx2mid,
        "genre2idx": genre2idx,
        "idx2genre": idx2genre,
    }
def build_zip2idx(cfg):
    # 读取 users
    users_df = pd.read_parquet(cfg["users"])

    if "zipCode" not in users_df.columns:
        raise ValueError("users.parquet 中没有 zipCode 字段")

    # 1. 取唯一值
    zip_codes = users_df["zipCode"].dropna().unique().tolist()

    # 可选排序（推荐）
    zip_codes = sorted(zip_codes)

    # 2. 构建映射（0 = padding）
    zip2idx = {}

    for i, z in enumerate(zip_codes, start=1):
        zip2idx[z] = i

    idx2zip = {i: z for z, i in zip2idx.items()}

    # 3. 保存
    save_pickle(zip2idx, cfg["zip2idx"])

    # 4. 打印信息
    print("zipCode count =", len(zip2idx))
    print("\nsample:")
    print(list(zip2idx.items())[:10])

    return zip2idx, idx2zip

build_id2idx(data_cfg)
build_zip2idx(data_cfg)