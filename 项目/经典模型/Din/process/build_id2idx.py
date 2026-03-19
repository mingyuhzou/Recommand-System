import polars as pl
from config import cfg
import pickle

train = pl.read_parquet(cfg["train"])
test = pl.read_parquet(cfg["test"])

data = pl.concat([train, test])

# =========================
# 1. user mapping
# =========================
user_mapping = (
    data.select("userId")
    .unique()
    .with_row_index(name="user_idx", offset=1)
)
user_id_to_idx = dict(zip(user_mapping["userId"], user_mapping["user_idx"]))

print(f"用户数: {user_mapping.height}")
sample_uid = data["userId"][0]
print(f"原始 UID {sample_uid} -> {user_id_to_idx[sample_uid]}")

with open(cfg["user_mapping_path"], "wb") as f:
    pickle.dump(user_id_to_idx, f)

# =========================
# 2. movie mapping
# movie_id + rated_movie_id_list 一起构建
# =========================
target_movies = data.select(pl.col("movie_id").alias("movie_raw"))

hist_movies = (
    data.select(pl.col("rated_movie_id_list"))
    .explode("rated_movie_id_list")
    .drop_nulls()
    .select(pl.col("rated_movie_id_list").alias("movie_raw"))
)

all_movies = pl.concat([target_movies, hist_movies]).unique()

movie_mapping = all_movies.with_row_index(name="movie_idx", offset=1)

movie_id_to_idx = dict(zip(movie_mapping["movie_raw"], movie_mapping["movie_idx"]))

print(f"电影数: {movie_mapping.height}")
sample_mid = data["movie_id"][0]
print(f"原始 MID {sample_mid} -> {movie_id_to_idx[sample_mid]}")

with open(cfg["movie_mapping_path"], "wb") as f:
    pickle.dump(movie_id_to_idx, f)