import polars as pl

# -----------------------------
# 1. 读取 users.dat
# -----------------------------
users = pl.read_csv(
    "users.dat",
    has_header=False,
    separator="\n",
    new_columns=["raw"],
)

users = (
    users
    .with_columns(pl.col("raw").str.split("::"))
    .with_columns([
        pl.col("raw").list.get(0).cast(pl.Int32).alias("userId"),
        pl.col("raw").list.get(1).alias("gender"),
        pl.col("raw").list.get(2).alias("age"),
        pl.col("raw").list.get(3).alias("occupation"),
        pl.col("raw").list.get(4).alias("zipcode"),
    ])
    .select(["userId", "gender", "age", "occupation", "zipcode"])
)

# -----------------------------
# 2. 读取 movies.dat
# -----------------------------
movies = pl.read_csv(
    "movies.dat",
    has_header=False,
    separator="\n",
    new_columns=["raw"],
    encoding="latin1",
)

movies = (
    movies
    .with_columns(pl.col("raw").str.split("::"))
    .with_columns([
        pl.col("raw").list.get(0).cast(pl.Int32).alias("movieId"),
        pl.col("raw").list.get(1).alias("title"),
        pl.col("raw").list.get(2).alias("genres"),
    ])
    .select(["movieId", "title", "genres"])
)

movies_feat = (
    movies
    .with_columns([
        pl.col("title").str.extract(r"\((\d{4})\)").cast(pl.Int32).alias("year"),
        pl.col("genres").str.split("|").alias("genre_list"),
    ])
    .select(["movieId", "year", "genre_list"])
)

# -----------------------------
# 3. 读取 ratings.dat
# -----------------------------
ratings = pl.read_csv(
    "ratings.dat",
    has_header=False,
    separator="\n",
    new_columns=["raw"],
)

ratings = (
    ratings
    .with_columns(pl.col("raw").str.split("::"))
    .with_columns([
        pl.col("raw").list.get(0).cast(pl.Int32).alias("userId"),
        pl.col("raw").list.get(1).cast(pl.Int32).alias("movieId"),
        pl.col("raw").list.get(2).cast(pl.Int32).alias("rating"),
        pl.col("raw").list.get(3).cast(pl.Int64).alias("timestamp"),
    ])
    .select(["userId", "movieId", "rating", "timestamp"])
)

# -----------------------------
# 4. 构造 label
#    rating < 3 -> 0
#    rating == 3 -> 丢弃
#    rating > 3 -> 1
# -----------------------------
ratings = (
    ratings
    .with_columns(
        pl.when(pl.col("rating") > 3).then(1)
        .when(pl.col("rating") < 3).then(0)
        .otherwise(None)
        .alias("label")
    )
    .drop_nulls("label")
)

# -----------------------------
# 5. 构造各离散特征映射（从1开始）
# -----------------------------
gender_values = users["gender"].unique().sort().to_list()
gender2idx = {v: i + 1 for i, v in enumerate(gender_values)}

age_values = users["age"].unique().sort().to_list()
age2idx = {v: i + 1 for i, v in enumerate(age_values)}

occupation_values = users["occupation"].unique().sort().to_list()
occupation2idx = {v: i + 1 for i, v in enumerate(occupation_values)}

zipcode_values = users["zipcode"].unique().sort().to_list()
zipcode2idx = {v: i + 1 for i, v in enumerate(zipcode_values)}

year_values = movies_feat["year"].drop_nulls().unique().sort().to_list()
year2idx = {v: i + 1 for i, v in enumerate(year_values)}

genre_values = (
    movies_feat
    .select("genre_list")
    .explode("genre_list")
    .drop_nulls()
    .unique()
    .sort("genre_list")
    .to_series()
    .to_list()
)
genre2idx = {v: i + 1 for i, v in enumerate(genre_values)}

# -----------------------------
# 6. 映射 users
# -----------------------------
users_mapped = (
    users
    .with_columns([
        pl.col("gender").replace_strict(gender2idx).cast(pl.Int32).alias("gender_id"),
        pl.col("age").replace_strict(age2idx).cast(pl.Int32).alias("age_id"),
        pl.col("occupation").replace_strict(occupation2idx).cast(pl.Int32).alias("occupation_id"),
        pl.col("zipcode").replace_strict(zipcode2idx).cast(pl.Int32).alias("zipcode_id"),
    ])
    .select(["userId", "gender_id", "age_id", "occupation_id", "zipcode_id"])
)

# -----------------------------
# 7. 映射 movies
# -----------------------------
movies_mapped = (
    movies_feat
    .with_columns([
        pl.col("year").replace_strict(year2idx).cast(pl.Int32).alias("year_id"),
        pl.col("genre_list").map_elements(
            lambda x: [genre2idx[g] for g in x] if x is not None else [],
            return_dtype=pl.List(pl.Int32),
        ).alias("genre_ids"),
    ])
    .select(["movieId", "year_id", "genre_ids"])
)

# -----------------------------
# 8. join 构造样本
# -----------------------------
samples = (
    ratings
    .join(users_mapped, on="userId", how="inner")
    .join(movies_mapped, on="movieId", how="inner")
    .select([
        "userId",
        "movieId",
        "gender_id",
        "age_id",
        "occupation_id",
        "zipcode_id",
        "year_id",
        "genre_ids",
        "label",
        "timestamp",
    ])
)

# -----------------------------
# 1. userId / movieId 映射（从1开始）
# -----------------------------
user_ids = samples["userId"].unique().sort().to_list()
user2idx = {u: i + 1 for i, u in enumerate(user_ids)}

movie_ids = samples["movieId"].unique().sort().to_list()
movie2idx = {m: i + 1 for i, m in enumerate(movie_ids)}

samples = samples.with_columns([
    pl.col("userId").replace_strict(user2idx).cast(pl.Int32).alias("user_id"),
    pl.col("movieId").replace_strict(movie2idx).cast(pl.Int32).alias("movie_id"),
])

# -----------------------------
# 2. timestamp 归一化（MinMax）
# -----------------------------
ts_min = samples["timestamp"].min()
ts_max = samples["timestamp"].max()

samples = samples.with_columns(
    ((pl.col("timestamp") - ts_min) / (ts_max - ts_min))
    .cast(pl.Float32)
    .alias("timestamp_norm")
)

# -----------------------------
# 3.（可选）只保留新字段
# -----------------------------
samples = samples.select([
    "user_id",
    "movie_id",
    "gender_id",
    "age_id",
    "occupation_id",
    "zipcode_id",
    "year_id",
    "genre_ids",
    "timestamp_norm",
    "label",
])

# 查看结果
print(samples.head())

N = samples.height
perm = np.arange(N)

np.random.seed(2019)
np.random.shuffle(perm)

samples = samples[perm]

# -----------------------------
# 2. 划分 8:1:1
# -----------------------------
train_end = int(N * 0.8)
valid_end = int(N * 0.9)

train = samples[:train_end]
valid = samples[train_end:valid_end]
test  = samples[valid_end:]

# -----------------------------
# 3. 保存
# -----------------------------
train.write_parquet("train.parquet")
valid.write_parquet("valid.parquet")
test.write_parquet("test.parquet")