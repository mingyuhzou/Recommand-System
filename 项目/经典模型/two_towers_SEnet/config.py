import os

package_name = "/Users/Zhuanz/Downloads/Recommand-System/项目/经典模型/two_towers_SEnet"

data_cfg = {
    "raw_ratings": os.path.join(package_name, "ml-1m/ratings.dat"),
    "raw_users": os.path.join(package_name, "ml-1m/users.dat"),
    "raw_movies": os.path.join(package_name, "ml-1m/movies.dat"),
    "utf8_movies": os.path.join(package_name, "ml-1m/movies_utf8.dat"),
    "db_path": os.path.join(package_name, "ml-1m/ml1m.duckdb"),

    "users": os.path.join(package_name, "ml-1m/users.parquet"),
    "movies": os.path.join(package_name, "ml-1m/movies.parquet"),
    "train": os.path.join(package_name, "ml-1m/train.parquet"),
    "test": os.path.join(package_name, "ml-1m/test.parquet"),

    "uid2idx": os.path.join(package_name, "ml-1m/uid2idx.pkl"),
    "mid2idx": os.path.join(package_name, "ml-1m/mid2idx.pkl"),
    "genre2idx": os.path.join(package_name, "ml-1m/genre2idx.pkl"),
    "zip2idx": os.path.join(package_name, "ml-1m/zip2idx.pkl"),

    "word2idx":os.path.join(package_name,"ml-1m/word2idx.pkl"),
    "word2emb":os.path.join(package_name, "ml-1m/word2emb.npy"),
}

model_cfg={
    "embed_dim":64,
    "dropout":0.2,
    "max_title_len":5,
    "max_genre_len":3,

    "occupation_num":21,
    "age_num":8,

    "batch_size":64,
    "epochs":200,
    "lr":0.001,

    "mlp_hidden_units":[128, 64]

}