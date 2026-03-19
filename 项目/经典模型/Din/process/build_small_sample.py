from pathlib import Path
import duckdb
from config import cfg

def build_dataset_duckdb_mini(
    ratings_path: str,
    movies_path: str,
    out_dir: str,
    sample_user_num: int = 1000,      # 先抽多少个用户
    train_user_num: int = 800,        # 这批用户里多少进训练
    seed: float = 0.42,
    max_hist_len: int = 15,
    max_inter_per_user: int = 30,     # 每个用户最多保留多少条交互
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "mini_train.parquet"
    test_path = out_dir / "mini_test.parquet"
    genre_vocab_path = out_dir / "mini_genre_vocab.parquet"

    con = duckdb.connect()
    con.execute("SET threads TO 8;")
    con.execute(f"SELECT setseed({seed});")

    # 1. 读 CSV
    con.execute(f"""
    CREATE OR REPLACE TABLE ratings AS
    SELECT
        CAST(userId AS BIGINT) AS userId,
        CAST(movieId AS BIGINT) AS movieId,
        CAST(rating AS DOUBLE) AS rating,
        CAST(timestamp AS BIGINT) AS timestamp,
        CASE WHEN CAST(rating AS DOUBLE) >= 4.0 THEN 1 ELSE 0 END AS label
    FROM read_csv_auto('{ratings_path}', header=True);
    """)

    con.execute(f"""
    CREATE OR REPLACE TABLE movies AS
    SELECT
        CAST(movieId AS BIGINT) AS movieId,
        COALESCE(NULLIF(genres, ''), '(no genres listed)') AS genres
    FROM read_csv_auto('{movies_path}', header=True);
    """)

    # 2. genre vocab
    con.execute("""
    CREATE OR REPLACE TABLE genre_vocab AS
    WITH genres AS (
        SELECT DISTINCT unnest(string_split(genres, '|')) AS genre
        FROM movies
    )
    SELECT
        row_number() OVER (ORDER BY genre) AS genre_id,
        genre
    FROM genres;
    """)

    # 3. movie -> genre_id list
    con.execute("""
    CREATE OR REPLACE TABLE movie_genre_df AS
    WITH exploded AS (
        SELECT
            m.movieId,
            unnest(string_split(m.genres, '|')) AS genre
        FROM movies m
    )
    SELECT
        e.movieId,
        list(g.genre_id ORDER BY g.genre_id) AS movie_cate_id_list
    FROM exploded e
    JOIN genre_vocab g
      ON e.genre = g.genre
    GROUP BY e.movieId;
    """)

    # 4. 先随机抽 sample_user_num 个用户
    con.execute(f"""
    CREATE OR REPLACE TABLE sampled_users AS
    SELECT userId
    FROM (
        SELECT DISTINCT userId
        FROM ratings
        ORDER BY random()
    )
    LIMIT {sample_user_num};
    """)

    sampled_user_cnt = con.execute(
        "SELECT COUNT(*) FROM sampled_users"
    ).fetchone()[0]
    print(f"sampled users: {sampled_user_cnt}")

    # 5. 只保留抽样用户，并限制每个用户最多 max_inter_per_user 条交互
    con.execute(f"""
    CREATE OR REPLACE TABLE ratings_small AS
    WITH ranked AS (
        SELECT
            r.*,
            row_number() OVER (
                PARTITION BY r.userId
                ORDER BY r.timestamp DESC, r.movieId DESC
            ) AS rn
        FROM ratings r
        JOIN sampled_users u
          ON r.userId = u.userId
    )
    SELECT
        userId,
        movieId,
        rating,
        timestamp,
        label
    FROM ranked
    WHERE rn <= {max_inter_per_user};
    """)

    # 6. 合并评分表
    con.execute("""
    CREATE OR REPLACE TABLE base AS
    SELECT
        r.userId,
        r.movieId,
        mg.movie_cate_id_list,
        r.label,
        r.timestamp
    FROM ratings_small r
    JOIN movie_genre_df mg
      ON r.movieId = mg.movieId;
    """)

    # 7. 构造历史序列
    con.execute(f"""
    CREATE OR REPLACE TABLE samples AS
    SELECT
        userId,
        movieId AS movie_id,
        movie_cate_id_list,
        COALESCE(
            list(movieId) OVER (
                PARTITION BY userId
                ORDER BY timestamp, movieId
                ROWS BETWEEN {max_hist_len} PRECEDING AND 1 PRECEDING
            ),
            []
        ) AS rated_movie_id_list,
        COALESCE(
            list(movie_cate_id_list) OVER (
                PARTITION BY userId
                ORDER BY timestamp, movieId
                ROWS BETWEEN {max_hist_len} PRECEDING AND 1 PRECEDING
            ),
            []
        ) AS rated_movie_cate_id_list,
        label
    FROM base;
    """)

    # 8. 在 sampled users 里再划分 train/test
    con.execute(f"""
    CREATE OR REPLACE TABLE train_users AS
    SELECT userId
    FROM (
        SELECT DISTINCT userId
        FROM samples
        ORDER BY random()
    )
    LIMIT {train_user_num};
    """)

    con.execute("""
    CREATE OR REPLACE TABLE train_df AS
    SELECT s.*
    FROM samples s
    JOIN train_users t
      ON s.userId = t.userId;
    """)

    con.execute("""
    CREATE OR REPLACE TABLE test_df AS
    SELECT s.*
    FROM samples s
    LEFT JOIN train_users t
      ON s.userId = t.userId
    WHERE t.userId IS NULL;
    """)

    # 9. 保存 parquet
    con.execute(f"COPY train_df TO '{train_path}' (FORMAT PARQUET);")
    con.execute(f"COPY test_df TO '{test_path}' (FORMAT PARQUET);")
    con.execute(f"COPY genre_vocab TO '{genre_vocab_path}' (FORMAT PARQUET);")

    # 10. 统计
    total_users = con.execute("SELECT COUNT(DISTINCT userId) FROM samples").fetchone()[0]
    train_users_cnt = con.execute("SELECT COUNT(*) FROM train_users").fetchone()[0]
    test_users_cnt = total_users - train_users_cnt
    train_shape = con.execute("SELECT COUNT(*) FROM train_df").fetchone()[0]
    test_shape = con.execute("SELECT COUNT(*) FROM test_df").fetchone()[0]

    print("===== MINI DATASET =====")
    print(f"total users: {total_users}")
    print(f"train users: {train_users_cnt}")
    print(f"test users : {test_users_cnt}")
    print(f"train rows : {train_shape}")
    print(f"test rows  : {test_shape}")
    print(f"train saved to: {train_path}")
    print(f"test saved  to: {test_path}")
    print(f"genre vocab saved to: {genre_vocab_path}")

    con.close()



build_dataset_duckdb_mini(
    ratings_path=cfg["ratings"],
    movies_path=cfg["movies"],
    out_dir="../ml_20m_mini",
    sample_user_num=1000,
    train_user_num=800,
    seed=0.42,
    max_hist_len=15,
    max_inter_per_user=30,
)