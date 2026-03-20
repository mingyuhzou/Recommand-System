import os
import duckdb

from config import data_cfg


def convert_latin1_to_utf8(src_path: str, dst_path: str):
    with open(src_path, "r", encoding="latin1") as f:
        content = f.read()
    with open(dst_path, "w", encoding="utf-8") as f:
        f.write(content)


def build_ml1m_user_split(data_cfg: dict, n_test: int = 5, min_user_pos: int = 10):
    if not os.path.exists(data_cfg["raw_ratings"]):
        raise FileNotFoundError(f"ratings.dat 不存在: {data_cfg['raw_ratings']}")
    if not os.path.exists(data_cfg["raw_users"]):
        raise FileNotFoundError(f"users.dat 不存在: {data_cfg['raw_users']}")
    if not os.path.exists(data_cfg["raw_movies"]):
        raise FileNotFoundError(f"movies.dat 不存在: {data_cfg['raw_movies']}")

    os.makedirs(os.path.dirname(data_cfg["db_path"]), exist_ok=True)
    os.makedirs(os.path.dirname(data_cfg["users"]), exist_ok=True)
    os.makedirs(os.path.dirname(data_cfg["movies"]), exist_ok=True)
    os.makedirs(os.path.dirname(data_cfg["train"]), exist_ok=True)
    os.makedirs(os.path.dirname(data_cfg["test"]), exist_ok=True)

    # movies.dat 先转 utf8
    convert_latin1_to_utf8(data_cfg["raw_movies"], data_cfg["utf8_movies"])

    con = duckdb.connect(data_cfg["db_path"])

    # 1. ratings_raw
    con.execute(f"""
    CREATE OR REPLACE TABLE ratings_raw AS
    SELECT * FROM read_csv(
        '{data_cfg["raw_ratings"]}',
        delim='\\n',
        columns={{'line': 'VARCHAR'}}
    );
    """)

    # 2. ratings
    con.execute("""
    CREATE OR REPLACE TABLE ratings AS
    SELECT
        CAST(split_part(line, '::', 1) AS INTEGER) AS userId,
        CAST(split_part(line, '::', 2) AS INTEGER) AS movieId,
        CAST(split_part(line, '::', 3) AS INTEGER) AS rating,
        CAST(split_part(line, '::', 4) AS BIGINT) AS timestamp
    FROM ratings_raw
    WHERE line IS NOT NULL AND line <> '';
    """)

    # 3. users_raw
    con.execute(f"""
    CREATE OR REPLACE TABLE users_raw AS
    SELECT * FROM read_csv(
        '{data_cfg["raw_users"]}',
        delim='\\n',
        columns={{'line': 'VARCHAR'}}
    );
    """)

    # 4. users
    con.execute("""
    CREATE OR REPLACE TABLE users AS
    SELECT
        CAST(split_part(line, '::', 1) AS INTEGER) AS userId,
        split_part(line, '::', 2) AS gender,
        CAST(split_part(line, '::', 3) AS INTEGER) AS age,
        CAST(split_part(line, '::', 4) AS INTEGER) AS occupation,
        split_part(line, '::', 5) AS zipCode
    FROM users_raw
    WHERE line IS NOT NULL AND line <> '';
    """)

    # 5. movies_raw
    con.execute(f"""
    CREATE OR REPLACE TABLE movies_raw AS
    SELECT * FROM read_csv(
        '{data_cfg["utf8_movies"]}',
        delim='\\n',
        columns={{'line': 'VARCHAR'}}
    );
    """)

    # 6. movies
    con.execute("""
    CREATE OR REPLACE TABLE movies AS
    SELECT
        CAST(split_part(line, '::', 1) AS INTEGER) AS movieId,
        split_part(line, '::', 2) AS title,
        split_part(line, '::', 3) AS genres
    FROM movies_raw
    WHERE line IS NOT NULL AND line <> '';
    """)

    # 7. 正样本评分表，只保留 rating >= 4
    con.execute("""
    CREATE OR REPLACE TABLE pos_ratings AS
    SELECT
        userId,
        movieId,
        rating,
        timestamp
    FROM ratings
    WHERE rating >= 4;
    """)

    # 8. 只保留正样本数足够多的用户
    con.execute(f"""
    CREATE OR REPLACE TABLE qualified_users AS
    SELECT
        userId,
        COUNT(*) AS pos_cnt
    FROM pos_ratings
    GROUP BY userId
    HAVING COUNT(*) >= {min_user_pos};
    """)

    # 9. 为每个用户按时间排序，标记顺序
    con.execute("""
    CREATE OR REPLACE TABLE pos_ratings_ranked AS
    SELECT
        p.userId,
        p.movieId,
        p.rating,
        p.timestamp,
        ROW_NUMBER() OVER (
            PARTITION BY p.userId
            ORDER BY p.timestamp ASC, p.movieId ASC
        ) AS rn_asc,
        COUNT(*) OVER (
            PARTITION BY p.userId
        ) AS user_pos_cnt
    FROM pos_ratings p
    JOIN qualified_users q
      ON p.userId = q.userId;
    """)

    # 10. train：每个用户前 user_pos_cnt - n_test 个
    con.execute(f"""
    CREATE OR REPLACE TABLE train AS
    SELECT
        p.userId,
        p.movieId,
        p.rating,
        p.timestamp,

        -- user 特征
        u.gender,
        u.age,
        u.occupation,
        u.zipCode,

        -- movie 特征
        m.title,
        m.genres

    FROM pos_ratings_ranked p
    JOIN users u
      ON p.userId = u.userId
    JOIN movies m
      ON p.movieId = m.movieId
    WHERE p.rn_asc <= p.user_pos_cnt - {n_test};
    """)

    # 11. test：每个用户最后 n_test 个
    con.execute(f"""
    CREATE OR REPLACE TABLE test AS
    SELECT
        p.userId,
        p.movieId,
        p.rating,
        p.timestamp,

        -- user 特征
        u.gender,
        u.age,
        u.occupation,
        u.zipCode,

        -- movie 特征
        m.title,
        m.genres

    FROM pos_ratings_ranked p
    JOIN users u
      ON p.userId = u.userId
    JOIN movies m
      ON p.movieId = m.movieId
    WHERE p.rn_asc > p.user_pos_cnt - {n_test};
    """)

    # 12. 导出 parquet
    con.execute(f"""
    COPY users TO '{data_cfg["users"]}' (FORMAT PARQUET);
    """)

    con.execute(f"""
    COPY movies TO '{data_cfg["movies"]}' (FORMAT PARQUET);
    """)

    con.execute(f"""
    COPY train TO '{data_cfg["train"]}' (FORMAT PARQUET);
    """)

    con.execute(f"""
    COPY test TO '{data_cfg["test"]}' (FORMAT PARQUET);
    """)

    # 13. 打印统计
    total_users = con.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    total_movies = con.execute("SELECT COUNT(*) FROM movies").fetchone()[0]
    pos_user_count = con.execute("SELECT COUNT(*) FROM qualified_users").fetchone()[0]
    train_user_count = con.execute("SELECT COUNT(DISTINCT userId) FROM train").fetchone()[0]
    test_user_count = con.execute("SELECT COUNT(DISTINCT userId) FROM test").fetchone()[0]
    train_row_count = con.execute("SELECT COUNT(*) FROM train").fetchone()[0]
    test_row_count = con.execute("SELECT COUNT(*) FROM test").fetchone()[0]

    print("users_count             =", total_users)
    print("movies_count            =", total_movies)
    print("qualified_users_count   =", pos_user_count)
    print("train_users_count       =", train_user_count)
    print("test_users_count        =", test_user_count)
    print("train_rows_count        =", train_row_count)
    print("test_rows_count         =", test_row_count)
    print("n_test_per_user         =", n_test)
    print("min_user_pos            =", min_user_pos)

    print("\ntrain sample:")
    print(con.execute("""
    SELECT * FROM train
    ORDER BY userId, timestamp
    LIMIT 10
    """).fetchdf())

    print("\ntest sample:")
    print(con.execute("""
    SELECT * FROM test
    ORDER BY userId, timestamp
    LIMIT 10
    """).fetchdf())

    print("\nper-user test count sample:")
    print(con.execute("""
    SELECT userId, COUNT(*) AS test_cnt
    FROM test
    GROUP BY userId
    ORDER BY userId
    LIMIT 10
    """).fetchdf())

    con.close()


build_ml1m_user_split(data_cfg, n_test=5, min_user_pos=10)