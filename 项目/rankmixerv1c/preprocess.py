import json
import os

import torch
import polars as pl
import pyarrow.parquet as pq
from torch.utils.data import IterableDataset
from typing import Dict, List, Any, Optional
import time
import numpy as np
from hbre_book.model.rankmixerv1c.util import (
    split_book_chapter,
    split_book_word_chapter,
    tf_hash_bucket,
)


class BookRecommendationIterableDataset(IterableDataset):
    """
    流式读取 Parquet - BPR样本版本

    对应TensorFlow版本的make_input_fn和parser_tf逻辑
    支持三元组样本：(正样本, 硬负样本, 随机负样本)
    """

    def __init__(
            self,
            data_file: str,
            needed_columns: List[str],
            book_lookup: Optional[Any] = None,
            word_lookup: Optional[Any] = None,
            export_folder=None,
            device: str = "cpu",
            row_batch_size: int = 1024,
            is_training: bool = True,
            train_ratio: float = 0.9,
            shuffle_buffer_size: int = 10000,
    ):
        super().__init__()
        self.data_file = data_file
        self.device = device
        self.book_lookup = book_lookup
        self.word_lookup = word_lookup
        self.needed_columns = needed_columns
        self.row_batch_size = row_batch_size
        self.is_training = is_training
        self.train_ratio = train_ratio
        self.shuffle_buffer_size = shuffle_buffer_size
        self.export_folder = export_folder

        self.numeric_features = [
            "impression_shelfcount_total",
            "impression_clickcount_total",
            "impression_punch_total",
            "impression_rewardcount_total",
            "negative_shelfcount_total",
            "negative_clickcount_total",
            "negative_punch_total",
            "negative_rewardcount_total",
        ]

        self.numeric_quartiles = self._compute_log_quartiles()

        self.book_type_map = {
            "TEMPLATE": 1,
            "HUABEN": 2,
            "HUABENV1": 3,
            "PREMIUM_SHORT": 4,
            "SHORT": 5,
            "UNK": 0,
        }

        self.plan_type_map = {
            "PC": 1,
            "IOS": 2,
            "IMITATION": 3,
            "ANDROID": 4,
            "UNK": 0,
        }
        self.OS_MAP = {
            "android": 1,
            "ios": 2,
            "harmonyos": 3,
        }

        # 获取总行数用于数据集划分
        pf = pq.ParquetFile(self.data_file)
        self.total_rows = pf.metadata.num_rows
        self.train_rows = int(self.total_rows * self.train_ratio)
        print(
            f"总样本数: {self.total_rows}, "
            f"训练样本: {self.train_rows}, "
            f"验证样本: {self.total_rows - self.train_rows}"
        )

    def _compute_log_quartiles(self):
        quartile_path = os.path.join(self.export_folder, "numeric_quartiles.json")

        # 1. 优先加载
        if os.path.exists(quartile_path):
            with open(quartile_path, "r", encoding="utf-8") as f:
                quartiles = json.load(f)
            print(f"加载四分位: {quartile_path}")
            return quartiles

        # 2. 统计
        pf = pq.ParquetFile(self.data_file)
        buffers = {name: [] for name in self.numeric_features}

        for batch in pf.iter_batches(columns=self.numeric_features, batch_size=200_000):
            df = pl.from_arrow(batch)

            for name in self.numeric_features:
                if name not in df.columns:
                    continue

                arr = (
                    df[name]
                    .cast(pl.Float64, strict=False)
                    .fill_null(0.0)
                    .clip(lower_bound=0.0)
                    .to_numpy()
                )

                if len(arr) > 0:
                    buffers[name].append(np.log1p(arr))

        quartiles = {}
        for name, chunks in buffers.items():
            if not chunks:
                quartiles[name] = [0.0, 0.0, 0.0]
                continue

            all_vals = np.concatenate(chunks, axis=0)
            q1, q2, q3 = np.quantile(all_vals, [0.25, 0.5, 0.75])
            quartiles[name] = [float(q1), float(q2), float(q3)]

        # 3. 保存
        os.makedirs(self.export_folder, exist_ok=True)
        with open(quartile_path, "w", encoding="utf-8") as f:
            json.dump(quartiles, f, ensure_ascii=False, indent=2)

        print(f"四分位已保存: {quartile_path}")
        return quartiles

    def _log_bucket(self, vals, q1, q2, q3):
        vals = (
            vals.cast(pl.Float64, strict=False)
            .fill_null(0.0)
            .clip(lower_bound=0.0)
            .to_numpy()
        )

        log_vals = np.log1p(vals)

        bucket = np.zeros_like(log_vals, dtype=np.int64)
        bucket[(log_vals > q1) & (log_vals <= q2)] = 1
        bucket[(log_vals > q2) & (log_vals <= q3)] = 2
        bucket[log_vals > q3] = 3

        return bucket

    def _select_row_groups(self, pf: pq.ParquetFile, start_row: int, end_row: int):
        """选择需要的行组"""
        rgs = []
        rg_starts = []
        acc = 0
        for i in range(pf.metadata.num_row_groups):
            rg_starts.append(acc)
            acc += pf.metadata.row_group(i).num_rows
            if acc > start_row and rg_starts[-1] < end_row:
                rgs.append(i)
        if not rgs:
            return [], 0
        first_rg_start_row = rg_starts[rgs[0]]
        return rgs, first_rg_start_row

    def __iter__(self):
        pf = pq.ParquetFile(self.data_file)
        data_start, data_end = (
            (0, self.train_rows)
            if self.is_training
            else (self.train_rows, self.total_rows)
        )
        row_groups, first_rg_start = self._select_row_groups(pf, data_start, data_end)
        if not row_groups:
            return
        current_row = first_rg_start

        for batch in pf.iter_batches(
                columns=self.needed_columns,
                batch_size=self.row_batch_size,
                row_groups=row_groups
        ):
            batch_start = current_row
            batch_end = current_row + len(batch)

            # 过滤批次中超出目标范围的行
            if batch_end <= data_start or batch_start >= data_end:
                current_row = batch_end
                continue
            if batch_start < data_start or batch_end > data_end:
                s = max(0, data_start - batch_start)
                e = min(len(batch), data_end - batch_start)
                batch = batch.slice(s, e - s)
            current_row = batch_end
            if len(batch) == 0:
                continue

            processed_batch = self._process_batch(batch)
            yield processed_batch

            if current_row >= data_end:
                break

    def _process_batch(self, batch) -> Dict[str, torch.Tensor]:
        """
        处理一个批次的数据 - 对应TensorFlow版本的parser_tf
        """
        df = pl.from_arrow(batch)
        processed_data = {}

        # =========================================================
        # 0. register_time：时间差分桶
        # =========================================================
        if "register_time" in df.columns:
            df = df.with_columns(
                pl.col("register_time")
                .cast(pl.Int64, strict=False)
                .fill_null(0)
                .alias("register_ts")
            )

            current_utc_ts = time.time()
            beijing_offset = 8 * 3600
            current_beijing_ts = current_utc_ts + beijing_offset

            df = df.with_columns(
                pl.when(pl.col("register_ts") > 0)
                .then(
                    ((current_beijing_ts - pl.col("register_ts")) / 86400)
                    .floor()
                    .cast(pl.Int64)
                )
                .otherwise(0)
                .alias("register_days")
            )

            df = df.with_columns([
                pl.when(pl.col("register_days") < 3).then(0)
                .when(pl.col("register_days") < 8).then(1)
                .when(pl.col("register_days") < 31).then(2)
                .when(pl.col("register_days") < 181).then(3)
                .otherwise(4)
                .alias("register_days_bucket")
            ])

            processed_data["register_time"] = torch.tensor(
                df["register_days_bucket"].to_numpy(),
                dtype=torch.long,
                device=self.device,
            )

        for column in df.columns:
            # =========================================================
            # 1. 用户基础离散特征
            # =========================================================
            if column == "uid":
                processed_data["uid"] = tf_hash_bucket(
                    df[column], 2 ** 18, self.device
                )

            elif column == "os":
                vals = (
                    df[column]
                    .cast(pl.Utf8, strict=False)
                    .fill_null("UNK")
                    .str.to_lowercase()
                    .to_list()
                )
                mapped = [self.OS_MAP.get(v, 0) for v in vals]
                processed_data["os"] = torch.tensor(
                    mapped, dtype=torch.long, device=self.device
                )

            # =========================================================
            # 2. 用户历史序列特征
            # =========================================================
            elif column in ["likemarks", "searchwords"]:
                processed_data[column] = self.word_lookup.lookup(
                    df[column].to_list()
                )

            elif column in ["top_read_books", "read_books"]:
                processed_data[column] = self.book_lookup.lookup(
                    df[column].to_list()
                )

            elif column == "read_search_keywords_books_day_30":
                book, word, chapter = split_book_word_chapter(
                    df[column].to_list(),
                    self.device,
                    self.book_lookup,
                    self.word_lookup,
                )
                processed_data[column] = {
                    "book": book,
                    "word": word,
                    "chapter": chapter,
                }

            # =========================================================
            # 3. 正样本基础内容特征
            # =========================================================
            elif column == "impression_bookid":
                processed_data["impression_bookid"] = self.book_lookup.lookup(
                    df[column].to_list()
                )

            elif column == "impression_bookmarks":
                processed_data["impression_bookmarks"] = self.word_lookup.lookup(
                    df[column].to_list()
                )

            elif column == "impression_bookname":
                processed_data["impression_bookname"] = self.word_lookup.lookup(
                    df[column].to_list()
                )

            elif column == "impression_tag":
                processed_data["impression_tag"] = self.word_lookup.lookup(
                    df[column].to_list()
                )

            elif column == "impression_bookwords":
                processed_data["impression_bookwords"] = self.word_lookup.lookup(
                    df[column].to_list()
                )

            elif column == "impression_bookinfo":
                processed_data["impression_bookinfo"] = self.word_lookup.lookup(
                    df[column].to_list()
                )

            # =========================================================
            # 4. 正样本统计 / 离散特征
            # =========================================================
            elif column == "impression_wordcount":
                vals = (
                    df[column]
                    .cast(pl.Int64, strict=False)
                    .fill_null(0)
                    .clip(0, 400000)
                )

                vals = vals.to_numpy()
                bucket = np.zeros_like(vals, dtype=np.int64)
                bucket[(vals >= 10_000) & (vals < 30_000)] = 1
                bucket[(vals >= 30_000) & (vals < 60_000)] = 2
                bucket[(vals >= 60_000) & (vals < 100_000)] = 3
                bucket[(vals >= 100_000) & (vals < 200_000)] = 4
                bucket[vals >= 200_000] = 5

                processed_data["impression_wordcount"] = torch.tensor(
                    bucket, dtype=torch.long, device=self.device
                )

            elif column == "impression_chapters":
                processed_data["impression_chapters"] = torch.tensor(
                    df[column].cast(pl.Int64, strict=False).fill_null(0).to_numpy(),
                    dtype=torch.long,
                    device=self.device,
                )

            elif column == "impression_book_type":
                vals = (
                    df[column]
                    .cast(pl.Utf8, strict=False)
                    .fill_null("UNK")
                    .str.replace_all('"', '')
                    .to_list()
                )
                mapped = [self.book_type_map.get(v, 0) for v in vals]
                processed_data["impression_book_type"] = torch.tensor(
                    mapped, dtype=torch.long, device=self.device
                )

            elif column == "impression_plan_type":
                vals = (
                    df[column]
                    .cast(pl.Utf8, strict=False)
                    .fill_null("UNK")
                    .str.replace_all('"', '')
                    .to_list()
                )
                mapped = [self.plan_type_map.get(v, 0) for v in vals]
                processed_data["impression_plan_type"] = torch.tensor(
                    mapped, dtype=torch.long, device=self.device
                )

            elif column == "impression_contractstatus":
                vals = (
                    df[column]
                    .cast(pl.Int64, strict=False)
                    .fill_null(-1)
                    .to_numpy()
                )
                vals = vals + 1
                processed_data["impression_contractstatus"] = torch.tensor(
                    vals,
                    dtype=torch.long,
                    device=self.device,
                )

            # =========================================================
            # 5. 负样本基础内容特征
            # =========================================================
            elif column == "negative_bookid":
                processed_data["negative_bookid"] = self.book_lookup.lookup(
                    df[column].to_list()
                )

            elif column == "negative_bookmarks":
                processed_data["negative_bookmarks"] = self.word_lookup.lookup(
                    df[column].to_list()
                )

            elif column == "negative_bookname":
                processed_data["negative_bookname"] = self.word_lookup.lookup(
                    df[column].to_list()
                )

            elif column == "negative_tag":
                processed_data["negative_tag"] = self.word_lookup.lookup(
                    df[column].to_list()
                )

            elif column == "negative_bookwords":
                processed_data["negative_bookwords"] = self.word_lookup.lookup(
                    df[column].to_list()
                )

            elif column == "negative_bookinfo":
                processed_data["negative_bookinfo"] = self.word_lookup.lookup(
                    df[column].to_list()
                )

            # =========================================================
            # 6. 负样本统计 / 离散特征
            # =========================================================
            elif column == "negative_wordcount":
                vals = (
                    df[column]
                    .cast(pl.Int64, strict=False)
                    .fill_null(0)
                    .clip(0, 400000)
                )

                vals = vals.to_numpy()
                bucket = np.zeros_like(vals, dtype=np.int64)
                bucket[(vals >= 10_000) & (vals < 30_000)] = 1
                bucket[(vals >= 30_000) & (vals < 60_000)] = 2
                bucket[(vals >= 60_000) & (vals < 100_000)] = 3
                bucket[(vals >= 100_000) & (vals < 200_000)] = 4
                bucket[vals >= 200_000] = 5

                processed_data["negative_wordcount"] = torch.tensor(
                    bucket, dtype=torch.long, device=self.device
                )

            elif column == "negative_chapters":
                processed_data["negative_chapters"] = torch.tensor(
                    df[column].cast(pl.Int64, strict=False).fill_null(0).to_numpy(),
                    dtype=torch.long,
                    device=self.device,
                )

            elif column == "negative_book_type":
                vals = (
                    df[column]
                    .cast(pl.Utf8, strict=False)
                    .fill_null("UNK")
                    .str.replace_all('"', '')
                    .to_list()
                )
                mapped = [self.book_type_map.get(v, 0) for v in vals]
                processed_data["negative_book_type"] = torch.tensor(
                    mapped, dtype=torch.long, device=self.device
                )

            elif column == "negative_plan_type":
                vals = (
                    df[column]
                    .cast(pl.Utf8, strict=False)
                    .fill_null("UNK")
                    .str.replace_all('"', '')
                    .to_list()
                )
                mapped = [self.plan_type_map.get(v, 0) for v in vals]
                processed_data["negative_plan_type"] = torch.tensor(
                    mapped, dtype=torch.long, device=self.device
                )

            elif column == "negative_contractstatus":
                vals = (
                    df[column]
                    .cast(pl.Int64, strict=False)
                    .fill_null(-1)
                    .to_numpy()
                )
                vals = vals + 1
                processed_data["negative_contractstatus"] = torch.tensor(
                    vals,
                    dtype=torch.long,
                    device=self.device,
                )

            # =========================================================
            # 7. 数值特征：log + 四分位数分桶
            # =========================================================
            elif column in self.numeric_features:
                q1, q2, q3 = self.numeric_quartiles[column]
                bucket = self._log_bucket(df[column], q1, q2, q3)

                processed_data[column] = torch.tensor(
                    bucket, dtype=torch.long, device=self.device
                )

        return processed_data


class VocabLookupTable:
    """
    词汇查找表 - 对应TensorFlow版本的vocabulary lookup逻辑
    """

    def __init__(self, vocab_list: List[str], device: str = "cpu"):
        """
        初始化词汇表

        Args:
            vocab_list: 词汇列表（第一个应该是UNK）
            device: 设备
        """
        # 直接使用原词汇表，不做任何处理
        self.vocab_to_id = {str(vocab): idx for idx, vocab in enumerate(vocab_list)}
        self.device = device
        self.default_id = 0  # UNK的索引

        print(f"创建词汇表，大小: {len(vocab_list)}")

    def lookup(self, strings):
        """
        批量查找，返回tensor

        Args:
            strings: 字符串或字符串列表

        Returns:
            索引张量
        """
        if isinstance(strings, (list, tuple)):
            if len(strings) > 0 and isinstance(strings[0], (list, tuple)):
                # 二维列表
                result = []
                for string_list in strings:
                    if string_list is None:
                        string_list = ["UNK"]
                    ids = [
                        self.vocab_to_id.get(str(s), self.default_id)
                        for s in string_list
                    ]
                    result.append(ids)
                return torch.tensor(result, dtype=torch.long, device=self.device)
            else:
                # 一维列表
                ids = [
                    self.vocab_to_id.get(str(s), self.default_id)
                    for s in strings
                ]
                return torch.tensor(ids, dtype=torch.long, device=self.device)
        else:
            # 单个值
            return torch.tensor(
                [self.vocab_to_id.get(str(strings), self.default_id)],
                dtype=torch.long,
                device=self.device,
            )
