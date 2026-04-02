import torch
import polars as pl
import pyarrow.parquet as pq
from torch.utils.data import IterableDataset
from typing import Dict, List, Any, Optional
import time

from hbre_book.model.mgdin.util import (
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

        # 获取总行数用于数据集划分
        pf = pq.ParquetFile(self.data_file)
        self.total_rows = pf.metadata.num_rows
        self.train_rows = int(self.total_rows * self.train_ratio)
        print(
            f"总样本数: {self.total_rows}, "
            f"训练样本: {self.train_rows}, "
            f"验证样本: {self.total_rows - self.train_rows}"
        )

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

        Args:
            batch: PyArrow批次

        Returns:
            处理后的批次字典
        """
        df = pl.from_arrow(batch)

        processed_data = {}

        if "register_time" in df.columns:
            # 1. 注册时间：转为秒级 UTC 时间戳（处理无效值）
            df = df.with_columns(
                pl.col("register_time")
                .cast(pl.Int64, strict=False)  # 1472140800 → 秒级 int64
                .fill_null(0)  # 空值填充为 0（后续视为无效时间）
                .alias("register_ts")
            )

            # 2. 计算北京时间的天数差（核心公式修改）
            current_utc_ts = time.time()  # 当前 UTC 秒级时间戳
            beijing_offset = 8 * 3600  # 北京时间比 UTC 快 8 小时（秒数）
            current_beijing_ts = current_utc_ts + beijing_offset  # 转为北京时间对应的 UTC 时间戳

            df = df.with_columns(
                # 公式：(当前北京时间对应的 UTC 戳 - 注册 UTC 戳) / 86400 → 北京时间天数差
                pl.when(pl.col("register_ts") > 0)  # 有效时间戳（>0）
                .then(
                    ((current_beijing_ts - pl.col("register_ts")) / 86400)
                    .floor()  # 向下取整（和 int() 效果一致，更规范）
                    .cast(pl.Int64)
                )
                .otherwise(0)  # 无效时间：注册天数设为 0（后续分桶为 0，代表新用户）
                .alias("register_days")
            )

            # 3. 分桶逻辑
            df = df.with_columns([
                pl.when(pl.col("register_days") < 3).then(0)
                .when(pl.col("register_days") < 8).then(1)
                .when(pl.col("register_days") < 31).then(2)
                .when(pl.col("register_days") < 181).then(3)
                .otherwise(4)
                .alias("register_days_bucket")
            ])

            # invalid_count = df.filter(pl.col("register_ts") <= 0).shape[0]
            # bucket_counts = df["register_days_bucket"].value_counts().sort("register_days_bucket")
            # for bucket, count in zip(bucket_counts["register_days_bucket"], bucket_counts["count"]):
            #     bucket_desc = \
            #     ["0-3天", "3-7天", "7-30天", "30-60天", "60-90天", "90-180天", "180-360天", "360-720天", ">720天"][
            #         bucket]
            #     print(f"  桶{bucket}（{bucket_desc}）: {count} 个用户")
            # 4. 转为 Tensor
            processed_data["register_time"] = torch.tensor(
                df["register_days_bucket"].to_numpy(), dtype=torch.long, device=self.device
            )

        for column in df.columns:
            # ========== 用户特征 ==========
            if column == "uid":
                # 对应TensorFlow版本的categorical_hash_embedding
                processed_data["uid"] = tf_hash_bucket(
                    df[column], 2 ** 18, self.device
                )

            elif column in ["likemarks", "searchwords"]:
                processed_data[column] = self.word_lookup.lookup(df[column].to_list())

            elif column in ["read_books"]:
                processed_data[column] = self.book_lookup.lookup(df[column].to_list())

            elif column in ["read_search_keywords_books_day_30"]:
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

            # ========== 正样本特征 ==========
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

            # ========== 负样本特征 ==========
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
