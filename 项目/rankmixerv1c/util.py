import hashlib
import random
from typing import Optional, List, Tuple, Any, Dict
import threading
import logging as log
import queue

import torch
import polars as pl
import numpy as np

# ========== 常量定义 ==========
MAX_CHAPTER_VALUE = 100  # 对应TensorFlow版本的章节最大值


class BackgroundPrefetch:
    """
    后台预取数据 - 对应TensorFlow版本的预取机制

    使用线程在后台预加载数据，减少训练等待时间
    """

    def __init__(self, iterable, max_prefetch: int = 2):
        """
        初始化预取器

        Args:
            iterable: 可迭代对象
            max_prefetch: 最大预取批次数
        """
        self.iterable = iter(iterable)
        self.q = queue.Queue(max_prefetch)
        self.t = threading.Thread(target=self._worker, daemon=True)
        self.t.start()

    def _worker(self):
        """后台工作线程"""
        try:
            for item in self.iterable:
                self.q.put(item)
        finally:
            self.q.put(None)

    def __iter__(self):
        return self

    def __next__(self):
        item = self.q.get()
        if item is None:
            raise StopIteration
        return item


def pick_list(sequence: List[List[Any]], length: int, padding=None) -> List[List[Any]]:
    """
    对列表序列进行填充或采样 - 对应TensorFlow版本的pick_list逻辑

    Args:
        sequence: 输入列表序列
        length: 目标长度
        padding: 填充值

    Returns:
        处理后的列表
    """
    final_list = []
    for every_user_data in sequence:
        if every_user_data is not None:
            if len(every_user_data) >= length:
                final_list.append(random.sample(every_user_data, length))
            elif padding:
                final_list.append(
                    every_user_data + [padding] * (length - len(every_user_data))
                )
        else:
            final_list.append([padding] * length)
    return final_list


def split_book_chapter(
        lst: List[List[str]], device: str, book_lookup
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    分割书ID和章节字符串 - 对应TensorFlow版本的split逻辑

    处理格式: "bookid:chapter"

    Args:
        lst: 字符串列表 [["10014862:3", "10024952:1", ...], ...]
        device: 设备
        book_lookup: 书籍查找表

    Returns:
        (book_tensor, chapter_bucket_tensor)
    """
    book_list = []
    chapter_list = []

    for every_user_data in lst:
        book = []
        chapter = []
        for data in every_user_data:
            split_str = data.split(":")
            if len(split_str) == 2:
                book.append(split_str[0])
                try:
                    # 对应TensorFlow版本的clip_by_value
                    chapter_val = max(0, min(int(split_str[1]), MAX_CHAPTER_VALUE))
                    chapter.append(chapter_val)
                except (ValueError, TypeError):
                    chapter.append(0)
            else:
                book.append("UNK")
                chapter.append(0)
        book_list.append(book)
        chapter_list.append(chapter)

    # 转换为张量
    book_tensor = book_lookup.lookup(book_list)
    chapter_tensor = torch.tensor(chapter_list, dtype=torch.long, device=device)

    return book_tensor, chapter_tensor

def split_book_word_chapter(lst, device, book_lookup, word_lookup):
    """
    分割书ID、书标签和阅读章节数字符串

    Args:
        lst: 输入列表
        device：转换为tensor后的设备，当前强制为cpu
        book_lookup: 书表
        word_lookup: 词表
    """
    book_list = []
    word_list = []
    chapter_list = []
    for every_user_data in lst:
        book = []
        word = []
        chapter = []
        for data in every_user_data:
            split_str = data.split(":")
            if len(split_str) == 3:
                book.append(split_str[1])
                word.append(split_str[0])
                try:
                    chapter.append(max(0, min(int(split_str[2]), MAX_CHAPTER_VALUE)))
                except (ValueError, TypeError):
                    chapter.append(0)
            else:
                book.append("UNK")
                word.append("UNK")
                chapter.append(0)
        book_list.append(book)
        word_list.append(word)
        chapter_list.append(chapter)
    book_tensor = book_lookup.lookup(book_list)
    word_tensor = word_lookup.lookup(word_list)
    chapter_tensor = torch.tensor(chapter_list, dtype=torch.long, device=device)
    return book_tensor, word_tensor, chapter_tensor

def tf_hash_bucket(
        inputs: pl.Series, hash_bucket_size: int, device: str
) -> torch.Tensor:
    """
    TensorFlow风格的哈希桶 - 对应TensorFlow版本的categorical_hash_embedding

    Args:
        inputs: Polars Series
        hash_bucket_size: 哈希桶大小
        device: 设备

    Returns:
        哈希后的索引张量
    """
    # 处理pl.Series类型,速度快
    if isinstance(inputs, pl.Series):
        s = inputs.cast(pl.Utf8)
        h = s.hash(seed=0)
        out = (h % hash_bucket_size).to_numpy()  # numpy uint64
        return torch.tensor(out, dtype=torch.long, device=device).unsqueeze(-1)

    # 兜底：处理列表或单个值
    if hasattr(inputs, "__iter__"):
        arr = [str(v) for v in inputs]
        hash_ids = []
        for s in arr:
            hash_int = int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16)
            hash_ids.append(hash_int % hash_bucket_size)
        return torch.tensor(hash_ids, dtype=torch.long, device=device)
    else:
        hash_int = int(hashlib.md5(str(inputs).encode("utf-8")).hexdigest(), 16)
        return torch.tensor(
            [hash_int % hash_bucket_size], dtype=torch.long, device=device
        )


def move_to_device(x: Any, device: str) -> Any:
    """
    递归地将数据移动到目标设备

    Args:
        x: 输入数据（张量、字典、列表等）
        device: 目标设备

    Returns:
        移动后的数据
    """
    if isinstance(x, torch.Tensor):
        return x.to(device, non_blocking=True)
    if isinstance(x, dict):
        return {k: move_to_device(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        t = [move_to_device(v, device) for v in x]
        return type(x)(t) if isinstance(x, tuple) else t
    return x


def ensure_train_size_polars(df: pl.DataFrame, target_count: int) -> pl.DataFrame:
    """
    确保 Polars DataFrame 训练集达到指定规模 (900万)，逻辑复现 TensorFlow/DuckDB 版本。
    如果超过目标，随机截断。如果不足目标，循环重复。
    """
    actual_count = df.height
    log.info(f"实际样本数: {actual_count:,}, 目标样本数: {target_count:,}")

    if actual_count == 0:
        raise ValueError("没有生成任何训练样本！")

    if actual_count >= target_count:
        log.info("✅ 样本充足，随机采样到目标数量")
        # 使用 Polars 的 sample(n=) 进行随机采样和截断
        # shuffle=True 确保了随机性
        df_sampled = df.sample(n=target_count, shuffle=True)
        log.info(f"✅ 最终训练集大小: {df_sampled.height:,}")
        return df_sampled
    else:
        # 样本不足，循环重复
        repeat_times = (target_count + actual_count - 1) // actual_count
        remaining = target_count % actual_count

        log.info(
            f"⚠️ 样本不足，需要重复采样 {repeat_times} 轮（完整轮数：{repeat_times - (1 if remaining > 0 else 0)}，剩余：{remaining:,}）")

        # 构建完整的重复列表 (full_rounds)
        full_rounds = repeat_times if remaining == 0 else repeat_times - 1

        # 步骤 1: 完整轮次复制
        # Polars 使用 list comprehension 和 pl.concat 实现高效复制
        if full_rounds > 0:
            df_list = [df] * full_rounds
        else:
            df_list = []

        # 步骤 2: 剩余部分采样
        if remaining > 0:
            # Polars 的 sample(n=) 再次用于获取精确的剩余数量
            df_remaining = df.sample(n=remaining, shuffle=True)
            df_list.append(df_remaining)

        # 合并所有数据
        df_final = pl.concat(df_list, how="vertical")

        # 验证
        final_count = df_final.height
        log.info(f"✅ 最终训练集大小: {final_count:,}")

        # 重新随机打乱，确保训练顺序不依赖复制顺序
        df_final = df_final.sample(fraction=1.0, shuffle=True)

        return df_final

def parse_day_last_bookids(x):
    if x is None:
        return []

    if not isinstance(x, str):
        x = str(x)

    x = x.strip()
    if not x:
        return []

    res = []
    for item in x.split(","):
        item = item.strip()
        if not item:
            continue

        parts = item.split(":")
        bid = parts[0].strip() if len(parts) > 0 else ""
        if bid and bid != "UNK":
            res.append(bid)
    return res


def parse_day_last_searchwords(x):
    if x is None:
        return []

    if not isinstance(x, str):
        x = str(x)

    x = x.strip()
    if not x:
        return []

    res = []
    for item in x.split(","):
        item = item.strip()
        if not item:
            continue

        parts = item.split(":")
        word = parts[0].strip() if len(parts) > 0 else ""
        if word and word != "UNK":
            res.append(word)
    return res

def safe_split(value, sep=","):
    if value is None:
        return []
    if not isinstance(value, str):
        value = str(value)
    parts = [x.strip() for x in value.split(sep)]
    return [x for x in parts if x != ""]

def merge_readbooks_with_day_last(readbooks_raw, day_last_raw):
    base = safe_split(readbooks_raw)
    base = [x for x in base if x != "UNK"]

    day_last_books = parse_day_last_bookids(day_last_raw)

    merged = day_last_books + base

    seen = set()
    result = []
    for x in merged:
        if x not in seen:
            seen.add(x)
            result.append(x)

    if not result:
        return ["UNK"]
    return result


def merge_searchwords_with_day_last(searchwords_raw, day_last_raw):
    base = safe_split(searchwords_raw)
    base = [x for x in base if x != "UNK"]

    day_last_words = parse_day_last_searchwords(day_last_raw)

    merged = day_last_words + base

    seen = set()
    result = []
    for x in merged:
        if x not in seen:
            seen.add(x)
            result.append(x)

    if not result:
        return ["UNK"]
    return result