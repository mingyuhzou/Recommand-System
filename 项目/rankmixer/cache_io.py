import os, json, time, random
from pathlib import Path
from typing import Iterator, List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, wait

import torch
from torch.utils.data import IterableDataset

MANIFEST = "manifest.json"


def save_manifest(cache_dir: str, manifest: dict):
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(cache_dir, MANIFEST), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def load_manifest(cache_dir: str) -> Optional[dict]:
    p = Path(cache_dir) / MANIFEST
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


class CacheWriter(IterableDataset):
    def __init__(
        self,
        base_ds: IterableDataset,
        cache_dir: str,
        shard_size: int = 128,
        meta: Optional[dict] = None,
        max_async_writes: int = 2,
    ):
        super().__init__()
        self.base_ds = base_ds
        self.cache_dir = cache_dir
        self.shard_size = shard_size
        self.meta = meta or {}
        self.max_async_writes = max_async_writes
        Path(cache_dir).mkdir(parents=True, exist_ok=True)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        shard_idx = 0
        in_shard: List[Dict[str, Any]] = []
        num_batches = 0
        executor = ThreadPoolExecutor(max_workers=self.max_async_writes)
        pending = []

        manifest = dict(self.meta)
        manifest.update(
            {
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "shard_size": self.shard_size,
                "num_shards": 0,
                "num_batches": 0,
            }
        )
        save_manifest(self.cache_dir, manifest)

        def submit_save(buf, idx):
            path = os.path.join(self.cache_dir, f"shard_{idx:05d}.pt")
            return executor.submit(torch.save, buf, path)

        try:
            for batch in self.base_ds:
                in_shard.append(batch)
                num_batches += 1

                # 满了就异步写一个 shard
                if len(in_shard) >= self.shard_size:
                    buf = in_shard  # 交给写线程
                    in_shard = []  # 新缓冲（双缓冲）
                    pending.append(submit_save(buf, shard_idx))
                    shard_idx += 1
                    # 限制最多同时写 N 个 shard，避免写线程堆积占内存
                    if len(pending) >= self.max_async_writes:
                        wait([pending.pop(0)])

                # 先把当前批交给训练
                yield batch
        finally:
            # flush 剩余批
            if in_shard:
                pending.append(submit_save(in_shard, shard_idx))
                shard_idx += 1
                in_shard = []

            # 等待所有写入完成
            if pending:
                wait(pending)
            executor.shutdown(wait=True)

            manifest["num_shards"] = shard_idx
            manifest["num_batches"] = num_batches
            save_manifest(self.cache_dir, manifest)


class CacheReader(IterableDataset):
    """
    读取缓存目录中的 shard_*.pt 文件，按批 yield 已处理好的 CPU 批字典。
    可按 epoch 打乱 shard 顺序：shuffle_files=True。
    """

    def __init__(self, cache_dir: str, shuffle_files: bool = True, shuffle_batches: bool = True):
        super().__init__()
        self.cache_dir = cache_dir
        self.shuffle_files = shuffle_files
        self.shuffle_batches = shuffle_batches
        self.manifest = load_manifest(cache_dir)
        self._shards = sorted(Path(cache_dir).glob("shard_*.pt"))

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        files = list(self._shards)

        # 1. 打乱shard文件顺序
        if self.shuffle_files:
            random.shuffle(files)

        for path in files:
            batches: List[Dict[str, Any]] = torch.load(path, map_location="cpu")

            # 2. 打乱shard内部的批次顺序
            if self.shuffle_batches:
                random.shuffle(batches)  # 在内存中shuffle，极快（<10ms）

            for b in batches:
                yield b
