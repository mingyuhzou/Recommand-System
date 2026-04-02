import argparse
import asyncio
import logging
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
import polars as pl
import torch

from hbre_book.util.project import get_local_path
from hbre_book.ability import oss
import grpc
import grpc.aio
from hbre_book.grpc import model_v1_pb2_grpc as model_pb2_grpc, model_v1_pb2 as model_pb2
from hbre_book.model.mgdin.preprocess import VocabLookupTable
from hbre_book.model.mgdin.mgdin import MGDIN
from hbre_book.model.mgdin.util import (
    tf_hash_bucket,
)

import time

os.environ["TZ"] = "Asia/Shanghai"
time.tzset()
log = logging.getLogger(__name__)

# -------------------- 参数解析 --------------------
model_name = Path(__file__).resolve().parent.name
log.info(f"当前模型：「{model_name}」")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_dir",
    type=str,
    default=f"/fast/book_re/model/{model_name}",
    help="模型目录",
)
parser.add_argument("--port", default=50051, help="server listen port")
parser.add_argument("--download_model", action="store_true", help="是否从oss下载模型")
args = parser.parse_args()

USER_HASH_BUCKET_SIZE = 2**18  # 用户ID哈希桶大小
match_pools = {}


def startup_event():
    # 加载词表和书表
    global book_lookup, word_lookup

    book_table = pl.read_parquet(
        args.model_dir + "/book_counter.parquet", columns=["bookid"]
    )["bookid"].to_list()
    word_table = pl.read_parquet(
        args.model_dir + "/word_counter.parquet", columns=["word"]
    )["word"].to_list()
    book_lookup = VocabLookupTable(book_table, "cpu")
    word_lookup = VocabLookupTable(word_table, "cpu")

    # 加载模型
    global saved_model
    ts = time.time()
    saved_model = MGDIN(
        num_fields=12,
        field_dim=64,
        group_sizes=[2, 3, 4, 6,12],
        book_vocab_size=len(book_table),
        word_vocab_size=len(word_table),
        num_layers=3,
        attn_dim=64,
        branch_dim=64,
        ffn_hidden_dim=128,
        mlp_hidden_dims=(128, 64),
        dropout=0.2,
    )

    print("num_branches =", len(saved_model.branches))
    print("mlp_in_dim =", saved_model.mlp[0].in_features)

    state = torch.load(
        os.path.join(args.model_dir, "model.pth"), map_location="cpu", weights_only=True
    )
    saved_model.load_state_dict(state)
    saved_model = saved_model.float().to("cpu")
    saved_model.eval()
    log.info(f"load model time elapsed: {time.time() - ts:.3f}s")

    # 创建 executor
    global executor
    executor = ThreadPoolExecutor()


async def health():
    ts = time.time()
    result = []
    log.info(f"[health] time elapsed: {time.time() - ts:.3f}s")
    return {"code": 0, "msg": "ok", "data": result}


async def ranking(param):
    ts = time.time()
    if not param.books:
        return {"code": 0, "message": "ok", "data": {"books": [], "model": model_name}}
    books = await asyncio.get_running_loop().run_in_executor(
        executor, get_ranked_books, param
    )
    log.info(
        f"[ranking] user: {param['user'].uid}, books_size: {len(books)}, time elapsed: {time.time() - ts:.3f}s"
    )
    return {"code": 0, "message": "ok", "data": {"books": books, "model": model_name}}


def get_ranked_books(param):
    inputs = build_infer_inputs(param)  # 需要返回Tensor
    try:
        with torch.no_grad():
            result = saved_model.predict(inputs)
        books_ranking = result["books_ranking"].cpu().numpy()
        book_index = np.argsort(-(books_ranking.reshape(-1)))
        return [param.books[i].bookid for i in book_index]
    except Exception as e:
        log.exception(e)
        return [book.bookid for book in param.books]



# 待写
def build_infer_inputs(param):

    likemarks = (
        param.user.likemarks.split(",")
        if param.user.likemarks
        else ["UNK"]
    )
    searchwords = (
        param.user.searchwords.split(",")
        if param.user.searchwords
        else ["UNK"]
    )

    # ========== 【关键修改】特征增强逻辑 ==========
    # 判断标准：searchwords 为 "UNK" (表示为空) 并且 likemarks 不是 "UNK" (表示有值)
    is_search_empty = (len(searchwords) == 1 and searchwords[0] == "UNK")
    is_likemarks_valid = not (len(likemarks) == 1 and likemarks[0] == "UNK")

    if is_search_empty and is_likemarks_valid:
        # log.info(f"Searchwords缺失，使用Likemarks填充: {likemarks}") # 可选日志
        searchwords = likemarks
    # ==========================================


    read_books = (
        param.user.readbooks.split(",")
        if param.user.readbooks
        else ["UNK"]
    )
    read_search_keywords_books_day_30 = (
        param.user.read_search_keywords_books_day_30.split(",")
        if param.user.read_search_keywords_books_day_30
        else ["UNK:UNK:0"]
    )

    # ========== 修复register_days类型问题 ==========
    # 1. 先统一转为字符串，再处理strip（兼容int/str类型）
    register_days_raw = param.user.register_days
    # 处理None/空值 + 统一转字符串 + 去空格
    register_days_str = str(register_days_raw).strip() if register_days_raw is not None else ""

    # 2. 若为空，用当前时间戳（整数）作为默认值，不赋值给protobuf字段
    if not register_days_str:
        default_register_ts = int(time.time())  # 整数时间戳
    else:
        # 3. 解析原始值，兼容字符串/整数类型
        try:
            default_register_ts = int(register_days_str)
        except ValueError:
            default_register_ts = int(time.time())  # 解析失败用默认值

    # 后续逻辑不变（基于default_register_ts计算）
    register_ts = default_register_ts
    register_ts = max(register_ts, 0)  # 避免负数时间戳
    current_utc_ts = time.time()
    beijing_offset = 8 * 3600
    current_beijing_ts = current_utc_ts + beijing_offset

    if register_ts > 0:
        register_days = ((current_beijing_ts - register_ts) / 86400) // 1
        register_days = max(0, int(register_days))
    else:
        register_days = 0

    # 分桶逻辑（不变）
    if register_days < 3:
        register_days_bucket = 0
    elif register_days < 8:
        register_days_bucket = 1
    elif register_days < 31:
        register_days_bucket = 2
    elif register_days < 181:
        register_days_bucket = 3
    else:
        register_days_bucket = 4

    uid_list = []
    likemarks_list = []
    searchwords_list = []
    read_books_list = []
    register_days_list = []
    books_list, words_list, chapters_list = [], [], []

    impression_bookid_list = []
    impression_bookmarks_list = []
    impression_bookname_list = []
    impression_booktag_list = []
    impression_bookwords_list = []
    impression_bookinfo_list = []
    for book in param.books:
        uid_list.append(param.user.uid)
        likemarks_list.append(likemarks[:4])
        searchwords_list.append(searchwords[:4])
        read_books_list.append(read_books[:4])
        register_days_list.append(register_days_bucket)
        book_list = []
        word_list = []
        chapter_list = []
        for data in read_search_keywords_books_day_30:
            split_str = data.split(":")
            word_list.append(split_str[0])
            book_list.append(split_str[1])
            try:
                # 对应TensorFlow版本的clip_by_value
                chapter_val = max(0, min(int(split_str[2]), 100))
                chapter_list.append(chapter_val)
            except (ValueError, TypeError, IndexError):
                chapter_list.append(0)
        words_list.append(word_list)
        books_list.append(book_list)
        chapters_list.append(chapter_list)
        impression_bookid_list.append(book.bookid)
        bookmarks = book.bookmarks.split(",")
        if len(bookmarks) < 4:
            bookmarks += ["UNK"] * (4 - len(bookmarks))
        impression_bookmarks_list.append(bookmarks[:4])

        bookname = book.bookname.split(",")
        if len(bookname) < 4:
            bookname += ["UNK"] * (4 - len(bookname))
        impression_bookname_list.append(bookname[:4])

        booktag = book.tag.split(",")
        if len(booktag) < 1:
            booktag += ["UNK"] * (1 - len(booktag))
        impression_booktag_list.append(booktag[:1])

        bookwords = book.bookwords.split(",")
        if len(bookwords) < 4:
            bookwords += ["UNK"] * (4 - len(bookwords))
        impression_bookwords_list.append(bookwords[:4])

        bookinfo = book.bookinfo.split(",")
        if len(bookinfo) < 4:
            bookinfo += ["UNK"] * (4 - len(bookinfo))
        impression_bookinfo_list.append(bookinfo[:4])

    inputs = {
        # 用户特征
        'uid': tf_hash_bucket(
            pl.Series("uid", uid_list),
            USER_HASH_BUCKET_SIZE,
            "cpu"
        ),
        'likemarks': word_lookup.lookup(likemarks_list),
        'searchwords': word_lookup.lookup(searchwords_list),
        'read_books': book_lookup.lookup(read_books_list),
        'register_time': torch.tensor(
            register_days_list,
            dtype=torch.long,
            device="cpu"
        ),
        'read_search_keywords_books_day_30': {
            "word": word_lookup.lookup(words_list),
            "book": book_lookup.lookup(books_list),
            "chapter": torch.tensor(chapters_list, dtype=torch.long, device="cpu"),
        },

        # 书籍特征
        'impression_bookid': book_lookup.lookup(impression_bookid_list),
        'impression_bookmarks': word_lookup.lookup(impression_bookmarks_list),
        'impression_bookname': word_lookup.lookup(impression_bookname_list),
        'impression_tag': word_lookup.lookup(impression_booktag_list),
        'impression_bookwords': word_lookup.lookup(impression_bookwords_list),
        'impression_bookinfo': word_lookup.lookup(impression_bookinfo_list),
    }

    log.info(f'user: {param.user.uid},likemarks: {likemarks_list},searchwords: {searchwords_list}')

    return inputs


def download_model():
    oss.download_last_data_type(model_name, oss.DataType.MODEL, args.model_dir)


def process_env():
    log.info("process env start!")
    if sys.platform != "linux":
        # 开发环境调试自己的数据路径
        args.model_dir = get_local_path(model_name) + '/keras_export'
        args.download_model = False
        log.info("当前系统是开发或测试环境")
    log.info("process env done!")


class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        # 处理健康检查请求
        self.send_response(200)
        self.end_headers()
        ts = time.time()
        log.info(f"[health]  time elapsed: {time.time() - ts:.3f}s")
        self.wfile.write(b"im OK")


class ModelServicer(model_pb2_grpc.GreeterServicer):

    async def Match(self, request, context):
        return model_pb2.MatchResponse(message='Hello, %s!' % request)

    async def Ranking(self, request, context):
        try:
            if not request.books:
                return model_pb2.RankingResponse(code=0, message="ok",
                                                 data=model_pb2.RankingData(model=model_name, books=[]))
            ts = time.time()
            books = await asyncio.get_running_loop().run_in_executor(
                executor, get_ranked_books, request
            )
            log.info(
                f"[ranking] user: {request.user.uid}, size: {len(books)}, time elapsed: {time.time() - ts:.3f}s"
            )
            return model_pb2.RankingResponse(code=0, message="ok",
                                             data=model_pb2.RankingData(model=model_name, books=books))
        except Exception as e:
            logging.exception(e)
            return model_pb2.RankingResponse(code=1, message=str(e),
                                             data=model_pb2.RankingData(model=model_name, books=[]))


def run_http_server():
    httpd = HTTPServer(('0.0.0.0', 8080), HealthCheckHandler)
    httpd.serve_forever()


async def serve():
    server = grpc.aio.server()
    model_pb2_grpc.add_GreeterServicer_to_server(ModelServicer(), server)
    server.add_insecure_port(f'[::]:{args.port}')
    await server.start()

    threading.Thread(target=run_http_server).start()
    try:
        await server.wait_for_termination()
    except KeyboardInterrupt:
        await server.stop(0)


def main():
    process_env()
    if args.download_model:
        download_model()
    startup_event()
    asyncio.run(serve())


if __name__ == "__main__":
    main()
