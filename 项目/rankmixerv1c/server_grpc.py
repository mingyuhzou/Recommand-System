import argparse
import asyncio
import json
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
from hbre_book.model.rankmixerv1c.preprocess import VocabLookupTable
from hbre_book.model.rankmixerv1c.rankmixer import RankMixer
from hbre_book.model.rankmixerv1c.util import (
    tf_hash_bucket, merge_searchwords_with_day_last, merge_readbooks_with_day_last,
)
import jieba.analyse
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

def extract_keywords(text: str, topk: int) -> list[str]:
    return jieba.analyse.extract_tags(str(text) or "", topk)

def join_as_text(value):
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return " ".join(str(x) for x in value if x is not None and str(x).strip() != "")
    return str(value)

def startup_event():
    global book_lookup, word_lookup

    book_table = pl.read_parquet(
        args.model_dir + "/book_counter.parquet", columns=["bookid"]
    )["bookid"].to_list()
    word_table = pl.read_parquet(
        args.model_dir + "/word_counter.parquet", columns=["word"]
    )["word"].to_list()
    book_lookup = VocabLookupTable(book_table, "cpu")
    word_lookup = VocabLookupTable(word_table, "cpu")

    quartile_path = os.path.join(args.model_dir , "numeric_quartiles.json")
    with open(quartile_path, "r", encoding="utf-8") as f:
        numeric_quartiles = json.load(f)

    global saved_model
    ts = time.time()

    saved_model = RankMixer(
        book_vocab_size=len(book_table),
        word_vocab_size=len(word_table),
        embed_dim=64,
        token_dim=128,
        num_tokens=8,
        num_layers=2,
        hidden_ratio=4.0,
        dropout=0.1,
    )

    state = torch.load(
        os.path.join(args.model_dir, "model.pth"),
        map_location="cpu",
        weights_only=True,
    )
    saved_model.load_state_dict(state)
    saved_model = saved_model.float().to("cpu")
    saved_model.eval()

    log.info(f"load model time elapsed: {time.time() - ts:.3f}s")

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

def build_infer_inputs(param):
    user = param.user
    n = len(param.books)

    def _print_user_all_fields():
        print("\n" + "=" * 80)
        print(f"[USER 全字段] uid={getattr(user, 'uid', None)}")
        print("=" * 80)

        for field in user.DESCRIPTOR.fields:
            try:
                value = getattr(user, field.name)
                print(f"user.{field.name} = {value}")
            except Exception as e:
                print(f"user.{field.name} = <error: {e}>")

        print("=" * 80 + "\n")

    _print_user_all_fields()

    quartile_path = os.path.join(args.model_dir, "numeric_quartiles.json")
    with open(quartile_path, "r", encoding="utf-8") as f:
        numeric_quartiles = json.load(f)

    book_type_map = {
        "UNK": 0,
        "TEMPLATE": 1,
        "HUABEN": 2,
        "HUABENV1": 3,
        "PREMIUM_SHORT": 4,
        "SHORT": 5,
    }

    plan_type_map = {
        "UNK": 0,
        "PC": 1,
        "IOS": 2,
        "IMITATION": 3,
        "ANDROID": 4,
    }

    os_map = {
        "android": 1,
        "ios": 2,
        "harmonyos": 3,
    }

    def bucket_by_log_quartile(x, q1, q2, q3):
        try:
            x = float(x)
        except (TypeError, ValueError):
            x = 0.0

        x = max(0.0, x)
        x = np.log1p(x)

        if x <= q1:
            return 0
        elif x <= q2:
            return 1
        elif x <= q3:
            return 2
        else:
            return 3

    def pad_list(seq, size, pad_value):
        seq = list(seq[:size])
        if len(seq) < size:
            seq.extend([pad_value] * (size - len(seq)))
        return seq

    def safe_split(value, sep=","):
        if value is None:
            return []
        if not isinstance(value, str):
            value = str(value)
        parts = [x.strip() for x in value.split(sep)]
        return [x for x in parts if x != ""]

    likemarks = safe_split(getattr(param.user, "likemarks", None))
    if not likemarks:
        likemarks = ["UNK"]

    searchwords = merge_searchwords_with_day_last(
        getattr(param.user, "searchwords", None),
        getattr(param.user, "read_search_keywords_books_day_last", None),
    )

    top_read_books = safe_split(getattr(param.user, "top_read_books", None))
    if not top_read_books:
        top_read_books = ["UNK"]

    is_search_empty = (len(searchwords) == 1 and searchwords[0] == "UNK")
    is_likemarks_valid = not (len(likemarks) == 1 and likemarks[0] == "UNK")
    if is_search_empty and is_likemarks_valid:
        searchwords = likemarks.copy()

    read_books = merge_readbooks_with_day_last(
        getattr(param.user, "readbooks", None),
        getattr(param.user, "read_books_day_last", None),
    )

    read_search_keywords_books_day_30 = safe_split(
        getattr(param.user, "read_search_keywords_books_day_30", None)
    )
    if not read_search_keywords_books_day_30:
        read_search_keywords_books_day_30 = ["UNK:UNK:0"]

    register_days_raw = getattr(param.user, "register_days", None)
    register_days_str = str(register_days_raw).strip() if register_days_raw is not None else ""

    if not register_days_str:
        default_register_ts = int(time.time())
    else:
        try:
            default_register_ts = int(register_days_str)
        except ValueError:
            default_register_ts = int(time.time())

    register_ts = max(default_register_ts, 0)
    current_utc_ts = time.time()
    beijing_offset = 8 * 3600
    current_beijing_ts = current_utc_ts + beijing_offset

    if register_ts > 0:
        register_days = ((current_beijing_ts - register_ts) / 86400) // 1
        register_days = max(0, int(register_days))
    else:
        register_days = 0

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
    top_read_books_list = []
    read_books_list = []
    register_days_list = []
    os_list = []
    books_list, words_list, chapters_list = [], [], []

    impression_bookid_list = []
    impression_bookmarks_list = []
    impression_bookname_list = []
    impression_booktag_list = []
    impression_bookinfo_list = []
    impression_wordcount_list = []

    impression_shelfcount_total_list = []
    impression_clickcount_total_list = []
    impression_punch_total_list = []
    impression_rewardcount_total_list = []
    impression_contractstatus_list = []

    impression_book_type_list = []
    impression_plan_type_list = []

    user_os = getattr(param.user, "os", "UNK")
    user_os = str(user_os).strip().lower() if user_os is not None else "unk"
    user_os_id = os_map.get(user_os, 0)

    for book in param.books:
        uid_list.append(param.user.uid)

        # 用户侧长度统一改为 4
        likemarks_list.append(pad_list(likemarks, 8, "UNK"))
        searchwords_list.append(pad_list(searchwords, 8, "UNK"))
        top_read_books_list.append(pad_list(top_read_books, 8, "UNK"))
        read_books_list.append(pad_list(read_books, 8, "UNK"))

        register_days_list.append(register_days_bucket)
        os_list.append(user_os_id)

        book_list = []
        word_list = []
        chapter_list = []
        for data in read_search_keywords_books_day_30[:8]:
            split_str = str(data).split(":")
            if len(split_str) >= 2:
                word_list.append(split_str[0] if split_str[0] else "UNK")
                book_list.append(split_str[1] if split_str[1] else "UNK")
            else:
                word_list.append("UNK")
                book_list.append("UNK")

            try:
                chapter_val = max(0, min(int(split_str[2]), 100))
                chapter_list.append(chapter_val)
            except (ValueError, TypeError, IndexError):
                chapter_list.append(0)

        words_list.append(pad_list(word_list, 8, "UNK"))
        books_list.append(pad_list(book_list, 8, "UNK"))
        chapters_list.append(pad_list(chapter_list, 8, 0))

        impression_bookid_list.append(str(book.bookid))

        bookmarks = safe_split(getattr(book, "bookmarks", None))
        if not bookmarks:
            bookmarks = ["UNK"]
        impression_bookmarks_list.append(pad_list(bookmarks, 4, "UNK"))

        bookname_raw = getattr(book, "bookname", None)
        bookname_text = join_as_text(bookname_raw)
        bookname_keywords = extract_keywords(bookname_text, 4)
        if not bookname_keywords:
            bookname_keywords = ["UNK"]
        impression_bookname_list.append(pad_list(bookname_keywords, 4, "UNK"))

        bookinfo_raw = getattr(book, "bookinfo", None)
        bookinfo_text = join_as_text(bookinfo_raw)
        bookinfo_keywords = extract_keywords(bookinfo_text, 8)
        if not bookinfo_keywords:
            bookinfo_keywords = ["UNK"]
        impression_bookinfo_list.append(pad_list(bookinfo_keywords, 8, "UNK"))

        booktag = safe_split(getattr(book, "tag", None))
        if not booktag:
            booktag = ["UNK"]
        impression_booktag_list.append(pad_list(booktag, 1, "UNK"))

        raw_wc = getattr(book, "wordcount", 0)
        try:
            wc = int(raw_wc)
        except (TypeError, ValueError):
            wc = 0

        wc = max(0, min(wc, 400000))

        if wc < 10000:
            wc_bucket = 0
        elif wc < 30000:
            wc_bucket = 1
        elif wc < 60000:
            wc_bucket = 2
        elif wc < 100000:
            wc_bucket = 3
        elif wc < 200000:
            wc_bucket = 4
        else:
            wc_bucket = 5
        impression_wordcount_list.append(wc_bucket)

        impression_shelfcount_total_list.append(
            bucket_by_log_quartile(
                getattr(book, "shelfcount_total", 0),
                *numeric_quartiles["impression_shelfcount_total"]
            )
        )
        impression_clickcount_total_list.append(
            bucket_by_log_quartile(
                getattr(book, "clickcount_total", 0),
                *numeric_quartiles["impression_clickcount_total"]
            )
        )
        impression_punch_total_list.append(
            bucket_by_log_quartile(
                getattr(book, "punch_total", 0),
                *numeric_quartiles["impression_punch_total"]
            )
        )
        impression_rewardcount_total_list.append(
            bucket_by_log_quartile(
                getattr(book, "rewardcount_total", 0),
                *numeric_quartiles["impression_rewardcount_total"]
            )
        )

        raw_book_type = getattr(book, "book_type", "UNK")
        book_type = str(raw_book_type).replace('"', "").strip() if raw_book_type is not None else "UNK"
        impression_book_type_list.append(book_type_map.get(book_type, 0))

        raw_plan_type = getattr(book, "plan_type", "UNK")
        plan_type = str(raw_plan_type).replace('"', "").strip() if raw_plan_type is not None else "UNK"
        impression_plan_type_list.append(plan_type_map.get(plan_type, 0))

        raw_contractstatus = getattr(book, "contractstatus", None)
        try:
            contractstatus = int(raw_contractstatus)
        except (TypeError, ValueError):
            contractstatus = -1
        impression_contractstatus_list.append(contractstatus + 1)

        raw_chapters = getattr(book, "chapters", 0)
        try:
            chapters = int(raw_chapters)
        except (TypeError, ValueError):
            chapters = 0

        if len(uid_list) == 1:
            print("========== DEBUG FIRST SAMPLE ==========")

            print("\n==== USER ====")

            print("uid:")
            print("  raw      :", param.user.uid)

            print("likemarks:")
            print("  raw      :", getattr(param.user, "likemarks", None))
            print("  processed:", likemarks_list[0])

            print("searchwords:")
            print("  raw      :", getattr(param.user, "searchwords", None))
            print("  processed:", searchwords_list[0])

            print("top_read_books:")
            print("  raw      :", getattr(param.user, "top_read_books", None))
            print("  processed:", top_read_books_list[0])

            print("readbooks:")
            print("  raw      :", getattr(param.user, "readbooks", None))
            print("  processed:", read_books_list[0])

            print("read_search_keywords_books_day_30:")
            print("  raw      :", getattr(param.user, "read_search_keywords_books_day_30", None))
            print("  word     :", words_list[0])
            print("  book     :", books_list[0])
            print("  chapter  :", chapters_list[0])

            print("register_days:")
            print("  raw      :", getattr(param.user, "register_days", None))
            print("  bucket   :", register_days_list[0])

            print("os:")
            print("  raw      :", getattr(param.user, "os", None))
            print("  id       :", os_list[0])

            print("\n==== ITEM ====")

            print("bookid:")
            print("  raw      :", getattr(book, "bookid", None))
            print("  processed:", impression_bookid_list[0])

            print("bookmarks:")
            print("  raw      :", getattr(book, "bookmarks", None))
            print("  processed:", impression_bookmarks_list[0])

            print("bookname:")
            print("  raw      :", getattr(book, "bookname", None))
            print("  text     :", join_as_text(getattr(book, "bookname", None)))
            print("  keywords :", extract_keywords(join_as_text(getattr(book, "bookname", None)), 4))
            print("  processed:", impression_bookname_list[0])

            print("bookinfo:")
            print("  raw      :", getattr(book, "bookinfo", None))
            print("  text     :", join_as_text(getattr(book, "bookinfo", None)))
            print("  keywords :", extract_keywords(join_as_text(getattr(book, "bookinfo", None)), 8))
            print("  processed:", impression_bookinfo_list[0])

            print("tag:")
            print("  raw      :", getattr(book, "tag", None))
            print("  processed:", impression_booktag_list[0])

            print("wordcount:")
            print("  raw      :", getattr(book, "wordcount", None))
            print("  bucket   :", impression_wordcount_list[0])

            print("shelfcount_total:")
            print("  raw      :", getattr(book, "shelfcount_total", None))
            print("  bucket   :", impression_shelfcount_total_list[0])

            print("clickcount_total:")
            print("  raw      :", getattr(book, "clickcount_total", None))
            print("  bucket   :", impression_clickcount_total_list[0])

            print("punch_total:")
            print("  raw      :", getattr(book, "punch_total", None))
            print("  bucket   :", impression_punch_total_list[0])

            print("rewardcount_total:")
            print("  raw      :", getattr(book, "rewardcount_total", None))
            print("  bucket   :", impression_rewardcount_total_list[0])

            print("book_type:")
            print("  raw      :", getattr(book, "book_type", None))
            print("  id       :", impression_book_type_list[0])

            print("plan_type:")
            print("  raw      :", getattr(book, "plan_type", None))
            print("  id       :", impression_plan_type_list[0])

            print("contractstatus:")
            print("  raw      :", getattr(book, "contractstatus", None))
            print("  id       :", impression_contractstatus_list[0])

            print("=========================================")

    inputs = {
        "uid": tf_hash_bucket(
            pl.Series("uid", uid_list),
            USER_HASH_BUCKET_SIZE,
            "cpu"
        ),
        "likemarks": word_lookup.lookup(likemarks_list),
        "searchwords": word_lookup.lookup(searchwords_list),
        "top_read_books": book_lookup.lookup(top_read_books_list),
        "read_books": book_lookup.lookup(read_books_list),
        "register_time": torch.tensor(
            register_days_list,
            dtype=torch.long,
            device="cpu"
        ),
        "os": torch.tensor(
            os_list,
            dtype=torch.long,
            device="cpu"
        ),
        "read_search_keywords_books_day_30": {
            "word": word_lookup.lookup(words_list),
            "book": book_lookup.lookup(books_list),
            "chapter": torch.tensor(chapters_list, dtype=torch.long, device="cpu"),
        },

        "impression_bookid": book_lookup.lookup(impression_bookid_list),
        "impression_bookmarks": word_lookup.lookup(impression_bookmarks_list),
        "impression_bookname": word_lookup.lookup(impression_bookname_list),
        "impression_tag": word_lookup.lookup(impression_booktag_list),
        "impression_bookinfo": word_lookup.lookup(impression_bookinfo_list),
        "impression_wordcount": torch.tensor(
            impression_wordcount_list,
            dtype=torch.long,
            device="cpu"
        ),

        "impression_shelfcount_total": torch.tensor(
            impression_shelfcount_total_list,
            dtype=torch.long,
            device="cpu"
        ),
        "impression_clickcount_total": torch.tensor(
            impression_clickcount_total_list,
            dtype=torch.long,
            device="cpu"
        ),
        "impression_punch_total": torch.tensor(
            impression_punch_total_list,
            dtype=torch.long,
            device="cpu"
        ),
        "impression_rewardcount_total": torch.tensor(
            impression_rewardcount_total_list,
            dtype=torch.long,
            device="cpu"
        ),

        "impression_book_type": torch.tensor(
            impression_book_type_list,
            dtype=torch.long,
            device="cpu"
        ),
        "impression_plan_type": torch.tensor(
            impression_plan_type_list,
            dtype=torch.long,
            device="cpu"
        ),
        "impression_contractstatus": torch.tensor(
            impression_contractstatus_list,
            dtype=torch.long,
            device="cpu"
        ),
    }

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
