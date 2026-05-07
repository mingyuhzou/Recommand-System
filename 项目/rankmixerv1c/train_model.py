import argparse
import logging as log
import os
import platform
import sys
import time
import zipfile
import shutil
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from hbre_book.ability import oss
from hbre_book.model.common import SampleRatio, Sample
from hbre_book.model.rankmixerv1c.rankmixer import RankMixer
from hbre_book.model.rankmixerv1c.preprocess import (
    VocabLookupTable,
    BookRecommendationIterableDataset,
)
from hbre_book.model.rankmixerv1c.util import (
    move_to_device,
    tf_hash_bucket,
    BackgroundPrefetch,
    ensure_train_size_polars
)
from hbre_book.model.rankmixerv1c.cache_io import CacheWriter, CacheReader, load_manifest
from hbre_book.util.project import get_local_path
from hbre_book.util.timer import StepProfiler

# 设置logging
log.basicConfig(level=log.INFO)
logger = log.getLogger(__name__)
profiler = StepProfiler(logger=log)

os.environ["TZ"] = "Asia/Shanghai"
time.tzset()

# -------------------- 参数解析 --------------------
model_name = Path(__file__).resolve().parent.name
log.info(f"当前模型：「{model_name}」")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_dir", type=str, default=f"/work/hbre/data/{model_name}", help="数据目录"
)
parser.add_argument(
    "--common_data_dir",
    type=str,
    default="/work/hbre/data/common",
    help="common目录",
)
parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
parser.add_argument("--days", type=int, default=35, help="训练数据天数")
parser.add_argument(
    "--download_train_samples", action="store_true", help="是否下载训练数据"
)
parser.add_argument("--download_common", action="store_true", help="是否下载common数据")
parser.add_argument("--upload_to_oss", action="store_true", help="是否上传数据到oss")
args = parser.parse_args()

# 训练常量
BATCH_SIZE = 1024  # 对应TensorFlow版本的FLAGS.batch_size
USER_HASH_BUCKET_SIZE = 2 ** 18  # 对应TensorFlow版本的uid哈希桶
AUTHOR_HASH_BUCKET_SIZE = 2 ** 16  # 作者ID哈希桶
EARLY_STOPPING_PATIENCE = 1  # 对应TensorFlow版本的早停耐心
INFERENCE_BATCH_SIZE = 32
TARGET_TRAIN_SIZE = 9000000

PRETRAIN_WORD_FILENAME = 'word_embeddings_64.npz'
PRETRAIN_BOOK_FILENAME = 'book_embeddings_64.npz'

# 定义训练中需要的特征（对应TensorFlow版本的feature_columns）
feature_columns = [
    # ===== 用户侧 =====
    "uid",
    "os",
    "register_time",
    "likemarks",
    "searchwords",
    "read_books",
    "top_read_books",
    "read_search_keywords_books_day_30",

    # ===== 正样本 =====
    "impression_bookid",
    "impression_bookmarks",
    "impression_bookname",
    "impression_tag",
    "impression_bookinfo",
    "impression_wordcount",
    "impression_chapters",
    "impression_book_type",
    "impression_plan_type",
    "impression_contractstatus",
    "impression_shelfcount_total",
    "impression_clickcount_total",
    "impression_punch_total",
    "impression_rewardcount_total",

    # ===== 负样本 =====
    "negative_bookid",
    "negative_bookmarks",
    "negative_bookname",
    "negative_tag",
    "negative_bookinfo",
    "negative_wordcount",
    "negative_chapters",
    "negative_book_type",
    "negative_plan_type",
    "negative_contractstatus",
    "negative_shelfcount_total",
    "negative_clickcount_total",
    "negative_punch_total",
    "negative_rewardcount_total",

]


def get_export_dir() -> str:
    """获取导出目录"""
    return os.path.join(args.data_dir, "keras_export")


def pretrainnpz_to_dic(path: str, filename: str) -> Dict[str, np.ndarray]:
    """
    将预训练的npz文件转换为字典 - 对应TensorFlow版本的函数

    Args:
        path: 文件路径
        filename: 文件名

    Returns:
        嵌入字典 {item_id: vector}
    """
    embeddings_index = {}

    try:
        file_path = os.path.join(path, filename)
        log.info(f"正在加载预训练嵌入: {file_path}")

        wordVectors = np.load(file_path)
        items = wordVectors["keys"]  # 词汇/书籍ID列表
        vectors = wordVectors["vectors"]  # 对应的向量矩阵

        log.info(f"预训练嵌入数量: {len(items)} (文件: {filename})")

        for i, item in enumerate(items):
            embeddings_index[str(item)] = vectors[i]  # 确保key为字符串

        return embeddings_index

    except Exception as e:
        log.error(f"加载预训练嵌入失败: {filename}, 错误: {e}")
        return {}


def build_embedding_matrix_from_pretrained(vocabulary: List[str],
                                           embedding_index: Dict[str, np.ndarray],
                                           embed_dim: int,
                                           vocab_type: str = "word") -> np.ndarray:
    """
    从预训练嵌入构建对齐的嵌入矩阵 - 对应TensorFlow版本的逻辑

    Args:
        vocabulary: 目标词汇表
        embedding_index: 预训练嵌入字典
        embed_dim: 嵌入维度
        vocab_type: 词汇类型 ("word" 或 "book")

    Returns:
        对齐后的嵌入矩阵 [vocab_size, embed_dim]
    """
    vocab_size = len(vocabulary)

    # 1. 随机初始化嵌入矩阵（对应TensorFlow版本的random.standard_normal）
    embedding_matrix = np.random.standard_normal(size=(vocab_size, embed_dim)).astype(np.float32)

    # 2. 从预训练嵌入中查找并替换
    find_num = 0
    log.info(f"total {vocab_size} {vocab_type}s")

    for i, item in enumerate(vocabulary):
        embedding_vector = embedding_index.get(str(item))
        if embedding_vector is not None:
            # 找到预训练向量，直接替换
            if len(embedding_vector) == embed_dim:
                embedding_matrix[i] = embedding_vector
                find_num += 1
            elif len(embedding_vector) < embed_dim:
                # 维度不足，填充0
                embedding_matrix[i, :len(embedding_vector)] = embedding_vector
                embedding_matrix[i, len(embedding_vector):] = 0.0
                find_num += 1
            else:
                # 维度过大，截断
                embedding_matrix[i] = embedding_vector[:embed_dim]
                find_num += 1

    log.info(f"在预训练嵌入中找到 {find_num} 个{vocab_type}")

    # 3. 归一化所有向量（对应TensorFlow版本的归一化逻辑）
    norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
    # 避免除零
    norms = np.where(norms == 0, 1.0, norms)
    embedding_matrix = embedding_matrix / norms

    log.info(f"{vocab_type}嵌入矩阵归一化完成")

    return embedding_matrix


def load_and_build_pretrained_embeddings(common_data_dir: str,
                                         book_table: List[str],
                                         word_table: List[str]) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    完整的预训练嵌入加载和构建流程

    Args:
        common_data_dir: 公共数据目录
        book_table: 书籍词汇表
        word_table: 词汇表

    Returns:
        (word_weights, book_weights) - 构建好的嵌入权重张量
    """
    log.info("=" * 60)
    log.info("开始加载和构建预训练嵌入...")
    log.info("=" * 60)

    word_weights = None
    book_weights = None

    # 1. 检查预训练文件是否存在
    word_pretrain_path = os.path.join(common_data_dir, PRETRAIN_WORD_FILENAME)
    book_pretrain_path = os.path.join(common_data_dir, PRETRAIN_BOOK_FILENAME)

    if not os.path.exists(word_pretrain_path):
        log.error(f"词汇预训练文件不存在: {word_pretrain_path}")
    if not os.path.exists(book_pretrain_path):
        log.error(f"书籍预训练文件不存在: {book_pretrain_path}")

    # 2. 加载预训练嵌入索引（对应TensorFlow版本的embedding_index）
    word_embedding_index = pretrainnpz_to_dic(common_data_dir, PRETRAIN_WORD_FILENAME)
    book_embedding_index = pretrainnpz_to_dic(common_data_dir, PRETRAIN_BOOK_FILENAME)

    # 3. 构建词汇嵌入矩阵
    if word_embedding_index:
        log.info("构建词汇嵌入矩阵...")
        word_embedding_matrix = build_embedding_matrix_from_pretrained(
            vocabulary=word_table,
            embedding_index=word_embedding_index,
            embed_dim=64,
            vocab_type="word"
        )
        word_weights = torch.from_numpy(word_embedding_matrix)

        # 保存为TensorFlow兼容格式
        word_matrix_path = os.path.join(args.data_dir, "word_embedding_matrix.npz")
        np.savez(word_matrix_path, embedding_matrix=word_embedding_matrix)
        log.info(f"词汇嵌入矩阵已保存: {word_matrix_path}")

    else:
        log.warning(f"未找到词汇预训练嵌入: {PRETRAIN_WORD_FILENAME}")

    # 4. 构建书籍嵌入矩阵
    if book_embedding_index:
        log.info("构建书籍嵌入矩阵...")
        book_embedding_matrix = build_embedding_matrix_from_pretrained(
            vocabulary=book_table,
            embedding_index=book_embedding_index,
            embed_dim=64,
            vocab_type="book"
        )
        book_weights = torch.from_numpy(book_embedding_matrix)

        # 保存为TensorFlow兼容格式
        book_matrix_path = os.path.join(args.data_dir, "book_embedding_matrix.npz")
        np.savez(book_matrix_path, embedding_matrix=book_embedding_matrix)
        log.info(f"书籍嵌入矩阵已保存: {book_matrix_path}")

    else:
        log.warning(f"未找到书籍预训练嵌入: {PRETRAIN_BOOK_FILENAME}")

    # 5. 打印加载总结
    log.info("=" * 60)
    log.info("预训练嵌入加载总结:")
    if word_weights is not None:
        log.info(f"  ✓ 词汇嵌入: {word_weights.shape} (已归一化)")
    else:
        log.info(f"  ✗ 词汇嵌入: 加载失败")

    if book_weights is not None:
        log.info(f"  ✓ 书籍嵌入: {book_weights.shape} (已归一化)")
    else:
        log.info(f"  ✗ 书籍嵌入: 加载失败")
    log.info("=" * 60)

    return word_weights, book_weights


def bpr_loss(score_list: List[torch.Tensor]) -> torch.Tensor:

    pos_score, neg_score = score_list[0], score_list[1]

    loss = torch.mean(F.softplus(-(pos_score - neg_score)))

    return loss


def custom_accuracy(score_list: List[torch.Tensor]) -> torch.Tensor:
    """
    自定义准确率 - 对应TensorFlow版本

    Args:
        score_list: [pos_score, neg_score]

    Returns:
        准确率
    """
    pos_score, neg_score = score_list[0], score_list[1]

    # pos > neg_hard and pos > neg_easy
    predictions = (pos_score > neg_score).float()

    true_labels = torch.ones_like(predictions)

    accuracy = torch.mean((predictions == true_labels).float())

    return accuracy


def train(
        train_loader_epoch0,
        evaluate_loader,
        book_tables_length,
        word_tables_length,
        device,
        cache_dir,
        book_table,
        word_table
):
    """

    Args:
        train_loader_epoch0: 第一个epoch的训练数据加载器
        evaluate_loader: 验证数据加载器
        book_tables_length: 书籍表长度
        word_tables_length: 词表长度
        device: 训练设备
        cache_dir: 缓存目录
        book_table: 书籍词汇表
        word_table: 词汇表
    """
    cached_train_loader = None

    # 创建模型
    model = RankMixer(
        book_vocab_size=book_tables_length,
        word_vocab_size=word_tables_length,
        embed_dim=64,
        token_dim=128,
        num_tokens=8,
        num_layers=2,
        hidden_ratio=4.0,
        dropout=0.1,
    ).to(device)

    # 加载并构建预训练权重
    word_weights, book_weights = load_and_build_pretrained_embeddings(
        args.common_data_dir, book_table, word_table
    )

    # 将预训练权重加载到模型中
    if word_weights is not None or book_weights is not None:
        log.info("正在将预训练权重注入模型...")
        model.load_pretrained_weights(word_weights=word_weights, book_weights=book_weights)

        # 统计使用情况
        pretrain_status = []
        if word_weights is not None:
            pretrain_status.append("词汇嵌入(预训练+归一化)")
        else:
            pretrain_status.append("词汇嵌入(随机初始化)")

        if book_weights is not None:
            pretrain_status.append("书籍嵌入(预训练+归一化)")
        else:
            pretrain_status.append("书籍嵌入(随机初始化)")

        log.info(f"权重加载完成: {' + '.join(pretrain_status)}")
    else:
        log.warning("未找到任何预训练权重，使用随机初始化")

    # 对应TensorFlow版本的Adam优化器
    optimizer = optim.Adam(model.parameters(), lr=0.0003, weight_decay=0)

    best_auc = float("-inf")

    if device == "cuda":
        # [修改] 为了对齐 CPU 逻辑和精度，暂时关闭 AMP
        scaler = None
        use_amp = False
        log.info("启用CUDA全精度训练 (FP32) - 禁用AMP以对齐CPU逻辑")
    else:
        scaler = None
        use_amp = False
        log.info("使用CPU全精度训练 (FP32)")

    wait = 0

    for epoch in range(args.epochs):
        model.train()
        log.info(f"\nStart of epoch {epoch}")

        # 选择数据源
        if epoch == 0:
            log.info("📝 Epoch 0: 从Parquet读取并写入缓存")
            train_loader = train_loader_epoch0
        else:
            if cached_train_loader is None:
                log.info("💾 创建缓存读取器")
                manifest = load_manifest(cache_dir)
                if manifest and manifest.get('num_shards', 0) > 0:
                    cached_train_loader = DataLoader(
                        CacheReader(cache_dir, shuffle_files=True),
                        batch_size=None,
                        num_workers=0,
                        pin_memory=(device == "cuda"),
                    )
                    log.info(f"✓ 缓存加载成功: {manifest.get('num_shards')} 个shard")
                else:
                    log.warning("⚠️ 缓存无效，回退到Parquet读取")
                    cached_train_loader = train_loader_epoch0

            train_loader = cached_train_loader

        epoch_start = time.time()
        total_step_time = 0
        step_count = 0
        epoch_loss = 0.0

        data_wait = 0.0
        next_batch_ready_t = time.time()

        for step, inputs in enumerate(BackgroundPrefetch(train_loader, max_prefetch=3)):
            data_wait += time.time() - next_batch_ready_t
            inputs = move_to_device(inputs, device)
            step_start = time.time()

            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                if device == "cuda":
                    with torch.amp.autocast("cuda"):
                        # 模型前向传播
                        pos_score = model(inputs, flag="impression")
                        neg_score = model(inputs, flag="negative")

                        loss_value = bpr_loss([pos_score, neg_score])
                        acc = custom_accuracy([pos_score, neg_score])

                        scaler.scale(loss_value).backward()
                        scaler.step(optimizer)
                        scaler.update()
                elif device == "mps":
                    with torch.amp.autocast(device_type="mps", dtype=torch.float16):
                        pos_score = model(inputs, flag="impression")
                        neg_score = model(inputs, flag="negative")

                        loss_value = bpr_loss([pos_score, neg_score])
                        acc = custom_accuracy([pos_score, neg_score])

                        scaler.scale(loss_value).backward()
                        scaler.step(optimizer)
                        scaler.update()
            else:
                # CPU训练
                pos_score = model(inputs, flag="impression")
                neg_score = model(inputs, flag="negative")

                loss_value = bpr_loss([pos_score, neg_score])
                acc = custom_accuracy([pos_score, neg_score])

                loss_value.backward()
                optimizer.step()

            epoch_loss += loss_value.item()
            step_time = time.time() - step_start
            total_step_time += step_time
            step_count += 1

            # 定期打印训练状态（对应TensorFlow版本的1000步）
            if step % 1000 == 0:
                log.info(
                    f"Train--> step:{step}, loss:{loss_value.item():.4f}, acc:{acc.item():.4f}")
                diff = (pos_score - neg_score).detach()
                log.info(
                    f"Train--> step:{step}, "
                    f"loss:{loss_value.item():.6f}, "
                    f"acc:{acc.item():.4f}, "
                    f"pos_mean:{pos_score.mean().item():.4f}, "
                    f"neg_mean:{neg_score.mean().item():.4f}, "
                    f"diff_mean:{diff.mean().item():.4f}, "
                    f"diff_min:{diff.min().item():.4f}, "
                    f"diff_max:{diff.max().item():.4f}"
                )

            next_batch_ready_t = time.time()

        avg_train_loss = epoch_loss / step_count if step_count > 0 else 0.0

        log.info(f"data_wait={data_wait:.2f}s")
        training_time = time.time() - epoch_start
        log.info(f"训练阶段总耗时: {training_time:.2f}s")
        log.info(f"所有step累计耗时: {total_step_time:.2f}s")
        log.info(f"平均训练损失: {avg_train_loss:.4f}")

        # 验证阶段
        log.info("=" * 80)
        eval_start = time.time()
        model.eval()
        all_scores = []
        all_labels = []
        total_val_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0

        with torch.no_grad():
            for eval_step, eval_inputs in enumerate(evaluate_loader):
                eval_inputs = move_to_device(eval_inputs, device)

                pos_score = model(eval_inputs, flag="impression")
                neg_score = model(eval_inputs, flag="negative")

                val_loss = bpr_loss([pos_score, neg_score])
                val_acc = custom_accuracy([pos_score, neg_score])

                total_val_loss += val_loss.item()
                total_accuracy += val_acc.item()
                num_batches += 1

                all_scores.extend(pos_score.detach().cpu().view(-1).tolist())
                all_labels.extend([1] * pos_score.numel())

                all_scores.extend(neg_score.detach().cpu().view(-1).tolist())
                all_labels.extend([0] * neg_score.numel())

        avg_val_loss = total_val_loss / num_batches if num_batches > 0 else 0.0

        if len(set(all_labels)) < 2:
            auc = 0.0
        else:
            auc = roc_auc_score(all_labels, all_scores)

        log.info(f"Epoch:{epoch}, Val Loss: {avg_val_loss:.4f}, AUC: {auc:.4f}")

        eval_time = time.time() - eval_start
        log.info(f"验证阶段总耗时: {eval_time:.2f}s")

        avg_val_loss = total_val_loss / num_batches if num_batches > 0 else 0.0
        average_accuracy = total_accuracy / num_batches if num_batches > 0 else 0.0

        log.info(
            f"Epoch:{epoch}, "
            f"Val Loss:{avg_val_loss:.6f}, "
            f"AUC:{auc:.4f}, "
            f"Avg Accuracy:{average_accuracy:.4f}, "
            f"num_batches:{num_batches}"
        )


        # 早停和模型保存
        save_start = time.time()
        os.makedirs(get_export_dir(), exist_ok=True)

        if auc > best_auc:
            best_auc = auc
            wait = 0
            torch.save(model.state_dict(), os.path.join(get_export_dir(), "model.pth"))
            log.info(f"保存最佳模型，AUC: {best_auc:.4f}")
        else:
            wait += 1
            log.info(f"验证AUC未改善，等待计数: {wait}/{EARLY_STOPPING_PATIENCE}")

        save_time = time.time() - save_start
        log.info(f"模型保存耗时: {save_time:.4f}s")

        if wait >= EARLY_STOPPING_PATIENCE:
            log.info(f"在 {epoch} 个时期后早停。")
            break

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("\nModel Summary:")
    log.info(f"Total parameters: {total_params:,}")
    log.info(f"Trainable parameters: {trainable_params:,}")


def download_train_samples():
    """下载训练样本"""
    oss.download_last_data_type(
        "common",
        oss.DataType.TRAIN_SAMPLES,
        args.common_data_dir,
        days=args.days,
        sample=Sample.NEWUSER_PAIRWISE,
        ratio=SampleRatio.RATIO_1_1,  # 1:1比例
    )


def download_common():
    """下载公共数据"""
    oss.download_last_data_type("common", oss.DataType.USER_BOOK, args.common_data_dir)
    oss.download_last_data_type(
        "common", oss.DataType.READ_DATA, args.common_data_dir, days=args.days
    )
    oss.download_last_data_type("common", oss.DataType.COUNTER, args.common_data_dir)


def upload():
    """上传模型到OSS"""
    version = time.strftime("%Y%m%d%H%M", time.localtime(time.time()))
    zip_filename = f"{model_name}_{version}_{platform.node()}.zip"
    zip_filepath = os.path.join(args.data_dir, zip_filename)
    log.info(f"Add saved model files to {zip_filename}")

    saved_model_dir = get_export_dir()
    with zipfile.ZipFile(zip_filepath, "w", zipfile.ZIP_DEFLATED) as myzip:
        for base, _, files in os.walk(saved_model_dir):
            r_base = base[len(saved_model_dir):]
            for ifile in files:
                log.info(os.path.join(r_base, ifile))
                myzip.write(os.path.join(base, ifile), os.path.join(r_base, ifile))

    log.info(f"Uploading {zip_filename} to oss ...")
    oss.oss_upload(f"hbre-book/data/{model_name}/model/{zip_filename}", zip_filepath)
    os.remove(zip_filepath)
    log.info("Upload saved model to oss done!")


def process_env():
    """处理环境变量"""
    log.info("process env start!")
    if sys.platform == "darwin":
        args.common_data_dir = get_local_path("common")
        args.data_dir = get_local_path(model_name)
        args.download_train_samples = False
        args.download_common = False
        args.upload_to_oss = False
        log.info("当前系统是开发或测试环境")
    log.info("process env done!")


def main():
    """主函数"""
    process_env()

    if args.download_train_samples:
        download_train_samples()
    if args.download_common:
        download_common()

    # 设定训练设备
    if torch.cuda.is_available():
        log.info("使用GPU进行训练")
        device = "cuda"
    else:
        log.info("使用CPU进行训练")
        device = "cpu"

    # 加载词表和书籍表
    with profiler.step("从训练样本统计构建词表和书籍表"):
            os.makedirs(get_export_dir(), exist_ok=True)

            # ✅ 配置：最小频次阈值
            MIN_BOOK_FREQ = 0  # 书籍最少出现10次
            MIN_WORD_FREQ = 0  # 词汇最少出现10次

            # ✅ 读取训练样本（一次性读取所有需要的列）
            sample_file = "train_samples_days_35_sample_pairwise_ratio_1v1v1_rankmixerv1c.parquet"
            sample_path = os.path.join(args.common_data_dir, sample_file)

            log.info(f"读取训练样本: {sample_path}")

            # 定义需要读取的列
            uid_columns = ["uid"]

            register_column = ["register_time"]

            book_columns = [
                "impression_bookid",
                "negative_bookid",
                "top_read_books",
                "read_books",
            ]

            word_columns = [
                "likemarks",
                "searchwords",
                "impression_bookmarks",
                "impression_bookname",
                "impression_tag",
                "impression_bookinfo",
                "negative_bookmarks",
                "negative_bookname",
                "negative_tag",
                "negative_bookinfo",
            ]

            # 复合字段（需要同时统计word和book）
            compound_word_book_columns = [
                "read_search_keywords_books_day_30",
            ]

            # 添加以下代码：初始化UNK统计字典
            unk_statistics = {
                "register_time": {"total": 0, "unk": 0},
                # 书籍类特征
                "impression_bookid": {"total": 0, "unk": 0},
                "negative_bookid": {"total": 0, "unk": 0},
                "top_read_books": {"total": 0, "unk": 0},
                "read_books": {"total": 0, "unk": 0},
                "readpunch": {"total": 0, "unk": 0},
                "read_search_keywords_books_day_30": {"total": 0, "unk": 0},
                # 词汇类特征
                "likemarks": {"total": 0, "unk": 0},
                "searchwords": {"total": 0, "unk": 0},
                "impression_bookmarks": {"total": 0, "unk": 0},
                "impression_bookname": {"total": 0, "unk": 0},
                "impression_tag": {"total": 0, "unk": 0},
                "impression_bookinfo": {"total": 0, "unk": 0},
                "negative_bookmarks": {"total": 0, "unk": 0},
                "negative_bookname": {"total": 0, "unk": 0},
                "negative_tag": {"total": 0, "unk": 0},
                "negative_bookinfo": {"total": 0, "unk": 0},
            }

            sample_df = pl.read_parquet(sample_path, columns=feature_columns)
            log.info(f"✅ 读取训练样本: {len(sample_df):,} 条")

            def count_unk_in_field(value, feature_name, is_compound=False):
                """
                统计单个字段中的UNK数量

                Args:
                    value: 字段值（可能是单值或列表）
                    feature_name: 特征名称
                    is_compound: 是否是复合字段（如 bookid:chapter）
                """
                if value is None:
                    return

                # 处理单个值或列表
                items = value if isinstance(value, list) else [value]

                for item in items:
                    if item is None:
                        continue

                    unk_statistics[feature_name]["total"] += 1

                    item_str = str(item)

                    # 复合字段需要提取第一部分
                    if is_compound and ':' in item_str:
                        item_str = item_str.split(':')[0]

                    # 判断是否为UNK
                    if item_str in ['UNK', 'PAD', 'None', '']:
                        unk_statistics[feature_name]["unk"] += 1

            # ========== 第1步：统计书籍频次 ==========
            with profiler.step_sub("统计书籍频次并构建book词表"):
                from collections import Counter

                book_counter = Counter()

                # 1.1 统计简单书籍列（直接是bookid）
                for col in book_columns:
                    log.info(f"  处理列: {col}")
                    for book_list in sample_df[col].to_list():
                        # 添加UNK统计
                        count_unk_in_field(book_list, col, is_compound=False)

                        if book_list is None:
                            continue

                        # 处理单个值或列表
                        books = book_list if isinstance(book_list, list) else [book_list]

                        for book in books:
                            if book is None:
                                continue

                            # 统一转为字符串
                            book_str = str(book)

                            # 过滤特殊值
                            if book_str and book_str not in ['', 'UNK', 'PAD', 'None']:
                                book_counter[book_str] += 1

                # 1.2 统计复合字段中的书籍（格式：word:bookid:chapter）
                for col in compound_word_book_columns:
                    log.info(f"  处理复合列: {col}")
                    for compound_list in sample_df[col].to_list():
                        if compound_list is None or not isinstance(compound_list, list):
                            continue

                        for item in compound_list:
                            if item is None:
                                continue

                            parts = str(item).split(':')
                            if len(parts) >= 2:
                                # 提取 bookid（第二个字段）
                                book_str = parts[1]

                                if book_str and book_str not in ['', 'UNK', 'PAD', 'None']:
                                    book_counter[book_str] += 1

                total_books = len(book_counter)
                log.info(f"✅ 统计完成: {total_books:,} 本书（过滤前）")

                # ✅ 过滤低频书籍
                book_counter_filtered = {
                    book: cnt
                    for book, cnt in book_counter.items()
                    if cnt >= MIN_BOOK_FREQ
                }

                filtered_count = total_books - len(book_counter_filtered)
                log.info(f"🔥 过滤掉 {filtered_count:,} 本低频书籍（出现<{MIN_BOOK_FREQ}次）")
                log.info(f"📊 保留 {len(book_counter_filtered):,} 本书籍")

                # 构建DataFrame并排序
                book_counter_df = pl.DataFrame({
                    "bookid": list(book_counter_filtered.keys()),
                    "total_cnt": list(book_counter_filtered.values())
                }).sort("total_cnt", descending=True)

                # 在开头插入 UNK
                unk_row = pl.DataFrame({
                    "bookid": ["UNK"],
                    "total_cnt": [-1]
                })
                book_counter_df = pl.concat([unk_row, book_counter_df])

                log.info(f"✅ 最终book词表: {len(book_counter_df):,} 本书（含UNK）")

                # 保存
                output_path = os.path.join(get_export_dir(), "book_counter.parquet")
                book_counter_df.write_parquet(output_path)
                log.info(f"💾 已保存: {output_path}")

            # ========== 第2步：统计词汇频次 ==========
            with profiler.step_sub("统计词汇频次并构建word词表"):
                from collections import Counter

                word_counter = Counter()

                # 2.1 统计简单词汇列
                for col in word_columns:
                    log.info(f"  处理列: {col}")
                    for word_list in sample_df[col].to_list():
                        # 添加UNK统计
                        count_unk_in_field(word_list, col, is_compound=False)

                        if word_list is None:
                            continue

                        # 处理单个值或列表
                        words = word_list if isinstance(word_list, list) else [word_list]

                        for word in words:
                            if word is None:
                                continue

                            word_str = str(word)

                            # 过滤特殊值
                            if word_str and word_str not in ['', 'UNK', 'PAD', 'None']:
                                word_counter[word_str] += 1

                total_words = len(word_counter)
                log.info(f"✅ 统计完成: {total_words:,} 个词（过滤前）")

                # ✅ 过滤低频词汇
                word_counter_filtered = {
                    word: cnt
                    for word, cnt in word_counter.items()
                    if cnt >= MIN_WORD_FREQ
                }

                filtered_count = total_words - len(word_counter_filtered)
                log.info(f"🔥 过滤掉 {filtered_count:,} 个低频词汇（出现<{MIN_WORD_FREQ}次）")
                log.info(f"📊 保留 {len(word_counter_filtered):,} 个词汇")

                # 构建DataFrame并排序
                word_counter_df = pl.DataFrame({
                    "word": list(word_counter_filtered.keys()),
                    "cnt": list(word_counter_filtered.values())
                }).sort("cnt", descending=True)

                # 在开头插入 UNK
                unk_row = pl.DataFrame({
                    "word": ["UNK"],
                    "cnt": [-1]
                })
                word_counter_df = pl.concat([unk_row, word_counter_df])

                log.info(f"✅ 最终word词表: {len(word_counter_df):,} 个词（含UNK）")

                # 保存
                output_path = os.path.join(get_export_dir(), "word_counter.parquet")
                word_counter_df.write_parquet(output_path)
                log.info(f"💾 已保存: {output_path}")

    def print_unk_statistics():
        """打印UNK统计报告"""
        log.info("\n" + "=" * 100)
        log.info("UNK Statistics Report".center(100))
        log.info("=" * 100)
        log.info(f"{'Feature Name':<35} {'Total Elements':<18} {'UNK Count':<18} {'UNK Ratio':<18}")
        log.info("-" * 100)

        # 分组显示
        log.info("\n📚 Book Features:")
        log.info("-" * 100)
        book_features = [
            "impression_bookid", "negative_bookid",
            "top_read_books", "read_books", "readpunch", "read_search_keywords_books_day_30"
        ]
        for feature_name in book_features:
            stats = unk_statistics[feature_name]
            total = stats['total']
            unk_count = stats['unk']
            ratio = (unk_count / total * 100) if total > 0 else 0
            log.info(f"{feature_name:<35} {total:<18,} {unk_count:<18,} {ratio:>16.2f}%")

        log.info("\n📝 Word Features:")
        log.info("-" * 100)
        word_features = [
            "likemarks", "searchwords", "impression_bookmarks", "impression_bookname",
            "impression_tag","impression_bookinfo",
            "negative_bookmarks", "negative_bookname",
            "negative_tag","negative_bookinfo",
        ]
        for feature_name in word_features:
            stats = unk_statistics[feature_name]
            total = stats['total']
            unk_count = stats['unk']
            ratio = (unk_count / total * 100) if total > 0 else 0
            log.info(f"{feature_name:<35} {total:<18,} {unk_count:<18,} {ratio:>16.2f}%")

        # 总体统计
        log.info("\n📊 Overall Statistics:")
        log.info("-" * 100)
        total_all = sum(s['total'] for s in unk_statistics.values())
        unk_all = sum(s['unk'] for s in unk_statistics.values())
        ratio_all = (unk_all / total_all * 100) if total_all > 0 else 0
        log.info(f"{'TOTAL':<35} {total_all:<18,} {unk_all:<18,} {ratio_all:>16.2f}%")
        log.info("=" * 100 + "\n")

    # 添加以下代码：打印UNK统计
    with profiler.step_sub("打印UNK统计信息"):
        print_unk_statistics()


    # ========== 第4步：创建查找表 ==========
    with profiler.step_sub("创建词表查找表"):
            book_table = book_counter_df["bookid"].cast(pl.Utf8).to_list()
            word_table = word_counter_df["word"].to_list()

            book_lookup = VocabLookupTable(book_table, "cpu")
            word_lookup = VocabLookupTable(word_table, "cpu")

            log.info(f"book_lookup: {len(book_table):,} 个书籍")
            log.info(f"word_lookup: {len(word_table):,} 个词汇")


    with profiler.step("构建训练/验证 DataLoader"):
        cache_dir = os.path.join(args.data_dir, "cache_v2")

        if os.path.exists(cache_dir):
            import shutil
            shutil.rmtree(cache_dir)
            log.info(f"✓ 已清空旧缓存: {cache_dir}")

        original_file = os.path.join(
            args.common_data_dir,
            "train_samples_days_35_sample_pairwise_ratio_1v1v1_rankmixerv1c.parquet"
        )

        # 🆕 一次性读取并打乱（仅用于数据预览/调试）
        log.info("读取并打乱训练数据...")
        try:
            df = pl.read_parquet(original_file)
            log.info(f"原始样本数: {len(df):,}")

            df_shuffled = df.sample(fraction=1.0, shuffle=True, seed=42)
            log.info("✓ 数据已打乱")

            shuffled_path = os.path.join(args.common_data_dir, "train_samples_days_35_sample_pairwise_ratio_1v1v1_rankmixerv1c_shuffled.parquet")
            df_shuffled.write_parquet(shuffled_path)

        except Exception as e:
            log.error(f"打乱失败: {e}")
            raise

        # 训练集 - 带缓存
        base_train_ds = BookRecommendationIterableDataset(
            data_file=shuffled_path,
            needed_columns=feature_columns,
            book_lookup=book_lookup,
            word_lookup=word_lookup,
            export_folder=get_export_dir(),
            device="cpu",
            row_batch_size=BATCH_SIZE,
            is_training=True,
        )

        train_ds_with_cache = CacheWriter(
            base_train_ds,
            cache_dir,
        )

        # 验证集 - 不缓存
        val_ds = BookRecommendationIterableDataset(
            data_file=shuffled_path,
            needed_columns=feature_columns,
            book_lookup=book_lookup,
            word_lookup=word_lookup,
            export_folder=get_export_dir(),
            device="cpu",
            row_batch_size=BATCH_SIZE,
            is_training=False,
        )

        train_loader_epoch0 = DataLoader(
            train_ds_with_cache,
            batch_size=None,
            num_workers=0,
            pin_memory=(device == "cuda"),
        )

        evaluate_loader = DataLoader(
            val_ds,
            batch_size=None,
            num_workers=0,
            pin_memory=(device == "cuda"),
        )

    with profiler.step("训练并保存模型"):
        train(
            train_loader_epoch0,
            evaluate_loader,
            len(book_table),
            len(word_table),
            device,
            cache_dir,
            book_table,
            word_table
        )


    if args.upload_to_oss:
        upload()

    profiler.summarize()

def main1():
    """主函数"""
    process_env()

    if args.download_train_samples:
        download_train_samples()
    if args.download_common:
        download_common()

    # 设定训练设备
    if torch.cuda.is_available():
        log.info("使用GPU进行训练")
        device = "cuda"
    else:
        log.info("使用CPU进行训练")
        device = "cpu"

    with profiler.step("直接加载现有 book/word 词表"):
        os.makedirs(get_export_dir(), exist_ok=True)

        book_counter_path = os.path.join(get_export_dir(), "book_counter.parquet")
        word_counter_path = os.path.join(get_export_dir(), "word_counter.parquet")

        if not os.path.exists(book_counter_path):
            raise FileNotFoundError(f"book_counter 不存在: {book_counter_path}")
        if not os.path.exists(word_counter_path):
            raise FileNotFoundError(f"word_counter 不存在: {word_counter_path}")

        book_counter_df = pl.read_parquet(book_counter_path)
        word_counter_df = pl.read_parquet(word_counter_path)

        if "bookid" not in book_counter_df.columns:
            raise ValueError(f"book_counter.parquet 缺少 bookid 列，当前列: {book_counter_df.columns}")
        if "word" not in word_counter_df.columns:
            raise ValueError(f"word_counter.parquet 缺少 word 列，当前列: {word_counter_df.columns}")

        book_table = book_counter_df["bookid"].cast(pl.Utf8).to_list()
        word_table = word_counter_df["word"].cast(pl.Utf8).to_list()

        book_lookup = VocabLookupTable(book_table, "cpu")
        word_lookup = VocabLookupTable(word_table, "cpu")

        log.info(f"✅ 已加载 book_lookup: {len(book_table):,} 个书籍")
        log.info(f"✅ 已加载 word_lookup: {len(word_table):,} 个词汇")

    with profiler.step("构建训练/验证 DataLoader"):
        cache_dir = os.path.join(args.data_dir, "cache_v2")

        if os.path.exists(cache_dir):
            import shutil
            shutil.rmtree(cache_dir)
            log.info(f"✓ 已清空旧缓存: {cache_dir}")

        original_file = os.path.join(
            args.common_data_dir,
            "train_samples_days_35_sample_pairwise_ratio_1v1v1_rankmixerv1c.parquet"
        )
        shuffled_path = os.path.join(
            args.common_data_dir,
            "train_samples_days_35_sample_pairwise_ratio_1v1v1_rankmixerv1c_shuffled.parquet"
        )


        train_ds_with_cache = CacheWriter(
            BookRecommendationIterableDataset(
                data_file=shuffled_path,
                needed_columns=feature_columns,
                book_lookup=book_lookup,
                word_lookup=word_lookup,
                export_folder=get_export_dir(),
                device="cpu",
                row_batch_size=BATCH_SIZE,
                is_training=True,
            ),
            cache_dir,
        )

        val_ds = BookRecommendationIterableDataset(
            data_file=shuffled_path,
            needed_columns=feature_columns,
            book_lookup=book_lookup,
            word_lookup=word_lookup,
            export_folder=get_export_dir(),
            device="cpu",
            row_batch_size=BATCH_SIZE,
            is_training=False,
        )

        train_loader_epoch0 = DataLoader(
            train_ds_with_cache,
            batch_size=None,
            num_workers=0,
            pin_memory=(device == "cuda"),
        )

        evaluate_loader = DataLoader(
            val_ds,
            batch_size=None,
            num_workers=0,
            pin_memory=(device == "cuda"),
        )

    with profiler.step("训练并保存模型"):
        train(
            train_loader_epoch0,
            evaluate_loader,
            len(book_table),
            len(word_table),
            device,
            cache_dir,
            book_table,
            word_table
        )

    if args.upload_to_oss:
        upload()

    profiler.summarize()

if __name__ == "__main__":
    if sys.platform == "darwin":
        main1()
    else:
        main()