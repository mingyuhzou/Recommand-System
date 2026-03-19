# 简介

基于 **MovieLens-20M 数据集** 探索了 DIN（Deep Interest Network）点击率预测模型（CTR）



## 数据与特征

使用 MovieLens-20M 数据集，构建如下核心特征：

```text
userId                          用户ID
movie_id                        当前候选电影
movie_cate_id_list              当前电影类别（多值特征）
rated_movie_id_list             用户历史点击电影序列
rated_movie_cate_id_list        历史电影类别序列
label                           是否点击（rating >= 4 → 1，否则 0）
```

其中：

* `movie_cate_id_list`：多标签离散特征（如 genres）
* `rated_movie_id_list`：行为序列（DIN 核心输入）
* `rated_movie_cate_id_list`：行为序列的属性特征



## 项目结构

```text
.
├── config.py                 # 全局配置（路径、embedding维度等）
├── train.py                  # 训练入口（DIN + AUC评估）
│
├── Data/
│   └── movielens.py          # Dataset（padding + ID映射 + 序列处理）
│
├── model/
│   └── Din.py                # DIN模型实现（Attention + Dice + MLP）
│
├── process/
│   ├── build.py              # 构建完整训练/测试数据集（DuckDB）
│   ├── build_small_sample.py # 构建小规模数据（快速调试）
│   ├── build_id2idx.py       # 构建 user/movie ID 映射
│   └── review.ipynb          # 数据分析与探索
│
├── ml_20m/                  # 全量数据
│   ├── ratings.csv
│   ├── movies.csv
│   ├── train.parquet
│   ├── test.parquet
│   ├── uid2idx.pkl
│   ├── mid2idx.pkl
│   └── genre_vocab.parquet
│
├── ml_20m_mini/             # 小规模数据（调试用）
│   ├── train.parquet
│   ├── test.parquet
│   └── mini_genre_vocab.parquet
│
└── readme
```



## ID 映射（Embedding 输入）

为了适配 embedding，需要将原始 ID 映射为连续索引：

```text
userId        → user_idx
movie_id      → movie_idx
```

注意：

* `movie_id` 和 `rated_movie_id_list` 必须使用同一套映射
* padding 使用 `0`，因此索引从 `1` 开始

---

## 模型说明（DIN）

核心结构：

```text
Embedding
   ↓
Target Item Embedding
   ↓
User Behavior Sequence
   ↓
DIN Attention（Local Activation Unit）
   ↓
User Interest Representation
   ↓
MLP（Dice 激活）
   ↓
Logit
```

特点：

* 使用 **Attention 建模用户兴趣动态变化**
* 使用 **Dice 激活函数** 提升表达能力
* 支持多值特征（类别 pooling）

---

## 训练与评估

任务：CTR（二分类）

* Loss：`BCEWithLogitsLoss`
* 评估指标：**AUC**


---

## 快速开始

### 1. 构建数据集

```bash
python process/build.py
```

或小规模测试：

```bash
python process/build_small_sample.py
```

---

### 2. 构建 ID 映射

```bash
python process/build_id2idx.py
```

---

### 3. 训练模型

```bash
python train.py
```

---

