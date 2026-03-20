import os
import pickle
import numpy as np
import pandas as pd
from gensim.models import Word2Vec

from config import data_cfg,model_cfg


def save_pickle(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def build_title_word_mapping_and_w2v(vector_size=model_cfg['embed_dim'], window=5, min_count=1, workers=4, epochs=20):
    movies_df = pd.read_parquet(data_cfg["movies"])

    if "title" not in movies_df.columns:
        raise ValueError("movies.parquet 中没有 title 字段")

    # 1. 准备语料
    sentences = []
    for text in movies_df["title"]:
        if not isinstance(text, str):
            continue
        words = [w for w in text.split() if w]
        if not words:
            continue
        sentences.append(words)


    # 训练 Word2Vec
    w2v_model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        epochs=epochs,
        sg=1,  # 1=skip-gram, 0=cbow
    )
    vocab = list(w2v_model.wv.index_to_key)

    word2idx = {
        "<PAD>": 0,
        "<UNK>": 1,
    }

    for i, w in enumerate(vocab, start=2):
        word2idx[w] = i

    save_pickle(word2idx, data_cfg["word2idx"])

    word2emb = {
        word: w2v_model.wv[word]
        for word in w2v_model.wv.index_to_key
    }
    word2emb["<PAD>"] = np.zeros(vector_size, dtype=np.float32)
    word2emb["<UNK>"] = np.random.normal(0, 0.1, vector_size).astype(np.float32)

    embedding_matrix = np.zeros((len(word2idx), vector_size), dtype=np.float32)

    embedding_matrix[word2idx["<UNK>"]] = word2emb["<UNK>"]

    for word, idx in word2idx.items():
        if word in ("<PAD>", "<UNK>"):
            continue
        embedding_matrix[idx] = word2emb[word]

    np.save(data_cfg["word2emb"], embedding_matrix)

build_title_word_mapping_and_w2v()