import numpy as np

def average_precision_at_k(y_true, y_score, k=12):
    """
    y_true: 真实标签 (0/1)
    y_score: 模型打分
    """
    order = np.argsort(y_score)[::-1][:k]
    y_true_k = np.array(y_true)[order]

    hits = 0
    ap = 0.0
    for i, rel in enumerate(y_true_k, start=1):
        if rel == 1:
            hits += 1
            ap += hits / i

    total_pos = np.sum(y_true)
    if total_pos == 0:
        return 0.0

    return ap / min(total_pos, k)


def MAP_at_k(list_of_y_true, list_of_y_score, k=12):
    """
    多个 query 的 MAP@k
    """
    aps = [
        average_precision_at_k(y_t, y_s, k)
        for y_t, y_s in zip(list_of_y_true, list_of_y_score)
    ]
    return np.mean(aps)
