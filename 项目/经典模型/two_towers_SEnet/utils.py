import numpy as np

def ages_split_buckets(arr,bins = [18, 25, 35, 45, 50, 56]):
    arr = arr.to_numpy() if hasattr(arr, "to_numpy") else np.array(arr)

    bucket_ids = np.digitize(arr, bins, right=False)  # 从1开始

    return bucket_ids