import os

package_path='/Users/Zhuanz/Downloads/Recommand-System/项目/经典模型/Din'

cfg={
    # 文件路径
    "movies": os.path.join(package_path,"ml_20m/movies.csv"),
    "ratings": os.path.join(package_path,"ml_20m/ratings.csv"),
    "train":os.path.join(package_path,"ml_20m/train.parquet"),
    "test":os.path.join(package_path,"ml_20m/test.parquet"),
    # 小版本文件，方便测试跑通
    "mini_train":os.path.join(package_path,"ml_20m_mini/train.parquet"),
    "mini_test":os.path.join(package_path,"ml_20m_mini/test.parquet"),

    # 映射表 uid->idx, movie_id->idx
    "user_mapping_path":os.path.join(package_path,"ml_20m/uid2idx.pkl"),
    "movie_mapping_path":os.path.join(package_path,"ml_20m/mid2idx.pkl"),

    # 模型参数
    "max_hist_len":15,
    "max_cate_len":5,
    "embed_dim":64,
    "item_count":30000,
    "cate_count":200,
    "user_count":150000,
}