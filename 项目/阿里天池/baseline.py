#%%
import gc
import logging
import math
import random
import time
import warnings
from datetime import datetime
from operator import itemgetter
from pathlib import Path

logger=logging.getLogger(__name__)
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd


#%%
def reduce_mem(df):
    starttime=time.time()
    numerics=['int16','int32','int64','float16','float32','float64']
    start_mem=df.memory_usage().sum()/1024**2
    for col in df.columns:
        if col in numerics:
            c_min=df[col].min()
            c_max=df[col].max()
            if pd.isnull(c_min) or pd.isnull(c_max):
                continue
            if str(df[col].types)[:3]=='int':
                if c_min>np.iinfo(np.int8).min and c_max<np.iinfo(np.int8).max:
                    df[col]=df[col].astype(np.int8)
                elif c_min>np.iinfo(np.int16).min and c_max<np.iinfo(np.int16).max:
                    df[col]=df[col].astype(np.int16)
                elif c_min>np.iinfo[np.int32].min and c_max<np.iinfo(np.int32).max:
                    df[col]=df[col].astype(np.int32)
            else:
                if c_min>np.iinfo(np.float16).min and c_max<np.iinfo(np.float16).max:
                    df[col]=df[col].astype(np.float16)
                elif c_min>np.iinfo(np.float32).min and c_max<np.iinfo(np.float32).max:
                    df[col]=df[col].astype(np.float32)
    end_mem=df.memory_usage().sum()/1024**2
    print(f'--Memory usage after optimization: {end_mem:.2f} MB, about {(start_mem-end_mem)/starttime*100}%, time spend {(time.time()-starttime)/60} min')
#%%
def get_all_click_sample(sample_nums=10000):
    """
    训练集中采样一部分数据调试
    :param sample_nums:
    :return:
    """
    all_click=pd.read_csv('./data/train_click_log.csv')
    all_user_ids=all_click.user_id.unique()

    sample_user_ids=np.random.choice(all_user_ids,size=sample_nums,replace=False)
    all_click=all_click[all_click['user_id'].isin(sample_user_ids)]

    # 按照多列去重，('user_id','click_article_id','click_timestamp')都一样才视为相同
    all_click=all_click.drop_duplicates(['user_id','click_article_id','click_timestamp'])

def get_all_click_df(offline):
    if offline:
        all_click=pd.read_csv('./data/train_click_log.csv')[:20000]
    else:
        all_click=pd.concat([pd.read_csv('./data/train_click_log.csv')[:10000],pd.read_csv('./data/testA_click_log.csv')[:10000]])
    return all_click
#%%
all_click=get_all_click_df(offline=False)
#%%
print(all_click.head())