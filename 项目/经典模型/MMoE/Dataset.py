import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader,Dataset
import numpy as np

def load_and_process(seed=3):
    """
    返回
    train_data        # (N,input_dim)
    train_label       # [income_label(N,), marital_label(N,)]
    validation_data
    validation_label
    test_data
    test_label
    output_info 任务的个数，(每个人物的维度、名称)
    """
    column_names = [
        'age', 'class_worker', 'det_ind_code', 'det_occ_code', 'education',
        'wage_per_hour', 'hs_college', 'marital_stat', 'major_ind_code',
        'major_occ_code', 'race', 'hisp_origin', 'sex', 'union_member',
        'unemp_reason', 'full_or_part_emp', 'capital_gains', 'capital_losses',
        'stock_dividends', 'tax_filer_stat', 'region_prev_res', 'state_prev_res',
        'det_hh_fam_stat', 'det_hh_summ', 'instance_weight', 'mig_chg_msa',
        'mig_chg_reg', 'mig_move_reg', 'mig_same', 'mig_prev_sunbelt',
        'num_emp', 'fam_under_18', 'country_father', 'country_mother',
        'country_self', 'citizenship', 'own_or_self', 'vet_question',
        'vet_benefits', 'weeks_worked', 'year', 'income_50k'
    ]

    label_columns = ['income_50k', 'marital_stat']

    categorical_columns = ['class_worker', 'det_ind_code', 'det_occ_code', 'education', 'hs_college',  'major_ind_code','major_occ_code', 'race', 'hisp_origin', 'sex', 'union_member', 'unemp_reason','full_or_part_emp', 'tax_filer_stat', 'region_prev_res', 'state_prev_res', 'det_hh_fam_stat','det_hh_summ', 'mig_chg_msa', 'mig_chg_reg', 'mig_move_reg', 'mig_same', 'mig_prev_sunbelt','fam_under_18', 'country_father', 'country_mother', 'country_self', 'citizenship','vet_question']

    # 读取数据集
    train_df=pd.read_csv('data/census-income.data.gz',header=None,names=column_names)
    test_df=pd.read_csv('data/census-income.test.gz',header=None,names=column_names)

    # 标签，标签的列名前有空格
    y_train_income=(train_df['income_50k']==' 50000+.').astype(np.int64)
    y_train_marital=(train_df['marital_stat']==' Never married').astype(np.int64)

    y_test_income=(test_df['income_50k']==' 50000+.').astype(np.int64)
    y_test_marital=(test_df['marital_stat']==' Never married').astype(np.int64)

    # 特征
    X_train=train_df.drop(label_columns,axis=1)
    X_test=test_df.drop(label_columns,axis=1)

    # 处理类别特征
    X_all=pd.concat([X_train,X_test],axis=0)
    X_all=pd.get_dummies(X_all,columns=categorical_columns)
    X_all = X_all.astype(np.float32)

    X_train=X_all.iloc[:len(X_train)]
    X_test=X_all.iloc[len(X_train):]

    # 划分验证集和测试集
    X_val, X_test, y_val_income, y_test_income, y_val_marital, y_test_marital = train_test_split(
    X_test, y_test_income, y_test_marital, test_size=0.5, random_state=seed
)

    return (
        torch.tensor(X_train.values, dtype=torch.float32),
        [torch.tensor(y_train_income.values, dtype=torch.float32),torch.tensor(y_train_marital.values, dtype=torch.float32)],
        torch.tensor(X_val.values, dtype=torch.float32),
        [torch.tensor(y_val_income.values, dtype=torch.float32),torch.tensor(y_val_marital.values, dtype=torch.float32)],
        torch.tensor(X_test.values, dtype=torch.float32),
        [torch.tensor(y_test_income.values, dtype=torch.float32),torch.tensor(y_test_marital.values, dtype=torch.float32)],
        [(2, 'income'), (2, 'marital')]
    )
class censusData(Dataset):
    def __init__(self,train,label):
        super(censusData,self).__init__()
        self.train=train
        self.label=label
    def __getitem__(self,idx):
        x=self.train[idx]
        y=torch.stack([self.label[0][idx],self.label[1][idx]]) # (2,)
        return x,y

    def __len__(self):
        return self.train.shape[0]