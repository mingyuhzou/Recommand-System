import cudf
train = cudf.read_parquet('../data/processData/train.parquet')
train=cudf.read_parquet('../data/processData/train.parquet')
test=cudf.read_parquet('../data/processData/test.parquet')
train_pairs = cudf.concat([train, test])[['session', 'aid']]
del train, test

train_pairs['aid_next'] = train_pairs.groupby('session').aid.shift(-1)
train_pairs = train_pairs[['aid', 'aid_next']].dropna().reset_index(drop=True)