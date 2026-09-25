#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random, os

from sklearn.metrics import r2_score

import lightgbm as lgb
from lightgbm import log_evaluation, early_stopping, LGBMRegressor as LGBMR

import gc
# torch.cuda.empty_cache()
gc.collect()


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True

seed = 42
seed_everything(seed)

#%% Read in data
import time

start_time = time.time()

for i in range(10):
    df = pd.read_parquet(f'data/raw/train.parquet/partition_id={i}')
    if i == 0:
        data = df.loc[df.symbol_id == 0]
    else:
        data = pd.concat([data, df.loc[df.symbol_id == 0]])


data = data.drop(columns=['symbol_id']).reset_index(drop=True)

display(data.head(10))
print(data.shape)
end_time = time.time()
print(f'Total time: {end_time - start_time}')

#%% get symbol dummies

dd = data.groupby(by='date_id').mean().reset_index()

display(dd)
print(dd.shape)
print(data.date_id.min(), data.date_id.max())

assert (data.loc[data.date_id == 1535]['responder_6'].mean() ==
         dd.loc[dd.date_id == 1535]['responder_6'].values[0])

#%% Check number of missing values

date_check = dd.loc[:,'date_id'].to_list()
# print(date_check)
# print(len(date_check))

missing_list = []
p = 1
for i,v in enumerate(date_check):
    if i+p != v:
        missing_list.append(i)
        p += 1

print(len(missing_list))
print(len(dd) - dd.date_id.max())
print(missing_list)


#%% And missing rows

display(dd.iloc[20:30,:])
display(dd.iloc[-20:-10,:])

new_index = [i for i in range(dd.date_id.max())]
dd.index = dd.date_id.values
dd = dd.reindex(new_index)

display(dd.iloc[20:30,:])
display(dd.iloc[-20:-10,:])

#%% Process daily series as a normal time series

dd.responder_6.plot()
plt.show()

#%% Fill NA



#%%

TRAIN_SIZE = 1500
dd_train = dd.iloc[:TRAIN_SIZE,:]
print(dd_train.shape)
dd_test = dd.iloc[TRAIN_SIZE:, :]
print(dd_test.shape)

# %%
