#%%
from dask.distributed import Client, LocalCluster, wait
import dask.dataframe as dd
from dask.distributed import LocalCluster
import dask

import lightgbm as lgb
from lightgbm import log_evaluation, early_stopping

import time 
import numpy as np
import pandas as pd

from sklearn.metrics import r2_score


print(dask.__version__)


# #%%
# if __name__ == "__main__":
#     cluster = LocalCluster()
#     client = Client(cluster)
#     client

# %%
cluster = LocalCluster(processes=False)
client = Client(cluster)

#%%

client
cluster

# %%

cluster.close()
client.close()

# %% ------------------ READ DATA -----------------------------------

ddf = dd.read_parquet('data/raw/train.parquet')
display(ddf.head())
print(len(ddf))
print(type(ddf))

# %%
# Count the unique 'date_id' values
unique_dates = ddf['date_id'].nunique().compute()
print(f"Number of unique days (date_id): {unique_dates}")

# Filter data
train_all = ddf.loc[ddf['date_id'] > 300]
#print(type(train_all))

# Choose the following features
# # tag
# num_list = [0,1,2,3,4,32,33,34,35,36]
# feature_list = [f'feature_{i:02}' for i in num_list]
#  # All feature, half features
#feature_list = [f'feature_{i:02}' for i in range(40)]
# Best features
feature_list = ['feature_48', 'feature_06', 'feature_59']
use_features = ['responder_6', 'symbol_id'] + feature_list
train_all = train_all[use_features].compute()
#print(type(train_all))

# # Split X,y for final
y_train_all = train_all[['responder_6']]
train_all = train_all.drop(columns={'responder_6'})

print(train_all.shape)
#print(train_all.date_id.min(), train_all.date_id.max())

#%%
# Split in train, test
test_all = train_all.loc[train_all.date_id > 1670]
train_all = train_all.loc[train_all.date_id < 1670]
print('train: ', train_all.shape)
print('test: ', test_all.shape)


# Split X,y
y_train_all = train_all[['responder_6']]
y_test_all = test_all[['responder_6']]

train_all = train_all.drop(columns={'responder_6'})
test_all = test_all.drop(columns=['responder_6'])

#%%

print('train', type(train_all), train_all.shape)
print('ytrain', type(y_train_all), y_train_all.shape)
print('test', type(test_all), test_all.shape)
print('ytest', type(y_test_all), y_test_all.shape)

# %%


#n_workers = 3
cluster = LocalCluster(processes=False, n_workers= 3,) #n_workers=n_workers)
client = Client(cluster)
#client.wait_for_workers(n_workers)

client

# %%

print(f"View the dashboard: {cluster.dashboard_link}")

# %%

#dask_reg = lgb.DaskLGBMRegressor(
dask_reg = lgb.LGBMRegressor(
    max_depth=5,
    learning_rate=0.1,
    tree_learner="data_parallel",
    n_estimators=300,
    min_child_samples=1,
    verbose=1,
)

#%%

start_time = time.time()
dask_reg.fit(train_all, y_train_all)

end_time = time.time()
print(f'total time: {end_time - start_time}' )
# %%

y_pred = dask_reg.predict(test_all)
r2 = r2_score(y_test_all, y_pred)

print(f'r2: {r2}')

# %% ------------------------------ STD LGBM MODEL -----------------------------
SEED = 42
start_time = time.time()

trn_set = lgb.Dataset(train_all, y_train_all)
#tst_set = [lgb.Dataset(test_all, y_test_all)]

params = {
#    "boosting_type": "gbdt",
#    "objective": "regression",
#    "metric": 'rmse',
    "verbose": -1,
    'random_state': 42,
    'n_jobs':4,
    "feature_pre_filter" : False
}


# Custom scoring method
def ScoreMetric(ypred,eval_data): #, weight):
    """
    This function is a modification of the ready-made R-square function with sample weight. 
    We have this as a column in the dataset
    """
    ytrue = eval_data.get_label()
    weight = np.array([1.749479]*len(ytrue))
    score = r2_score(ytrue, ypred, sample_weight = weight)
    return 'weighted_r2', score, True # Return: Name, score, higher is better



params = ({#"device"           : "gpu",
        "boosting_type"     : 'gbdt',
        "objective"        : "regression_l2",
        "metrics"          : "None",
        #"n_estimators"     : 100,
        "max_depth"        : 10,
        "learning_rate"    : 0.03,
        "colsample_bytree" : 0.55,
        "subsample"        : 0.80,
        "random_state"     : SEED,
        "reg_lambda"       : 1.25,
        "reg_alpha"        : 0.001,
        "verbose"          : -1,
        "n_jobs"           : 4,
         }
      )

rgr = lgb.train(params, trn_set,
    valid_sets           = trn_set,
    feval               = ScoreMetric,
    callbacks          = [log_evaluation(100), early_stopping(100, verbose = False)],
)

end_time = time.time()
print(f'Training took {end_time - start_time}s')
#%%

y_pred = rgr.predict(test_all)
r2 = r2_score(y_test_all, y_pred)

print(f'r2: {r2}')

# %%
import joblib 

joblib.dump(rgr, f'models/lgb_large3.pkl')
print('SAVED')



# %%
