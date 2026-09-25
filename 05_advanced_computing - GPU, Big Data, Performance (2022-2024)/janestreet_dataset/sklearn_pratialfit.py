#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random, os, time

from sklearn.metrics import r2_score

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

from IPython.display import display
import lightgbm as lgb
from lightgbm import log_evaluation, early_stopping

#%%
# Load symbol 0
#df = pd.read_parquet('data/artifacts/train_symbol_0.parquet').reset_index(drop=True)
# Load all
df = pd.read_parquet(f'data/raw/train.parquet/partition_id=9')
   
display(df.head(10))
print(df.shape)
print(df.isnull().sum().sum())
print(df.date_id.min(), df.date_id.max())

# %%

def make_dataset(df, N_LAGS=1, date_split=1668, 
                 responder_lags=False,
                 drop_features=True):
    df_temp = df.copy(deep=True)
    def get_lags(df_lags, N_LAGS, responder_lags):
        """ Make N cols with lags
        lag all the responder one step
        """
        
        for i in range(N_LAGS):
            df_lags[f'r6_lag{i+1}'] = df_lags.loc[:,'responder_6'].shift(i+1)

        if responder_lags:
            responder_features = [(f'responder_{i}',i) for i in [3] ] # range(9) if i not in [6]]
            for name, i in responder_features:
                df_lags[f'responder_{i}_lag'] = df_lags.loc[:,name].shift(1)


        #df_lags.dropna(inplace=True)    
        return df_lags


    df_responder_lags = get_lags(df_temp, N_LAGS, responder_lags)

    if drop_features:
        responder_features = [f'responder_{i}' for i in range(9) if i not in [6]]
        drop_features = ['weight', 'time_id', 'r6_lag1', 'date_id' #'symbol_id',    
                        ] + responder_features
        df_responder_lags = df_responder_lags.drop(columns=drop_features)
        df_responder_lags = df_responder_lags.fillna(0)
    else:
        num_list = [0,1,2,3,4,32,33,34,35,36]
        feature_list = [f'feature_{i:02}' for i in num_list]
        r6_lags = [f'r6_lag{i}' for i in range(1,N_LAGS+1)]
        use_features = ['responder_6', 'date_id',] + r6_lags + feature_list  #'time_id', #
#                        'feature_48', 'feature_06', 'feature_59',]  #  #'responder_3_lag'
        df_responder_lags = df_responder_lags[use_features]
        df_responder_lags = df_responder_lags.fillna(0)
        
    Xtrain = df_responder_lags.loc[df.date_id < date_split]
    Xtest = df_responder_lags.loc[df.date_id >= date_split]
    ytrain = Xtrain.pop('responder_6')
    ytest = Xtest.pop('responder_6')

    print(Xtrain.columns)
    return Xtrain, Xtest, ytrain, ytest
 
df_t = df.copy(deep=True)   
Xtrain, Xtest, ytrain, ytest = make_dataset(df_t,N_LAGS=1, drop_features=False)


# %% Train sklearn model on one file
from sklearn.linear_model import SGDRegressor

start_time = time.time()
reg = SGDRegressor()
reg.fit(Xtrain,ytrain)

y_pred = reg.predict(Xtest)
r2 = r2_score(ytest, y_pred)
end_time = time.time()

print(f'Total time: {end_time - start_time}')
print(f'R2: {r2}')

# TOO SLOW

# %% -------------------------------------- XGB ------------------------------- 
# Train XGB on one file
from xgboost import XGBRegressor
from sklearn.base import clone

start_time = time.time()
model = XGBRegressor(n_estimators=5000, early_stopping_rounds=50, eval_metric=['rmse'], #max_bin = 26214,
                   n_jobs=-1, random_state=0, colsample_bytree=0.9, max_delta_step = 0.9, gamma = 0.001, 
                      max_depth = 9, eta=0.99) #, device="cuda")

xgb_model = clone(model).fit(Xtrain, ytrain, eval_set=[(Xtrain, ytrain), (Xtest, ytest)],verbose=100)


y_pred = xgb_model.predict(Xtest)
r2 = r2_score(ytest, y_pred)
end_time = time.time()

print(f'Total time: {end_time - start_time}')
print(f'R2: {r2}')

#%%
from dask.distributed import Client, LocalCluster
client = Client()
client

#%%
import dask
import xgboost
import dask_xgboost

params = {'objective': 'binary:logistic',
          'max_depth': 4, 'eta': 0.01, 'subsample': 0.5,
          'min_child_weight': 0.5}

start_time = time.time()
bst = dask_xgboost.train(client, params, Xtrain, ytrain, num_boost_round=10)

y_pred = bst.predict(Xtest)
r2 = r2_score(ytest, y_pred)
end_time = time.time()

print(f'Total time: {end_time - start_time}')
print(f'R2: {r2}')


# %% -------------------------- LGBM -----------------------------------

start_time = time.time()

trn_set = lgb.Dataset(Xtrain, ytrain)
num_round = 10000
tst_set = [lgb.Dataset(Xtest, ytest)]

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
        #"n_estimators"     : 5000,
        "max_depth"        : 10,
        "learning_rate"    : 0.03,
        "colsample_bytree" : 0.55,
        "subsample"        : 0.80,
        "random_state"     : seed,
        "reg_lambda"       : 1.25,
        "reg_alpha"        : 0.001,
        "verbose"          : -1,
        "n_jobs"           : 4,
         }
      )

rgr = lgb.train(params, trn_set,
    valid_sets           = tst_set,
    feval               = ScoreMetric,
    callbacks          = [log_evaluation(100), early_stopping(100, verbose = False)],
)

y_pred = rgr.predict(Xtest)

r2 = r2_score(ytest, y_pred)
end_time = time.time()
print(f'Total time: {end_time - start_time}')
print(f'R2: {r2}')




# %%
