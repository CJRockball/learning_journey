#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random, os, joblib

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

#%%

params = ({#"device"           : "gpu",
        "boosting_type"     : 'gbdt',
        "objective"        : "regression_l2",
        "metrics"          : "None",
        "n_estimators"     : 100,
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

#%% Read in data
import time

file_list = [4, 5, 6, 7, 8, 9]
model_dict = {}
start_time = time.time()
for k in range(39):
    print(f'symbol {k}')
    for i in file_list:
        df = pd.read_parquet(f'data/raw/train.parquet/partition_id={i}')
        df = df.loc[df.symbol_id == k]
        if i > 4:
            df_s = pd.concat([df_s, df], axis=0)
        else:
            df_s = df
            
    print(len(df_s))
    if len(df_s) != 0:
        # Filter features and symbol
        other_features = [] #['symbol_id'] # 'date_id', 'time_id',] 
        feature_features = [f'feature_{i:02}' for i in range(79)]
    #    symbol_features = [f'symbol__{i}' for i in range(39)]
        features = other_features + feature_features #+ symbol_features

        ytrain = df.pop('responder_6')
        Xtrain = df[features]


        trn_set = lgb.Dataset(Xtrain, ytrain)

        #Train model
        rgr = lgb.train(params, trn_set,
            valid_sets           = trn_set,
            feval               = ScoreMetric,
            callbacks          = [log_evaluation(100)] #, early_stopping(100, verbose = False)],
        )

        model_dict[k] = rgr
 
end_time = time.time()
print(f'Total time: {end_time - start_time}')   
#%% Save single model file
# save model
joblib.dump(rgr, f'models/lgb_multi_{i}.pkl')
print('SAVED')

#%% Save dict with all models to pickle
import pickle

# save dictionary to person_data.pkl file
with open('models/lgbm_ind_symb2.pkl', 'wb') as fp:
    pickle.dump(model_dict, fp)
    print('dictionary saved successfully to file')


# %%

test = Xtrain.iloc[0,:].to_frame().T

print(test.shape)
print(type(test))

#%%

df = pd.read_parquet(f'data/raw/train.parquet/partition_id=9')
display(df)

#%%

test_row = df.iloc[5,:].to_frame().T.reset_index(drop=True)
print(type(test_row))
display(test_row)

get_symbol = int(test_row.loc[0,"symbol_id"])
print(get_symbol)

feature_features = [f'feature_{i:02}' for i in range(79)]
one_liner = test_row[features]
display(one_liner)

# %%
# Read dictionary pkl file
with open('models/lgbm_ind_symb.pkl', 'rb') as fp:
    predict_dict = pickle.load(fp)
    print('Person dictionary')

y_pred = predict_dict[get_symbol].predict(one_liner)

print(y_pred)
print(type(y_pred[0]))
print(y_pred.shape)

# %%

out = np.array(y_pred)
print(type(out))
print(out.shape)

# %%

# %%


# %%
