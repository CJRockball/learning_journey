#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
#from IPython.display import display
from xgboost import XGBRegressor, XGBClassifier
import xgboost
print("Using XGBoost version",xgboost.__version__)
from catboost import CatBoostRegressor, Pool
from scipy.stats import rankdata 

import pandas.api.types
from lifelines import KaplanMeierFitter
from lifelines.utils import concordance_index

class ParticipantVisibleError(Exception):
    pass


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float: 
    del solution[row_id_column_name]
    del submission[row_id_column_name]
    
    event_label = 'efs'
    interval_label = 'efs_time'
    prediction_label = 'prediction'
    for col in submission.columns:
        if not pandas.api.types.is_numeric_dtype(submission[col]):
            raise ParticipantVisibleError(f'Submission column {col} must be a number')
    # Merging solution and submission dfs on ID
    merged_df = pd.concat([solution, submission], axis=1)
    merged_df.reset_index(inplace=True)
    merged_df_race_dict = dict(merged_df.groupby(['race_group']).groups)
    metric_list = []
    for race in merged_df_race_dict.keys():
        # Retrieving values from y_test based on index
        indices = sorted(merged_df_race_dict[race])
        merged_df_race = merged_df.iloc[indices]
        # Calculate the concordance index
        c_index_race = concordance_index(
                        merged_df_race[interval_label],
                        -merged_df_race[prediction_label],
                        merged_df_race[event_label])
        metric_list.append(c_index_race)
    return float(np.mean(metric_list)-np.sqrt(np.var(metric_list)))

#GET SCORE
def c_index(df_train, data_array):
    y_true = df_train[["ID","efs","efs_time","race_group"]].copy()
    y_pred = df_train[["ID"]].copy()
    y_pred["prediction"] = data_array
    m = score(y_true.copy(), y_pred.copy(), "ID")
    return m

#%%
print("Get Data")
df_train = pd.read_csv('data/ra/train.csv')
df_train = df_train.astype({'efs':bool})



with open('oof/lgb_km.npy', 'rb') as f:
    lgb_km = np.load(f)
with open('oof/lgb_na.npy', 'rb') as f:
    lgb_na = np.load(f)
with open('oof/lgb_cox.npy', 'rb') as f:
    lgb_cox = np.load(f)

with open('oof/xgb_km.npy', 'rb') as f:
    xgb_km = np.load(f)
with open('oof/xgb_na.npy', 'rb') as f:
    xgb_na = np.load(f)
with open('oof/xgb_cox.npy', 'rb') as f:
    xgb_cox = np.load(f)
    
with open('oof/cat_km.npy', 'rb') as f:
    cat_km = np.load(f)
with open('oof/cat_na.npy', 'rb') as f:
    cat_na = np.load(f)
with open('oof/cat_cox.npy', 'rb') as f:
    cat_cox = np.load(f)


#%%

oofs = [
    lgb_km, lgb_na, lgb_cox,
    xgb_km, xgb_na, xgb_cox,
    cat_km, cat_na, cat_cox,
       ]

weights = [1,1,1,
           0,0,0,
           1,1,1]


# Create ranked oofs
ranked_oofs = np.array([rankdata(p) for p in oofs])
ensemble_oofs = np.sum([w * p for w, p in zip(weights, ranked_oofs)], axis=0)

m = c_index(df_train, ensemble_oofs)
print('ensemble c_indexx: ',m)

# %%

oof_name = [
    'lgb_km', 'lgb_na', 'lgb_cox',
    'xgb_km', 'xgb_na', 'xgb_cox',
    'cat_km', 'cat_na', 'cat_cox',
       ]

c_score = []
for n in range(len(oofs)):
    ranked_oof = rankdata(oofs[n])
    m = c_index(df_train, ranked_oof)
    c_score.append(m)    
    print(f'{oof_name[n]} c_index: {round(m, 5)}')
    
df_score = pd.DataFrame(data=c_score, index = oof_name, columns=['c_score'])
    
# %%

display(df_score.sort_values(by=['c_score'], ascending=False))




# %%
