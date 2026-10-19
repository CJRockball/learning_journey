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

"""
To evaluate the equitable prediction of transplant survival outcomes,
we use the concordance index (C-index) between a series of event
times and a predicted score across each race group.
 
It represents the global assessment of the model discrimination power:
this is the model’s ability to correctly provide a reliable ranking
of the survival times based on the individual risk scores.
 
The concordance index is a value between 0 and 1 where:
 
0.5 is the expected result from random predictions,
1.0 is perfect concordance (with no censoring, otherwise <1.0),
0.0 is perfect anti-concordance (with no censoring, otherwise >0.0)

"""

import pandas.api.types
from lifelines import KaplanMeierFitter
from lifelines.utils import concordance_index

class ParticipantVisibleError(Exception):
    pass


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    """
    >>> import pandas as pd
    >>> row_id_column_name = "id"
    >>> y_pred = {'prediction': {0: 1.0, 1: 0.0, 2: 1.0}}
    >>> y_pred = pd.DataFrame(y_pred)
    >>> y_pred.insert(0, row_id_column_name, range(len(y_pred)))
    >>> y_true = { 'efs': {0: 1.0, 1: 0.0, 2: 0.0}, 'efs_time': {0: 25.1234,1: 250.1234,2: 2500.1234}, 'race_group': {0: 'race_group_1', 1: 'race_group_1', 2: 'race_group_1'}}
    >>> y_true = pd.DataFrame(y_true)
    >>> y_true.insert(0, row_id_column_name, range(len(y_true)))
    >>> score(y_true.copy(), y_pred.copy(), row_id_column_name)
    0.75
    """
    
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

#%% Get data

df_train = pd.read_csv('data/ra/train.csv')
#df_train = df_train.drop(columns=['ID'])
display(df_train)
print(df_train.shape)

df_test = pd.read_csv('data/ra/test.csv')
#df_test = df_test.drop(columns=['ID'])
print(df_test.shape)
display(df_test)

# %% Check number of categories in categorical features
cats_count = {2:0, 50:0, 'many':0}

for name in df_train.columns:
    num_cats = df_train[name].nunique()
    if num_cats == 2:
        cats_count[2] += 1    
    elif (num_cats < 50 and num_cats > 2):
        cats_count[50] += 1
    else: 
        cats_count['many'] += 1
        print(name)

print(cats_count)

# %% Define categories

target =['efs', 'efs_time']
nums = ['donor_age', 'age_at_hct'] #, 'efs_time']
cats = [name for name in df_train.columns if name not in (nums + target)]


#%% check nulls

print(df_train[cats].isnull().sum().sum())
print(df_train[nums].isnull().sum().sum())

#%% encode cats


for cname in cats:
    print(cname)    
    
    df_train[cname] = df_train[cname].fillna('Missing')
    df_test[cname] = df_test[cname].fillna('Missing')
    
    train_categories = df_train[cname].value_counts().sort_values().index.get_level_values(0).to_list()
    train_dict = {fname:i for i, fname in enumerate(train_categories)}

    test_categories = df_test[cname].unique()
    test_dict = {}
    for fname in test_categories:
        if fname in train_categories:
            test_dict[fname] = train_dict[fname]
        elif df_train[cname].isnull().any(): 
            test_dict[fname] = train_dict['Missing']
        else:
            test_dict[fname] = 0    
    
    df_train[cname] = df_train[cname].map(train_dict).astype(np.int16)
    df_test[cname] = df_test[cname].map(test_dict).astype(np.int16)       
            
# %%

print(df_train[cats].isnull().sum().sum())
print(df_train[nums].isnull().sum().sum())

# %% Feature Engineering

# # Fill na for nums. Put mean.    
# add_cols = []
# for cname in nums:
#     print(f'fillna for {cname}')
#     # Mark all cols ith nans
#     if df_train[cname].isnull().values.any():
#         print('Add nan col')
#         df_train[f"is_{cname}_na"] = df_train[cname].isna().astype(int)
#         df_test[f"is_{cname}_na"] = df_test[cname].isna().astype(int)
#         add_cols.append(f"is_{cname}_na")

#     # Fill all nans ith mean
#     m = df_train[cname].mean()
#     df_train[cname] = df_train[cname].fillna(m)
#     df_test[cname]  = df_test[cname].fillna(m)

# cats = cats + add_cols

for cname in nums:
    df_train[cname] = df_train[cname].fillna(-1)
    df_test[cname] = df_test[cname].fillna(-1)



# %%

print(df_train[cats].isnull().sum().sum())
print(df_train[nums].isnull().sum().sum())

#%% Combine and transform targets

def transform_survival_probability(df, time_col='efs_time', event_col='efs'):
    kmf = KaplanMeierFitter()
    kmf.fit(df[time_col], event_observed=df[event_col])
    survival_probabilities = kmf.survival_function_at_times(df[time_col]).values.flatten()
    return survival_probabilities


def update_target_with_survival_probabilities(df, time_col='efs_time', event_col='efs'):
    race_group = sorted(df['race_group'].unique())
    survival_probs_dict = {}
    for race in race_group:
        race_df = df[df['race_group'] == race]
        survival_probs_dict[race] = transform_survival_probability(race_df, time_col, event_col)
    for race in race_group:
        df.loc[df['race_group'] == race, 'target'] = survival_probs_dict[race]
    df.loc[df[event_col] == 0, 'target'] -= 0.15 # = -df.loc[df[event_col] == 0, 'target'] # 
    
    return df, survival_probs_dict

# will make "target" the label 
df_train, spd = update_target_with_survival_probabilities(df_train, time_col='efs_time', event_col='efs')

#%%

#df_y = df_train[['y']]
df_y = df_train[['target']]
df_X = df_train.drop(columns=target+['target']) #['y'])
df_X[cats] = df_X[cats].astype('category')

#%%
params = {'tree_method': 'hist',
                'n_estimators': 5000,
                'objective': 'reg:logistic', # 'reg:squarederror', # 'reg:squaredlogerror', #
                'random_state': 42,
                'enable_categorical': True,
                #'verbosity': 0,
                'early_stopping_rounds': 25,
                'eval_metric': "mae",
                
                "max_depth":3,  
                "colsample_bytree":0.5, 
                "subsample":0.8, 
                "learning_rate":0.1, 
                "min_child_weight":5
                }

FEATURES = nums + cats   
KFOLD = 5
kf = KFold(n_splits=KFOLD, shuffle=True, random_state=42)

oof_xgb = np.zeros(len(df_train))
pred_xgb = np.zeros(len(df_test))
best_score = []

for i, (train_idx, valid_idx) in enumerate(kf.split(df_X)):
    print(25*"#")
    print(f"FOLD {i}\n")
    Xtrain = df_X.loc[train_idx, FEATURES].copy()
    ytrain = df_y.loc[train_idx].copy()
    Xvalid = df_X.loc[valid_idx, FEATURES].copy()
    yvalid = df_y.loc[valid_idx].copy()
    
    xgb_model = XGBRegressor(**params)
    
    xgb_model.fit(Xtrain, ytrain, 
                    eval_set = [(Xvalid, yvalid)],
                    verbose = 100)
    
    print('\nBest Fit: ', xgb_model.best_score, '\n')
    best_score.append(xgb_model.best_score)
    y_pred = xgb_model.predict(Xvalid)
    oof_xgb[valid_idx] = y_pred
    
    if i>0:
        preds += np.exp(xgb_model.predict(df_test.loc[:,FEATURES])) - 1
    else:
        preds = np.exp(xgb_model.predict(df_test.loc[:,FEATURES])) - 1
    
preds /= KFOLD    
print(f"Mean Score {np.mean(best_score)}")

#%%

df_sub = pd.read_csv('data/ra/sample_submission.csv')
df_sub['prediction'] = preds
df_sub.to_csv('submissions/xgb1.csv', index=False)

df_check = pd.read_csv('submissions/xgb1.csv')
display(df_check)
 
# %% GET SCORE

y_true = df_train[["ID","efs","efs_time","race_group"]].copy()
y_pred = df_train[["ID"]].copy()
y_pred["prediction"] = oof_xgb # Use -00f_xgb for Chris' solution
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"\nOverall CV for XGBoost =",m)

#%%

display(y_true)
display(y_pred)

# %% CHECK FEATURE IMPORTANCE

feat_g = xgb_model.get_booster().get_score(importance_type='gain')
df_fi = pd.DataFrame.from_dict(data=feat_g, orient='index', columns=['value'])
df_fi = df_fi.sort_values(by=['value'], ascending=False)
display(df_fi.head(10))

print(type(feat_g))

#%% ---------------------------------- XGBOOST SURVIVAL ---------------------------
from sklearn.model_selection import StratifiedKFold

kf = StratifiedKFold(shuffle=True, random_state=1)
TARGET = ['efs', 'efs_time']
extra = ['target']
FEATURES = [col for col in df_train.columns if col not in TARGET+extra]
# XGBoost Cox regression
y = np.where(df_train.efs == 1, df_train.efs_time, -df_train.efs_time)
best_score = []
for fold, (idx_tr, idx_va) in enumerate(kf.split(df_train, df_train.race_group)):
    Xtrain = df_train.iloc[idx_tr][FEATURES]
    Xvalid = df_train.iloc[idx_va][FEATURES]
    ytrain = y[idx_tr]
    yvalid = y[idx_va]
    
    xgb_cox_params = {'objective': 'survival:cox', 
                      'grow_policy': 'depthwise', 
                      'n_estimators': 2000, 
                      'learning_rate': 0.1, #0.0254, 
                      'max_depth': 3, #8, 
                      'reg_lambda': 0.116, 
                      'reg_alpha': 0.139, 
                      'min_child_weight': 5, #23.8,
                      'colsample_bytree': 0.5, #0.59, 
                      'subsample': 0.8, #0.7, 
                      'tree_method': 'hist',
                      'enable_categorical': True,
                      'early_stopping_rounds': 200,
                      
                      }
    xgb_model = xgboost.XGBRegressor(**xgb_cox_params)
    xgb_model.fit(Xtrain, ytrain, 
                  eval_set = [(Xtrain, ytrain) ,(Xvalid, yvalid)],
                  verbose=100) # negative values are considered right censored
    
    
    best_score.append(xgb_model.best_score)
    y_va_pred = xgb_model.predict(Xvalid) # predicts hazard factor
    oof_xgb[idx_va] = y_va_pred
    
    if i>0:
        preds += np.exp(xgb_model.predict(df_test.loc[:,FEATURES])) - 1
    else:
        preds = np.exp(xgb_model.predict(df_test.loc[:,FEATURES])) - 1
    
preds /= KFOLD    
print(f"Mean Score {np.mean(best_score)}")

#%% GET SCORE

y_true = df_train[["ID","efs","efs_time","race_group"]].copy()
y_pred = df_train[["ID"]].copy()
y_pred["prediction"] = oof_xgb # Use -00f_xgb for Chris' solution
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"\nOverall CV for XGBoost =",m)

#%% Trying sksruv --------------------------------------------------------
# Function needs one hot encoded categorical features
# Feed ra y
# DOESN'T ORK
#from sksurv.ensemble import RandomSurvivalForest
from sksurv.ensemble import GradientBoostingSurvivalAnalysis

# Get ytrain and ytest in the right format, numpy recarray
def as_sksurv_recarray(y_frame):
    """Return scikit-survival's specific target format."""
    y_recarray = np.empty(
        shape=y_frame.shape[0],
        dtype=[("efs", np.bool_), ("efs_time", np.float64)],
    )
    y_recarray["efs"] = y_frame["efs"]
    y_recarray["efs_time"] = y_frame["efs_time"]
    return y_recarray

c_check = [cat for cat in cats if cat not in ['ID']]
df_train1 = pd.get_dummies(df_train, columns=c_check, drop_first=True)
df_test1 = pd.get_dummies(df_test, columns=c_check, drop_firt=True)



kf = StratifiedKFold(shuffle=True, random_state=1)
TARGET = ['efs', 'efs_time']
extra = ['target']
FEATURES = [col for col in df_train.columns if col not in TARGET+extra]
# XGBoost Cox regression
#y = np.where(df_train.efs == 1, df_train.efs_time, -df_train.efs_time)
y = df_train1[TARGET]
best_score = []
for fold, (idx_tr, idx_va) in enumerate(kf.split(df_train, df_train.race_group)):
    Xtrain1 = df_train1.iloc[idx_tr][FEATURES]
    Xvalid1 = df_train1.iloc[idx_va][FEATURES]
    ytrain1 = y[idx_tr]
    yvalid1 = y[idx_va]
    

    gbsa = GradientBoostingSurvivalAnalysis()
    # Make sure efs is np.bool_
    gbsa.fit(Xtrain1, as_sksurv_recarray(ytrain1))

    
    best_score.append(xgb_model.best_score)
    y_va_pred = xgb_model.predict(Xvalid1) # predicts hazard factor
    oof_xgb[idx_va] = y_va_pred
    
    if i>0:
        preds += np.exp(xgb_model.predict(df_test1.loc[:,FEATURES])) - 1
    else:
        preds = np.exp(xgb_model.predict(df_test1.loc[:,FEATURES])) - 1
    
preds /= KFOLD    
print(f"Mean Score {np.mean(best_score)}")

#%% GET SCORE

y_true = df_train1[["ID","efs","efs_time","race_group"]].copy()
y_pred = df_train1[["ID"]].copy()
y_pred["prediction"] = oof_xgb # Use -00f_xgb for Chris' solution
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"\nOverall CV for XGBoost =",m)





