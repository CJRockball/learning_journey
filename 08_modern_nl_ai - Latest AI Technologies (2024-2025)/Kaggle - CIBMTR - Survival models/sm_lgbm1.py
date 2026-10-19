#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, train_test_split
from xgboost import XGBRegressor, XGBClassifier
import xgboost
print("Using XGBoost version",xgboost.__version__)

import pandas.api.types
from lifelines import KaplanMeierFitter, CoxPHFitter, NelsonAalenFitter
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

#%% Get data

df_train = pd.read_csv('data/ra/train.csv')
#df_train = df_train.drop(columns=['ID'])
df_train = df_train.astype({'efs':bool})
display(df_train)
print(df_train.shape)

df_test = pd.read_csv('data/ra/test.csv')
#df_test = df_test.drop(columns=['ID'])
print(df_test.shape)
display(df_test)

# %% Define categories

target =['efs', 'efs_time']
# nums = ['donor_age', 'age_at_hct'] 
# cats = [name for name in df_train.columns if name not in (nums + target + ['ID'])]

cats = df_train.select_dtypes(include=["object_", 'bool']).columns
cats = [cat for cat in cats if cat not in target+['ID']]
nums = df_train.select_dtypes(include=['number']).columns
nums = [num for num in nums if num not in target+['ID']]

#%% encode cats


for cname in cats:
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

#%%

df_target = df_train[target].copy()
df_X = df_train.drop(columns=target)
df_X[cats] = df_X[cats].astype('category')
df_test[cats] = df_test[cats].astype('category')
FEATURES = nums + cats 

#%% ------------------------- CHOOSE SURVIVAL MODEL --------------------- 
# Nelson Aalen model

naf = NelsonAalenFitter()
naf.fit(durations=df_train['efs_time'], event_observed=df_train['efs'])
df_target['na_hazard'] = naf.cumulative_hazard_at_times(df_train['efs_time']).values
df_target['na_hazard'] = df_target['na_hazard'] * -1

#%% Kaplan-Meier by race

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
        df.loc[df['race_group'] == race, 'km'] = survival_probs_dict[race]
    df.loc[df[event_col] == 0, 'km'] -= 0.18
    
    return df

# # will make "target" the label 
df_train = update_target_with_survival_probabilities(df_train, time_col='efs_time', event_col='efs')
df_train.drop(columns='km')
df_target['km'] = df_train['km']

#%%

params = {
    "boosting_type": "gbdt",
    "objective": "regression",
    "metric": 'mae', #'rmse', #'custom',
    "verbose": -1,
    'random_state': 42,
    'n_jobs':4,
    "feature_pre_filter" : False
}

def make_params():
    param_test ={'learning_rate': 0.02, #choices([0.1, 0.05, 0.01, 0.005, 0.001],k=1)[0],
                 #'n_estimators': choices([150, 200, 250, 300, 350, 500],k=1)[0],
                 'num_leaves': 32, #sp_randint(30, 40).rvs(), 
                #'min_child_samples': 299, #sp_randint(250, 300).rvs(), 
                'min_child_weight':1, #choices([1e-2, 1e-1, 1, 1e1, 1e2,],k=1)[0],
                'subsample': 0.502742, #sp_uniform(loc=0.5, scale=0.2).rvs(), 
                'colsample_bytree': 0.373231, #sp_uniform(loc=0.3, scale=0.15).rvs(),
                'reg_alpha':1, #choices([1, 2, 5, 7, 10],k=1)[0], # 25, 50, 100],k=1)[0],
                'reg_lambda':1 # choices([0, 0.5, 1, 2, 5, 7, 10],k=1)[0], #, 25, 50, 100],k=1)[0]
                }
    return param_test



#%%
import lightgbm as lgb

FEATURES = nums + cats   
NUM_ROUNDS = 3000
KFOLD = 5
kf = KFold(n_splits=KFOLD, shuffle=True, random_state=42)
df_y = df_target[['km']].copy() #'na_hazard']].copy() #'cox_hazard']].copy() #
Xtest = df_test.loc[:,FEATURES]
oof_lgb = np.zeros(len(df_train))
pred_lgb = np.zeros(len(df_test))
best_score = []
    
for i, (train_idx, valid_idx) in enumerate(kf.split(df_X)):
    print(25*"#")
    print(f"FOLD {i}\n")
    
    # Set up data
    Xtrain = df_X.loc[train_idx, FEATURES].copy()
    ytrain = df_y.loc[train_idx].copy()
    Xvalid = df_X.loc[valid_idx, FEATURES].copy()
    yvalid = df_y.loc[valid_idx].copy()
    
    # Make model, train
    trn_data = lgb.Dataset(Xtrain, ytrain, categorical_feature=cats)
    lgb_eval = [lgb.Dataset(Xvalid, yvalid)]
    lgb_model_params = make_params()
    
    lgb_model = lgb.train(params|lgb_model_params, trn_data, NUM_ROUNDS,
                      valid_sets=lgb_eval, #feval=rmsle_lgbm, 
                      callbacks = [
                          lgb.log_evaluation(1000),
                          lgb.early_stopping(stopping_rounds=25)
                      ])
    
    # Eval, predict, save
    print('\nBest Fit: ', lgb_model.best_score['valid_0']['l1'], '\n')
    best_score.append(lgb_model.best_score['valid_0']['l1'])
        
    
    
    y_pred = lgb_model.predict(Xvalid)
    oof_lgb[valid_idx] = y_pred
    
    if i>0:
        preds += lgb_model.predict(Xtest)
    else:
        preds = lgb_model.predict(Xtest)
    
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
y_pred["prediction"] = oof_lgb # Use -00f_xgb for Chris' solution
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"\nOverall CV for XGBoost =",m)

#%%
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
import random
from random import sample, choices

params = {'tree_method': 'hist',
                'n_estimators': 5000,
                'objective': 'reg:absoluteerror', #'reg:squarederror', #'reg:logistic', #  'reg:squaredlogerror', #
                'random_state': 42,
                'enable_categorical': True,
                #'verbosity': 0,
                'early_stopping_rounds': 25,
                'eval_metric': "mae",
                # Using Fisher grouping of categorical features. None=One hot
                "max_cat_to_onehot" : None,
                }

def make_params():
    param_test ={'learning_rate': 0.01, #choices([0.1, 0.05, 0.01, 0.005, 0.001],k=1)[0],
                #'n_estimators': choices([150, 200, 250, 300, 350, 500],k=1)[0],
                'num_leaves': 32, #sp_randint(30, 40).rvs(), 
                'min_child_samples': 299, #sp_randint(250, 300).rvs(), 
                'min_child_weight':1, #choices([1e-2, 1e-1, 1, 1e1, 1e2,],k=1)[0],
                'subsample': 0.502742, #sp_uniform(loc=0.5, scale=0.2).rvs(), 
                'colsample_bytree': 0.373231, #sp_uniform(loc=0.3, scale=0.15).rvs(),
                'reg_alpha':1, #choices([1, 2, 5, 7, 10],k=1)[0], # 25, 50, 100],k=1)[0],
                'reg_lambda':1 # choices([0, 0.5, 1, 2, 5, 7, 10],k=1)[0], #, 25, 50, 100],k=1)[0]
                } 
    return param_test

def training_params():
    params = {
            'grow_policy': 'depthwise',
              'learning_rate':choices([5e-2],k=1)[0],
              #'n_estimators': 275, # 68,
              'max_depth': choices([ 5,],k=1)[0], #19, # 
              'min_child_weight': sp_uniform(loc=4.52, scale=0.03).rvs(), # 2.967
              'subsample': sp_uniform(loc=0.80, scale=0.05).rvs(), #1, #0.957, #  
              'colsample_bytree': sp_uniform(loc=0.68, scale=0.02).rvs(), #0.446875, # 
              'reg_alpha':choices([10],k=1)[0], # 25, 50, 100],k=1)[0],
              'reg_lambda':choices([5,],k=1)[0], #, 25, 50, 100],k=1)[0]  
              'gamma': choices([ 1e-3],k=1)[0],}
    
    return params


#%%
from sklearn.metrics import mean_absolute_error
import time

start_time = time.time()
n_trials = 5
features = training_params()
# Make df to save trials
col_list = list(features.keys()) + ['mean_mae', 'c-index', 'num-rounds']
random_search = pd.DataFrame(data=len(col_list)*[0], index=col_list)

FEATURES = nums + cats   
KFOLD = 5
kf = KFold(n_splits=KFOLD, shuffle=True, random_state=42)
df_y = df_target[['km']].copy() #'na_hazard']].copy() #'cox_hazard']].copy() #

for k in range(n_trials):
    print(f'*** Trial {k} ***')
    oof_xgb = np.zeros(len(df_train))
    pred_xgb = np.zeros(len(df_test))
    best_score = []
    num_iter = []
    test_params = training_params()

    for i, (train_idx, valid_idx) in enumerate(kf.split(df_X)):
    #print(25*"#")
    #print(f"FOLD {i}\n")
        Xtrain = df_X.loc[train_idx, FEATURES].copy()
        ytrain = df_y.loc[train_idx].copy()
        Xvalid = df_X.loc[valid_idx, FEATURES].copy()
        yvalid = df_y.loc[valid_idx].copy()
    

        xgb_model = XGBRegressor(**(params | test_params))
        xgb_model.fit(Xtrain, ytrain, 
                    eval_set = [(Xvalid, yvalid)],
                    verbose = 5000)   
    
        num_iter.append(xgb_model.best_iteration)
        best_score.append(xgb_model.best_score)
        y_pred = xgb_model.predict(Xvalid)
        oof_xgb[valid_idx] = y_pred


    fit_data = test_params #xgb_model.get_xgb_params()
    fit_data['mean_mae'] = np.mean(best_score)
    y_true = df_train[["ID","efs","efs_time","race_group"]].copy()
    y_pred = df_train[["ID"]].copy()
    y_pred["prediction"] = oof_xgb # Use -00f_xgb for Chris' solution
    m = score(y_true.copy(), y_pred.copy(), "ID")
    fit_data['c-index'] = m
    print(f'C-index {m}')
    fit_data['num-rounds'] = np.sum(num_iter)

    random_search[k] = random_search.index.map(fit_data)

print('**** DONE ****')
end_time = time.time()
print(f"Total time: {end_time - start_time}")   

# %%

display(random_search.T.sort_values(by='c-index'))


# %%

a = random_search.T.sort_values(by='c-index')
b = a.iloc[9,:].to_dict()

print(b)



# %%
