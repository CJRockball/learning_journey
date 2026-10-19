#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import time
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, train_test_split
from catboost import CatBoostRegressor, Pool
import lightgbm as lgb

from scipy.stats import rankdata 
import pandas.api.types
from lifelines import KaplanMeierFitter, CoxPHFitter, NelsonAalenFitter
from lifelines.utils import concordance_index

import os
import random
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
seed_everything()

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

def get_cindex(df, oof):
    # Get c-index
    y_true = df[["ID","efs","efs_time","race_group"]].copy()
    y_pred = df[["ID"]].copy()
    y_pred["prediction"] = oof 
    m = score(y_true.copy(), y_pred.copy(), "ID")
    return m


#%% Get data

df_train = pd.read_csv('data/ra/train.csv')
df_test = pd.read_csv('data/ra/test.csv')

# Define categories
target =['efs', 'efs_time']
cats = df_train.select_dtypes(include=["object_", 'bool']).columns
cats = [cat for cat in cats if cat not in target+['ID']]
nums = df_train.select_dtypes(include=['number']).columns
nums = [num for num in nums if num not in target+['ID']]


# Replace missing
for cname in cats:
    df_train[cname] = df_train[cname].fillna('Missing')
    df_test[cname] = df_test[cname].fillna('Missing')
for cname in nums:
    df_train[cname] = df_train[cname].fillna(-1)
    df_test[cname] = df_test[cname].fillna(-1)


# Ordinal encoding
def ord_trans(df, df1, features):
    for cname in features:
        train_categories = df[cname].value_counts().sort_values().index.get_level_values(0).to_list()
        train_dict = {fname:i for i, fname in enumerate(train_categories)}

        test_categories = df1[cname].unique()
        test_dict = {}
        for fname in test_categories:
            if fname in train_categories:
                test_dict[fname] = train_dict[fname]
            elif df[cname].isnull().any(): 
                test_dict[fname] = train_dict['Missing']
            else:
                test_dict[fname] = 0    
    
        df[cname] = df[cname].map(train_dict).astype(np.int16)
        df1[cname] = df1[cname].map(test_dict).astype(np.int16)       
    return df, df1

df_train, df_test = ord_trans(df_train, df_test, cats)


# Set up dfs
FEATURES = nums + cats 
df_target = df_train[target].copy()
df_X = df_train.loc[:, FEATURES] 
df_X[cats] = df_X[cats].astype('category')
df_test[cats] = df_test[cats].astype('category')


#%% Kaplan-Meier by race
# Kaplan-Meier by race
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

# Calculate km by race, add to df_target 
df_train = update_target_with_survival_probabilities(df_train, time_col='efs_time', event_col='efs')
df_train.drop(columns='km')
df_target['km'] = df_train['km']
KFOLD = 5
kf = KFold(n_splits=KFOLD, shuffle=True, random_state=42)
#%% TE
# from sklearn.preprocessing import TargetEncoder


# add_col = []
# for cname in cats:
#     ne_name = cname + "_TE"
#     enc_auto = TargetEncoder(smooth="auto", random_state=42)        
#     df_train[ne_name] = enc_auto.fit_transform(df_train[[str(cname)]], df_target['km'].values.ravel())
#     df_test[ne_name] = enc_auto.transform(df_test[[str(cname)]])
#     add_col.append(ne_name)

# cats = cats + add_col
# FEATURES = nums + cats

# df_X = df_train.loc[:,FEATURES] #.drop(columns=target)
# df_X[cats] = df_X[cats].astype('category')
# df_test[cats] = df_test[cats].astype('category')




#%%
   
def cv_func(df_X, df_y, df_train, cats_opt, kf, train_fcn, params, NUM_ROUNDS):
    oof_lgb = np.zeros(len(df_train))
    c_list = []

    for i, (train_idx, valid_idx) in enumerate(kf.split(df_X)):
    # Set up data
        Xtrain = df_X.loc[train_idx].copy()
        ytrain = df_y.loc[train_idx].copy()
        Xvalid = df_X.loc[valid_idx].copy()
        yvalid = df_y.loc[valid_idx].copy()
        
        # Fit model
        y_pred = train_fcn(Xtrain, Xvalid, ytrain, yvalid, cats_opt, params, NUM_ROUNDS)

        # Eval, predict, save
        oof_lgb[valid_idx] = y_pred
        c2 = get_cindex(df_train.iloc[valid_idx], y_pred)
        c_list.append(c2)

    full_score = get_cindex(df_train, oof_lgb)
    return c_list, full_score


def lgb_train(Xtrain, Xvalid, ytrain, yvalid, cats_opt, params, NUM_ROUNDS):
# Make model, train
    trn_data = lgb.Dataset(Xtrain, ytrain, categorical_feature=cats_opt)
    lgb_eval = [lgb.Dataset(Xvalid, yvalid)]
    
    lgb_model = lgb.train(params, trn_data, NUM_ROUNDS,
                      valid_sets=lgb_eval, #feval=rmsle_lgbm, 
                      callbacks = [
                          #lgb.log_evaluation(1000),
                          lgb.early_stopping(stopping_rounds=25, verbose=0)
                      ])
    
    y_pred = lgb_model.predict(Xvalid)
    
    return y_pred

#%%


params = {
    "boosting_type": "gbdt",
    "objective": "regression",
    "metric": 'mae', #'rmse', #'custom',
    "verbose": 0,
    'random_state': 42,
    'n_jobs':-1,
    
    'learning_rate': 0.02,
    'num_leaves': 32,
    'min_child_weight':1,
    'subsample': 0.502742,
    'colsample_bytree': 0.373231,
    'reg_alpha':1, 
    'reg_lambda':1,
    }

NUM_ROUNDS = 3000
df_y = df_target[['km']].copy() #'na_hazard']].copy() #'cox_hazard']].copy() #

start_time = time.time()
c_list, full_score = cv_func(df_X.loc[:, FEATURES], df_y, df_train, cats, kf, lgb_train, params, NUM_ROUNDS)
end_time = time.time()
print(f'Total time: {end_time - start_time}')
print(f'oof score: {full_score}, fold average: {np.mean(c_list)}')
end_time = time.time()

# %% Check for ne features
df_X2 = df_X.loc[:, FEATURES].copy()
c_list, full_score = cv_func(df_X2, df_y, df_train, cats, kf, lgb_train, params, NUM_ROUNDS)
best_score = full_score
mean_score = np.mean(c_list)
print(f'oof score {best_score}, Mean score: {mean_score}')

def add_cat_feature(df, new_feature):
    fname = "_".join(new_feature)
    # Make new feature
    feat_length = len(new_feature)
    if feat_length == 2:
        df[fname] = df[new_feature[0]].astype(str) + df[new_feature[1]].astype(str)
    elif feat_length == 3:
        df[fname] = df[new_feature[0]].astype(str) + df[new_feature[1]].astype(str) +\
                    df[new_feature[2]].astype(str)
    elif feat_length == 4:
        df[fname] = df[new_feature[0]].astype(str) + df[new_feature[1]].astype(str) +\
                        df[new_feature[2]].astype(str) + df[new_feature[3]].astype(str)
    
    # Ordinal encode new features
    train_categories = df[fname].value_counts().sort_values().index.get_level_values(0).to_list()
    train_dict = {cname:i for i, cname in enumerate(train_categories)}    
    df[fname] = df[fname].map(train_dict).astype('category') #np.int16)
    
    return df

start_time = time.time()
MORE = []
np.random.seed(17)
for k in range(100):
    if k % 10 == 0:
        print(f"Number of tests run: {k}, Number of features: {df_X2.shape[1]}")
    ct = np.random.choice([2, 3, 4], 1)[0]
    new_feature = list( np.random.choice(cats, ct, replace=False))
    df_X2 = add_cat_feature(df_X2, new_feature)
    
    c_list, new_score = cv_func(df_X2, df_y, df_train, cats, kf, lgb_train, params, NUM_ROUNDS)
    mean_score = np.mean(c_list)
    
    if new_score > best_score:
        MORE.append(new_feature)
        perc_imp = ((new_score - best_score) / best_score) * 100
        best_score = new_score
        print(f'{new_feature},\n oof score {new_score:.5f}, mean score {mean_score:.5f}, improvement {perc_imp}%')
        
    else:
        fname = "_".join(new_feature)
        df_X2 = df_X2.drop(columns=[fname])

end_time = time.time()
print(f'Total time: {end_time - start_time}')
print("DONE")

#%%
np.random.seed(42)
print(MORE)
print(df_X2.shape)
print(df_X2.columns)

# %%
# TODO
# 1 rerun df_X2 to check
# 2 add ne features and rerun df_train to check

c_list, new_score = cv_func(df_X2, df_y, df_train, cats, kf, lgb_train, params, NUM_ROUNDS)
print(f'Full core: {new_score}')

#%% Add feature to org df

df_X3 = df_X.loc[:, FEATURES].copy(deep=True)

# Add descrete features
add_list = [['cyto_score_detail', 'rheum_issue'], ['in_vivo_tcd', 'tbi_status', 'arrhythmia', 'rituximab'], \
    ['hepatic_severe', 'cyto_score_detail'], ['hepatic_mild', 'ethnicity', 'in_vivo_tcd'], \
        ['dri_score', 'graft_type']]
#[['rheum_issue', 'rituximab', 'peptic_ulcer', 'tce_match'], ['donor_related', 'in_vivo_tcd'], ['tce_match', 'obesity', 'dri_score', 'diabetes']]
add_col = []

def make_cats_features(df, df1, add_list):
    for lst in add_list:
        fname = "_".join(lst)
        if len(lst) == 2:
            df[fname] = df[lst[0]].astype(str) + df[lst[1]].astype(str)
            df1[fname] = df1[lst[0]].astype(str) + df1[lst[1]].astype(str)
        elif len(lst) == 3:
            df[fname] = df[lst[0]].astype(str) + df[lst[1]].astype(str) + df[lst[2]].astype(str)
            df1[fname] = df1[lst[0]].astype(str) + df1[lst[1]].astype(str) + df1[lst[2]].astype(str)
        elif len(lst) == 4:
            df[fname] = df[lst[0]].astype(str) + df[lst[1]].astype(str) + df[lst[2]].astype(str) + df_train[lst[3]].astype(str)
            df1[fname] = df1[lst[0]].astype(str) + df1[lst[1]].astype(str) + df1[lst[2]].astype(str) + df_test[lst[3]].astype(str)       
        add_col.append(fname)
        
        # Ordinal encode new features
        train_categories = df[fname].value_counts().sort_values().index.get_level_values(0).to_list()
        train_dict = {cname:i for i, cname in enumerate(train_categories)}    
        df[fname] = df[fname].map(train_dict).astype('category') #np.int16)
        
    return df, df1, add_col            


df_X3, df_test, add_col = make_cats_features(df_X3, df_test, add_list)

c_list, new_score = cv_func(df_X3, df_y, df_train, cats, kf, lgb_train, params, NUM_ROUNDS)
print(f'Full core: {new_score}')




# %%
print(MORE)


# %%
"""
['rheum_issue', 'rituximab', 'peptic_ulcer', 'tce_match'], 
oof score 0.67661, mean score 0.67420, improvement 0.02702585403758513%
['donor_related', 'in_vivo_tcd'], oof score 0.67679, 
mean score 0.67461, improvement 0.027536865989041884%
['tce_match', 'obesity', 'dri_score', 'diabetes'],
oof score 0.67738, mean score 0.67462, improvement 0.08681325567111146%

['cyto_score_detail', 'rheum_issue'],
 oof score 0.67645, mean score 0.67376, improvement 0.003396240062951288%
['in_vivo_tcd', 'tbi_status', 'arrhythmia', 'rituximab'],
 oof score 0.67663, mean score 0.67411, improvement 0.027733887120068704%
['hepatic_severe', 'cyto_score_detail'],
 oof score 0.67690, mean score 0.67433, improvement 0.03929177680939945%
['hepatic_mild', 'ethnicity', 'in_vivo_tcd'],
 oof score 0.67726, mean score 0.67460, improvement 0.05291608028508737%
['dri_score', 'graft_type'],
 oof score 0.67766, mean score 0.67553, improvement 0.059115212180747626%
[['cyto_score_detail', 'rheum_issue'], ['in_vivo_tcd', 'tbi_status', 'arrhythmia', 'rituximab'], ['hepatic_severe', 'cyto_score_detail'], ['hepatic_mild', 'ethnicity', 'in_vivo_tcd'], ['dri_score', 'graft_type']]



"""
