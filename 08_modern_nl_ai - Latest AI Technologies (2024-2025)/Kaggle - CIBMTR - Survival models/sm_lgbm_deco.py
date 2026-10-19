#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.model_selection import KFold, train_test_split
from xgboost import XGBRegressor, XGBClassifier
import xgboost
print("Using XGBoost version",xgboost.__version__)
import lightgbm as lgb

from scipy.stats import rankdata 
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
print('load')
df_train = pd.read_csv('data/ra/train.csv')
df_test = pd.read_csv('data/ra/test.csv')


# Define categories
target =['efs', 'efs_time']
cats = df_train.select_dtypes(include=["object_", 'bool']).columns
cats = [cat for cat in cats if cat not in target+['ID']]
nums = df_train.select_dtypes(include=['number']).columns
nums = [num for num in nums if num not in target+['ID']]

# encode cats
print('cats')
for cname in cats:
    df_train[cname] = df_train[cname].fillna('Missing')
    df_test[cname] = df_test[cname].fillna('Missing')


# Add descrete features
add_list = [['prod_type', 'vent_hist'], ['melphalan_dose', 'in_vivo_tcd', 'cyto_score', 'prim_disease_hct'], ['obesity', 'hepatic_severe', 'rheum_issue'], ['mrd_hct', 'melphalan_dose', 'ethnicity'], ['ethnicity', 'mrd_hct', 'prior_tumor'], ['tce_match', 'vent_hist', 'pulm_moderate'], ['in_vivo_tcd', 'arrhythmia']]
add_col = []

def make_cats_features(df_train, df_test, add_list):
    for lst in add_list:
        fname = "_".join(lst)
        if len(lst) == 2:
            print(lst[0])
            df_train[fname] = df_train[lst[0]].astype(str) + df_train[lst[1]].astype(str)
            df_test[fname] = df_test[lst[0]].astype(str) + df_test[lst[1]].astype(str)
        elif len(lst) == 3:
            df_train[fname] = df_train[lst[0]].astype(str) + df_train[lst[1]].astype(str) + df_train[lst[2]].astype(str)
            df_test[fname] = df_test[lst[0]].astype(str) + df_test[lst[1]].astype(str) + df_test[lst[2]].astype(str)
        elif len(lst) == 4:
            df_train[fname] = df_train[lst[0]].astype(str) + df_train[lst[1]].astype(str) + df_train[lst[2]].astype(str) + df_train[lst[3]].astype(str)
            df_test[fname] = df_test[lst[0]].astype(str) + df_test[lst[1]].astype(str) + df_test[lst[2]].astype(str) + df_test[lst[3]].astype(str)       
        add_col.append(fname)
        
    return df_train, df_test, add_col            

df_train, df_test, add_col = make_cats_features(df_train, df_test, add_list)
cats = cats + add_col





for cname in cats:
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

# # Encode num
# print('nums')
# for cname in nums:
#     df_train[cname] = df_train[cname].fillna(-1)
#     df_test[cname] = df_test[cname].fillna(-1)


# Set up dataframes
print('set up dfs')
df_target = df_train[target].copy()
df_X = df_train.drop(columns=target).copy()
df_X[cats] = df_X[cats].astype('category')
df_test[cats] = df_test[cats].astype('category')
FEATURES = nums + cats



#%% Calc targets KM, NA, CoxPH

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

#%%

def c_score(df, y_pred_):
    y_true = df[["ID","efs","efs_time","race_group"]].copy()
    y_pred = df[["ID"]].copy()
    y_pred["prediction"] = y_pred_
    m = score(y_true, y_pred, "ID")
    return m


def cv_func(df_X, df_y, df_train, cats_opt, kf, train_fcn, params, NUM_ROUNDS):
    oof_lgb = np.zeros(len(df_train))
    c_list = []
    full_index = []
    for i, (train_idx, valid_idx) in enumerate(kf.split(df_X)):
    
    # Set up data
        Xtrain = df_X.loc[train_idx,:].copy()
        ytrain = df_y.loc[train_idx].copy()
        Xvalid = df_X.loc[valid_idx,:].copy()
        yvalid = df_y.loc[valid_idx].copy()
        full_index = full_index + list(valid_idx)
        
        y_pred = train_fcn(Xtrain, Xvalid, ytrain, yvalid, cats_opt, params, NUM_ROUNDS)

        oof_lgb[valid_idx] = y_pred
        c2 = c_score(df_train.iloc[valid_idx], y_pred)
        c_list.append(c2)

    full_score = c_score(df_train, oof_lgb)
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
    "verbose": -1,
    'random_state': 42,
    'n_jobs':4,
    "feature_pre_filter" : False,
    
    'learning_rate': 0.02,
    'num_leaves': 32,
    'min_child_weight':1,
    'subsample': 0.502742,
    'colsample_bytree': 0.373231,
    'reg_alpha':1, 
    'reg_lambda':1
        }

NUM_ROUNDS = 3000
df_y = df_target[['km']].copy()
df_X = df_X.loc[:,FEATURES]
KFOLD = 5
kf = KFold(n_splits=KFOLD, shuffle=True, random_state=42)

start_time = time.time()
c_list, full_score = cv_func(df_X, df_y, df_train, cats, kf, lgb_train, params, NUM_ROUNDS)
end_time = time.time()
print(f'Total time {end_time - start_time}')

print(full_score)


# %%
