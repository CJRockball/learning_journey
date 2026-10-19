#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, train_test_split

from xgboost import XGBRegressor, XGBClassifier
import xgboost
print("Using XGBoost version",xgboost.__version__)
import lightgbm as lgb
from catboost import CatBoostRegressor, Pool


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

print("Get Data")
df_train = pd.read_csv('data/ra/train.csv')
df_train = df_train.astype({'efs':bool})
df_test = pd.read_csv('data/ra/test.csv')


# Define categories
target =['efs', 'efs_time']
cats = df_train.select_dtypes(include=["object_", 'bool']).columns
cats = [cat for cat in cats if cat not in target+['ID']]
nums = df_train.select_dtypes(include=['number']).columns
nums = [num for num in nums if num not in target+['ID']]

# encode cats
print('Feature Engineering')
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

# Feature Engineering, fill nans for numerical
for cname in nums:
    df_train[cname] = df_train[cname].fillna(-1)
    df_test[cname] = df_test[cname].fillna(-1)

# Data transforming
df_target = df_train[target].copy()
df_X = df_train.drop(columns=target)
df_X[cats] = df_X[cats].astype('category')
df_test[cats] = df_test[cats].astype('category')
FEATURES = nums + cats 

# Kaplan-Meier by race
print('KM')
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
df_target['km'] = df_train['km']
df_train.drop(columns='km')

#%%
print('LGB')
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
    param_test ={'learning_rate': 0.02,
                 'num_leaves': 32,
                'min_child_weight':1,
                'subsample': 0.502742,
                'colsample_bytree': 0.373231,
                'reg_alpha':1,
                'reg_lambda':1 
                }
    return param_test

FEATURES = nums + cats   
NUM_ROUNDS = 3000
KFOLD = 5
kf = KFold(n_splits=KFOLD, shuffle=True, random_state=42)
df_y = df_target[['km']].copy() #'na_hazard']].copy() #'cox_hazard']].copy() #
Xtest = df_test.loc[:,FEATURES]
oof_lgb = np.zeros(len(df_train))
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
        pred_lgb += lgb_model.predict(Xtest)
    else:
        pred_lgb = lgb_model.predict(Xtest)
    
pred_lgb /= KFOLD
pred_lgb = pred_lgb / np.max(pred_lgb)
oof_lgb = oof_lgb / np.max(oof_lgb)
print(f"Mean Score {np.mean(best_score)}")

#%%
print("XGB")
params = {'tree_method': 'hist',
                'n_estimators': 5000,
                'objective': 'reg:squarederror', #'reg:absoluteerror', #'reg:logistic', #  'reg:squaredlogerror', #
                'random_state': 42,
                'enable_categorical': True,
                'early_stopping_rounds': 25,
                'eval_metric': "mae",
                # Using Fisher grouping of categorical features. None=One hot
                "max_cat_to_onehot" : None,
                
                "max_depth":3,  
                "colsample_bytree":0.5, 
                "subsample":0.8, 
                "learning_rate":0.2, 
                "min_child_weight":5
                }

FEATURES = nums + cats   
KFOLD = 5
kf = KFold(n_splits=KFOLD, shuffle=True, random_state=42)
df_y = df_target[['km']].copy()
oof_xgb = np.zeros(len(df_train))
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
                    verbose = False)
    
    print('\nBest Fit: ', xgb_model.best_score, '\n')
    best_score.append(xgb_model.best_score)
    y_pred = xgb_model.predict(Xvalid)
    oof_xgb[valid_idx] = y_pred
    
    if i>0:
        pred_xgb += xgb_model.predict(df_test.loc[:,FEATURES])
    else:
        pred_xgb = xgb_model.predict(df_test.loc[:,FEATURES])
    
pred_xgb /= KFOLD
pred_xgb = pred_xgb / np.max(pred_xgb)
oof_xgb = oof_xgb / np.max(oof_xgb)   
print(f"Mean Score {np.mean(best_score)}")

#%%


FEATURES = nums + cats   

KFOLD = 5
kf = KFold(n_splits=KFOLD, shuffle=True, random_state=42)
df_y = df_target[['km']].copy() #'na_hazard']].copy() #'cox_hazard']].copy() #
Xtest = df_test.loc[:,FEATURES]
oof_cat = np.zeros(len(df_train))
best_score = []
    
for i, (train_idx, valid_idx) in enumerate(kf.split(df_X)):
    print(25*"#")
    print(f"FOLD {i}\n")
    
    # Set up data
    Xtrain = df_X.loc[train_idx, FEATURES].copy()
    ytrain = df_y.loc[train_idx].copy()
    Xvalid = df_X.loc[valid_idx, FEATURES].copy()
    yvalid = df_y.loc[valid_idx].copy()
    
    # Fit model
    train_pool = Pool(data=Xtrain, label=ytrain, cat_features=cats)
    valid_pool = Pool(data=Xvalid, label=yvalid, cat_features=cats)
        
    cat_params = make_params()
    cat_model = CatBoostRegressor(grow_policy='Lossguide',task_type='GPU', verbose=1000) #**cat_params,)
        
    cat_model.fit(train_pool, eval_set=valid_pool)
        
    # Eval, predict, save
    y_pred = cat_model.predict(Xvalid)
    mae_ = mean_absolute_error(yvalid, y_pred)
    
    print('\nBest Fit: ', mae_, '\n')
    best_score.append(mae_)
    oof_cat[valid_idx] = y_pred
    
    if i>0:
        pred_cat += cat_model.predict(Xtest)
    else:
        pred_cat = cat_model.predict(Xtest)
    
pred_cat /= KFOLD
pred_cat = pred_cat / np.max(pred_cat)
oof_cat = oof_cat / np.max(oof_cat)   
print(f"Mean Score {np.mean(best_score)}")


#%%

preds = pred_lgb + pred_xgb + pred_cat
oof = oof_lgb + oof_xgb + oof_cat

#%%

df_sub = pd.read_csv('data/ra/sample_submission.csv')
df_sub['prediction'] = preds
df_sub.to_csv('submissions/xgb1.csv', index=False)

df_check = pd.read_csv('submissions/xgb1.csv')
display(df_check)
 
# %% GET SCORE

y_true = df_train[["ID","efs","efs_time","race_group"]].copy()
y_pred = df_train[["ID"]].copy()
y_pred["prediction"] = oof
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"\nOverall CV for XGBoost =",m)

# %%
