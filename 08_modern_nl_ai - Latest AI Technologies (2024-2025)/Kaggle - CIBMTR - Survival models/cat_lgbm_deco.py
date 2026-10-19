""" Clean data set up
    Using function call in cv traning to reuse code
"""

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, train_test_split
from xgboost import XGBRegressor, XGBClassifier
import xgboost
print("Using XGBoost version",xgboost.__version__)
import lightgbm as lgb
from catboost import CatBoostRegressor, Pool

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

#GET SCORE
def c_index(df_train, data_array):
    y_true = df_train[["ID","efs","efs_time","race_group"]].copy()
    y_pred = df_train[["ID"]].copy()
    y_pred["prediction"] = data_array
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

# encode cats
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


# Num features, replace na
for cname in nums:
    df_train[cname] = df_train[cname].fillna(-1)
    df_test[cname] = df_test[cname].fillna(-1)


# Set up dataframes
df_target = df_train[target].copy()
df_X = df_train.drop(columns=target)
df_X[cats] = df_X[cats].astype('category')
df_test[cats] = df_test[cats].astype('category')
FEATURES = nums + cats 

#%% Kaplan-Meier by race

# KM
KFOLD = 5
oof_pred = np.zeros(len(df_target))
kf = KFold(n_splits=KFOLD, shuffle=True, random_state=42)

for train_index, valid_index in kf.split(df_target):
    train_data = df_target.iloc[train_index]
    valid_data = df_target.iloc[valid_index]
    
    kmf = KaplanMeierFitter()
    kmf.fit(durations=train_data['efs_time'], event_observed=train_data['efs'])
    oof_pred[valid_index] = kmf.survival_function_at_times(valid_data['efs_time']).values

df_target['km'] = oof_pred
m = c_index(df_train, df_target['km'])
print('km', m)


# NA
oof_pred = np.zeros(len(df_target))
for train_index, valid_index in kf.split(df_target):

    train_data = df_target.iloc[train_index]
    valid_data = df_target.iloc[valid_index]
            
    # Make NA
    naf = NelsonAalenFitter()
    naf.fit(durations=train_data['efs_time'], event_observed=train_data['efs'])
    oof_pred[valid_index] = - naf.cumulative_hazard_at_times(valid_data['efs_time']).values

df_target['na_hazard'] = oof_pred
m = c_index(df_train, df_target['na_hazard'])
print('na', m)


# CoxPH
oof_pred = np.zeros(len(df_target))
for train_index, valid_index in kf.split(df_target):

    train_data = df_train.iloc[train_index]
    valid_data = df_train.iloc[valid_index]
            

    cph = CoxPHFitter(penalizer=0.01) #, l1_ratio=0.01)
    cph.fit(train_data.loc[:,FEATURES + ['efs', 'efs_time']], duration_col='efs_time', event_col='efs', strata='race_group')
    #cph.print_summary()
    oof_pred[valid_index] = cph.predict_partial_hazard(valid_data.loc[:,FEATURES + ['efs', 'efs_time']])

df_target['cox_ph'] = oof_pred
m = c_index(df_train, df_target['cox_ph'])
print('cox_ph', m)


# Target for gbm surv 
df_target['target4'] = df_train['efs_time'].copy()
df_target.loc[df_train['efs'] == 0, 'target4'] *= -1
m = c_index(df_train, df_target['target4'])
print('target4', m)

#%% Fitting Kaplan-Meier
NUM_ROUNDS = 3000
Xtest = df_test.loc[:,FEATURES]


def cv_func(df_X, df_y, Xtest, df_train,  kf, train_fcn, cats_opt, params, NUM_ROUNDS):
    oof_pred = np.zeros(len(df_train))
    for i, (train_idx, valid_idx) in enumerate(kf.split(df_X)):
        print(25*"#")
        print(f"FOLD {i}\n")
    
    # Set up data
        Xtrain = df_X.loc[train_idx, FEATURES].copy()
        ytrain = df_y.loc[train_idx].copy()
        Xvalid = df_X.loc[valid_idx, FEATURES].copy()
        yvalid = df_y.loc[valid_idx].copy()
    
        y_pred, test_pred = train_fcn(Xtrain, Xvalid, Xtest, ytrain, yvalid, cats_opt, params, NUM_ROUNDS)

        # Predict, eval and save        
        oof_pred[valid_idx] = y_pred
    
        if i>0:
            test_preds += test_pred
        else:
            test_preds = test_pred
    
    test_preds /= KFOLD   
    m = c_index(df_train, oof_pred)
    print(f"c-index: {m}")
    
    return oof_pred, test_preds


def lgb_train(Xtrain, Xvalid, Xtest, ytrain, yvalid, cats_opt, params, NUM_ROUNDS):
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
    test_pred = lgb_model.predict(Xtest)
    return y_pred, test_pred


def cat_train(Xtrain, Xvalid, Xtest, ytrain, yvalid, cats_opt, params, NUM_ROUNDS):
    # Fit model
    train_pool = Pool(data=Xtrain, label=ytrain, cat_features=cats_opt)
    valid_pool = Pool(data=Xvalid, label=yvalid, cat_features=cats_opt)
        
    cat_model = CatBoostRegressor(**params, cat_features=cats)
        
    cat_model.fit(train_pool, eval_set=valid_pool)
        
    # Eval, predict, save
    y_pred = cat_model.predict(Xvalid)
    test_pred = cat_model.predict(Xtest)
    return y_pred, test_pred

#%%
cat_params = {
            'grow_policy':'Lossguide', 
            'learning_rate':0.1, 
            'task_type':'CPU', 
            'verbose':0
}


print('Cat, KM')
df_y = df_target[['km']].copy()
oof_cat1, test_cat1 = cv_func(df_X, df_y, Xtest, df_train, kf, cat_train, cats, cat_params, 0)

print('Cat, NA')
df_y = df_target[['na_hazard']].copy()
oof_cat2, test_cat2 = cv_func(df_X, df_y, Xtest, df_train, kf, cat_train, cats, cat_params, 0)

print('Cat CoxPH')
df_y = df_target[['target4']].copy()
oof_cat3, test_cat3 = cv_func(df_X, df_y, Xtest, df_train, kf, cat_train, cats, cat_params, 0)

#%%
cat_params = {
            'grow_policy':'Lossguide', 
            'loss_function': 'Cox',
            'learning_rate': 0.1, 
            'subsample':0.8,
            'reg_lambda': 8.0,
            'task_type':'CPU', 
            'verbose': 0
}

print('Cat Surv')
df_y = df_target[['target4']].copy()
oof_cat4, test_cat4 = cv_func(df_X, df_y, Xtest, df_train, kf, cat_train, cats, cat_params, 0)

#%%
cat_params = {
            'grow_policy':'Depthwise', 
            'loss_function': 'Cox',
            'learning_rate': 0.1,
            'subsample':0.8,
            'reg_lambda': 8.0, 
            'task_type':'CPU', 
            'verbose': 0
}

print('Cat Surv')
df_y = df_target[['target4']].copy()
oof_cat5, test_cat5 = cv_func(df_X, df_y, Xtest, df_train, kf, cat_train, cats, cat_params, 0)


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


print('LGB, KM')
df_y = df_target[['km']].copy()
oof_lgb1, test_lgb1 = cv_func(df_X, df_y, Xtest, df_train, kf, lgb_train, cats, params, NUM_ROUNDS)

print('LGB, NA')
df_y = df_target[['na_hazard']].copy()
oof_lgb2, test_preds2 = cv_func(df_X, df_y, Xtest, df_train, kf, lgb_train, cats, params, NUM_ROUNDS)

print('LGB CoxPH')
df_y = df_target[['cox_ph']].copy()
oof_lgb3, test_preds3 = cv_func(df_X, df_y, Xtest, df_train, kf, lgb_train, cats, params, NUM_ROUNDS)


# %%


oof = 4*rankdata(oof_cat1) + 4*rankdata(oof_cat2) + rankdata(oof_cat3) +\
    8*rankdata(oof_lgb1) + 8*rankdata(oof_lgb2) + rankdata(oof_lgb3) +\
        6*rankdata(oof_cat4) + 6*rankdata(oof_cat5)

m = c_index(df_train, oof)
print('ensemble', m)

# %%

# with open('oof/lgb_km1.npy', 'wb') as f:
#     np.save(f, oof_lgb1)
# with open('oof/lgb_na1.npy', 'wb') as f:
#     np.save(f, oof_lgb2)
# with open('oof/lgb_cox1.npy', 'wb') as f:
#     np.save(f, oof_lgb3)
    
    
# #%%

# with open('oof/lgb_km1.npy', 'rb') as f:
#     lgb_km = np.load(f)
# with open('oof/lgb_na1.npy', 'rb') as f:
#     lgb_na = np.load(f)
# with open('oof/lgb_cox1.npy', 'rb') as f:
#     lgb_cox = np.load(f)

# oof = rankdata(lgb_km) + rankdata(lgb_na) #+ rankdata(lgb_cox)  

# m = c_index(df_train, oof)
# print('ensemble', m)
