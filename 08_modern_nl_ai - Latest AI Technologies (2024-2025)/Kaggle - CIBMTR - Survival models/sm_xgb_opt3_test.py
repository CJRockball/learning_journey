#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import time
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, train_test_split
from catboost import CatBoostRegressor, Pool
import lightgbm as lgb
import xgboost

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
# nums = ['donor_age', 'age_at_hct'] #, 'efs_time']
# cats = [name for name in df_train.columns if name not in (nums + target)]

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
df_X = df_train.loc[:, FEATURES]  #.drop(columns=target)
df_X[cats] = df_X[cats].astype('category')
df_test[cats] = df_test[cats].astype('category')

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
KFOLD = 5
kf = KFold(n_splits=KFOLD, shuffle=True, random_state=42)

#%%
   
def cv_func(df_X, df_y, df_train, cats_opt, kf, train_fcn, params, NUM_ROUNDS):
    oof_lgb = np.zeros(len(df_train))
    c_list = []

    for i, (train_idx, valid_idx) in enumerate(kf.split(df_X)):
    # Set up data
        Xtrain = df_X.loc[train_idx, FEATURES].copy()
        ytrain = df_y.loc[train_idx].copy()
        Xvalid = df_X.loc[valid_idx, FEATURES].copy()
        yvalid = df_y.loc[valid_idx].copy()
    
        # Fit model
        y_pred = train_fcn(Xtrain, Xvalid, ytrain, yvalid, cats_opt, params, NUM_ROUNDS)

        # Eval, predict, save
        oof_lgb[valid_idx] = y_pred
        c2 = get_cindex(df_train.iloc[valid_idx], y_pred)
        c_list.append(c2)

    full_score = get_cindex(df_train, oof_lgb)
    return c_list, full_score



def xgb_train(Xtrain, Xvalid, ytrain, yvalid, cats_opt, params, NUM_ROUNDS):

    # Create DMatrix - is optimized for both memory efficiency and training speed.
    train_matrix = xgboost.DMatrix(data = Xtrain, label = ytrain, enable_categorical=True)
    valid_matrix = xgboost.DMatrix(data = Xvalid, label = yvalid, enable_categorical=True)
    evals = [(train_matrix, "train"), (valid_matrix, "validation")]
    # Create & Train the model
    model = xgboost.train(params=params, dtrain = train_matrix, evals=evals,
                              num_boost_round=2000,
                             early_stopping_rounds= 25, verbose_eval=0)

    # Make prediction
    y_pred = model.predict(xgboost.DMatrix(Xvalid, enable_categorical=True))
    
    return y_pred

#%%

xgb_params = {'tree_method': 'hist', #'gpu_hist', #
              #'device': 'cuda',
                'objective': 'reg:squarederror', #'reg:absoluteerror', #'reg:logistic', #  'reg:squaredlogerror', #
                'random_state': 42,
                'eval_metric': "rmse",
                # Using Fisher grouping of categorical features. None=One hot
                "max_cat_to_onehot" : None,
                
                "max_depth":3,  
                "colsample_bytree":0.5, 
                "subsample":0.8, 
                "learning_rate":0.2, 
                "min_child_weight":5
                }

NUM_ROUNDS = 2000
df_y = df_target[['km']].copy()

start_time = time.time()
c_list, full_score = cv_func(df_X.loc[:, FEATURES], df_y, df_train, cats, kf, xgb_train, xgb_params, NUM_ROUNDS)
print(full_score)
end_time = time.time()
print(f'Total time {end_time - start_time}')


# %% Check for ne features
df_X2 = df_X.loc[:, FEATURES].copy()
start_time1 = time.time()
c_list, full_score = cv_func(df_X2, df_y, df_train, cats, kf, xgb_train, xgb_params, NUM_ROUNDS)
end_time1 = time.time()
print(f'Time {end_time1 - start_time1}')
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
    df[fname] = df[fname].map(train_dict).astype(np.int16)
    
    return df

start_time = time.time()
MORE = []
for k in range(300):
    if k % 25 == 0:
        print(f"Number of tests run: {k}, Number of features: {df_X2.shape[1]}")
    ct = np.random.choice([2, 3, 4], 1)[0]
    new_feature = list( np.random.choice(cats, ct, replace=False))
    df_X2 = add_cat_feature(df_X2, new_feature)
    
    c_list, new_score = cv_func(df_X2, df_y, df_train, cats, kf, xgb_train, xgb_params, NUM_ROUNDS)

    mean_score = np.mean(c_list)
    
    if new_score > best_score:
        MORE.append(new_feature)
        imp_diff = ((new_score - best_score) / best_score) * 100
        best_score = new_score
        print(f'{new_feature},\n improvement {imp_diff:.5f} oof score {new_score:.5f}, mean score {mean_score:.5f}')
        
    else:
        fname = "_".join(new_feature)
        df_X2 = df_X2.drop(columns=[fname])

end_time = time.time()
print(f'Total time: {end_time - start_time}')
print("DONE")

# %%
# TODO
# 1 rerun df_X2 to check
# 2 add ne features and rerun df_train to check

c_list, new_score = cv_func(df_X2, df_y, df_train, cats, kf, xgb_train, xgb_params, NUM_ROUNDS)
print(f'Full core: {new_score}')

#%% Add feature to org df

df_X3 = df_train.copy(deep=True)

# for cname in cats:
#     df_train[cname] = df_train[cname].fillna('Missing')
#     df_test[cname] = df_test[cname].fillna('Missing')

# Add descrete features
add_list = [['mrd_hct', 'ethnicity'], ['hepatic_mild', 'pulm_severe'], ['pulm_moderate', 'pulm_severe']]
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

df_X3, df_test, add_col = make_cats_features(df_X3, df_test, add_list)
cats2 = cats + add_col

df_X3, df_test = ord_trans(df_X3, df_test, add_col)
df_X3[add_col] = df_X3[add_col].astype('category')

c_list, new_score = cv_func(df_X3, df_y, df_train, cats2, kf, xgb_train, xgb_params, NUM_ROUNDS)
print(f'Full core: {new_score}')




# %%
