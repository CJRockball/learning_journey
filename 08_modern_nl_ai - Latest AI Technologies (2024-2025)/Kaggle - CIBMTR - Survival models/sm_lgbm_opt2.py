#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, train_test_split
from xgboost import XGBRegressor, XGBClassifier
import xgboost
print("Using XGBoost version",xgboost.__version__)
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

# # Add descrete features
# add_list = [['prod_type', 'vent_hist'], ['melphalan_dose', 'in_vivo_tcd', 'cyto_score', 'prim_disease_hct'], ['obesity', 'hepatic_severe', 'rheum_issue'], ['mrd_hct', 'melphalan_dose', 'ethnicity'], ['ethnicity', 'mrd_hct', 'prior_tumor'], ['tce_match', 'vent_hist', 'pulm_moderate'], ['in_vivo_tcd', 'arrhythmia']]
# add_col = []

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

# df_train, df_test, add_col = make_cats_features(df_train, df_test, add_list)
# cats = cats + add_col



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

# Encode num
print('nums')
for cname in nums:
    df_train[cname] = df_train[cname].fillna(-1)
    df_test[cname] = df_test[cname].fillna(-1)


# Set up dataframes
print('set up dfs')
df_target = df_train[target].copy()
df_X = df_train.drop(columns=target).copy()
df_X[cats] = df_X[cats].astype('category')
df_test[cats] = df_test[cats].astype('category')
FEATURES = nums + cats


#%%
# Add features
print('add_features')
def add_features(df):
    add_cols = []
        # Helper function to safely divide
    def safe_divide(a, b, fill_value=0):
        return np.where(b != 0, a / b, fill_value)
    
        # Original features
    df['age_karnofsky'] = df['age_at_hct'] * df['karnofsky_score']
    df['age_comorbidity'] = df['age_at_hct'] * df['comorbidity_score']
    df['donor_recipient_age_diff'] = abs(df['donor_age'] - df['age_at_hct'])
    df['years_since_2000'] = df['year_hct'] - 2000
    df['hla_match_ratio'] = (df['hla_high_res_8'] + df['hla_low_res_8']) / 16
    df['age_squared'] = df['age_at_hct'] ** 2
    df['karnofsky_squared'] = df['karnofsky_score'] ** 2
    
    add_cols = add_cols + ['age_karnofsky','age_comorbidity','donor_recipient_age_diff',
                            'years_since_2000', 'hla_match_ratio', 'age_squared','karnofsky_squared']
    
        # HLA-related features
    df['total_hla_high_match'] = (
            df['hla_match_a_high'] + 
            df['hla_match_b_high'] + 
            df['hla_match_c_high'] + 
            df['hla_match_drb1_high'] +
            df['hla_match_dqb1_high']
        )
        
    df['total_hla_low_match'] = (
            df['hla_match_a_low'] + 
            df['hla_match_b_low'] + 
            df['hla_match_c_low'] + 
            df['hla_match_drb1_low'] +
            df['hla_match_dqb1_low']
        )
        
        # HLA match differences
    df['hla_high_low_diff_8'] = df['hla_high_res_8'] - df['hla_low_res_8']
    df['hla_high_low_diff_6'] = df['hla_high_res_6'] - df['hla_low_res_6']
    df['hla_high_low_diff_10'] = df['hla_high_res_10'] - df['hla_low_res_10']
    
    add_cols = add_cols + ['total_hla_high_match', 'total_hla_low_match', 'hla_high_low_diff_8',
                            'hla_high_low_diff_6','hla_high_low_diff_10']
    
        # Complex interactions
    df['age_hla_interaction'] = df['age_at_hct'] * df['total_hla_high_match']
    df['karnofsky_hla_interaction'] = df['karnofsky_score'] * df['total_hla_high_match']
    df['comorbidity_hla_interaction'] = df['comorbidity_score'] * df['total_hla_high_match']

        # Age-related features
    df['is_elderly'] = (df['age_at_hct'] > 65).astype(float)
    df['age_donor_ratio'] = safe_divide(df['age_at_hct'], df['donor_age'])
        
        # Age risk factor with safety checks
    karnofsky_safe = df['karnofsky_score'].clip(lower=1)
    df['age_risk_factor'] = safe_divide(df['age_at_hct'] * df['comorbidity_score'], karnofsky_safe)
    
        # Time-based features
    df['decade_of_hct'] = (df['year_hct'] // 10) * 10
    df['is_recent_hct'] = (df['year_hct'] >= 2015).astype(float)
    
        # Health status composite score
    karnofsky_norm = df['karnofsky_score'] / 100
    comorbidity_factor = 1 / (df['comorbidity_score'].clip(lower=0.1))
    age_factor = 100 / (df['age_at_hct'].clip(lower=1))
    df['health_index'] = (karnofsky_norm * comorbidity_factor * age_factor).clip(upper=100)
    
    add_cols = add_cols + ['age_hla_interaction', 'karnofsky_hla_interaction', 'comorbidity_hla_interaction',
                            'is_elderly', 'age_donor_ratio', 'age_risk_factor','decade_of_hct',
                            'is_recent_hct', 'health_index']
    
        # Polynomial features
    df['comorbidity_squared'] = df['comorbidity_score'] ** 2
    df['donor_age_squared'] = df['donor_age'] ** 2
        
        # Log transformations
    df['log_age'] = np.log1p(df['age_at_hct'] + 1)
    df['log_donor_age'] = np.log1p(df['donor_age'] + 1)
    df['log_comorbidity'] = np.log1p(df['comorbidity_score'] + 1)
    
        # Ratio features
    df['age_karnofsky_ratio'] = safe_divide(df['age_at_hct'], df['karnofsky_score'])
    df['comorbidity_karnofsky_ratio'] = safe_divide(df['comorbidity_score'], df['karnofsky_score'])
    
        # Binary flags
    df['perfect_hla_match'] = (df['hla_high_res_8'] == 8).astype(float)
    df['poor_health_status'] = ((df['karnofsky_score'] < 80) & 
                                   (df['comorbidity_score'] > 2)).astype(float)
    
    add_cols = add_cols + ['comorbidity_squared', 'donor_age_squared', 'age_karnofsky_ratio',
                           'comorbidity_karnofsky_ratio', 'perfect_hla_match', 'poor_health_status']
    
        # Clip all numeric columns to reasonable ranges
    numeric_cols = df.select_dtypes(include=['float32', 'float64']).columns
    for col in numeric_cols:
        df[col] = df[col].clip(lower=-1e6, upper=1e6)
            
    return df, add_cols

#df_train, add_cols = add_features(df_train)
#df_test, _ = add_features(df_test)
#nums = nums + add_cols


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


def calculate_na(df, time_col, event_col):
    naf = NelsonAalenFitter()
    naf.fit(durations=df[time_col], event_observed=df[event_col])
    chaz_race = - naf.cumulative_hazard_at_times(df[time_col]).values.flatten()
    return chaz_race    

def stratify_nelsonaalen(df, time_col='eft_time', event_col='efs'):
    race_group = sorted(df['race_group'].unique())
    chaz_dict = {}
    for race in race_group:
        race_df = df[df['race_group'] == race]
        chaz_dict[race] = calculate_na(race_df, time_col, event_col)
    for race in race_group:
        df.loc[df['race_group'] == race, 'na_hazard'] = chaz_dict[race]
    df.loc[df[event_col] == 0, 'na_hazard'] -= 0.12

    return df

# Calculate na by race, add to df_target 
df_train = stratify_nelsonaalen(df_train, time_col='efs_time', event_col='efs')
df_train.drop(columns='na_hazard')
df_target['na_hazard'] = df_train['na_hazard']


# # Fit CoxPH by race
# cph = CoxPHFitter(penalizer=0.05) #, l1_ratio=0.01)
# cph.fit(df_train.loc[:,FEATURES + ['efs', 'efs_time']], duration_col='efs_time', event_col='efs', strata='race_group')
# #cph.print_summary()
# # Get one value per ro for ranking
# # Predict partial hazards on the validation set
# cox_hazard = cph.predict_partial_hazard(df_X)
# df_target['cox_hazard'] = cox_hazard
# # Calculate the concordance index
# #c_index = concordance_index(df_train['efs_time'], -cox_hazard, df_train['efs'])
# #print(f"Concordance Index on Validation Set: {c_index:.4f}")


#%%

def calc_score2(df, y_pred_, check_index):
    y_true = df.loc[check_index, ["ID","efs","efs_time","race_group"]].copy()
    y_pred = df.loc[check_index, ["ID"]].copy()
    y_pred["prediction"] = y_pred_
    m = score(y_true.copy(), y_pred.copy(), "ID")
    print(f"\nOverall CV for lgb =",m)
    return m

params1 = {
    "boosting_type": "gbdt",
    "objective": "regression",
    "metric": 'mae', #'rmse', #'custom',
    "verbose": 0,
    'random_state': 42,
    'n_jobs':4,
    #"feature_pre_filter" : False,
    
    'learning_rate': 0.02,
    'num_leaves': 32,
    'min_child_weight':1,
    'subsample': 0.502742,
    'colsample_bytree': 0.373231,
    'reg_alpha':1, 
    'reg_lambda':1,
    }

FEATURES = nums + cats   
NUM_ROUNDS = 3000
KFOLD = 5
kf = KFold(n_splits=KFOLD, shuffle=True, random_state=42)
df_y = df_target[['km']].copy() #'na_hazard']].copy() #
Xtest = df_test.loc[:,FEATURES]
oof_lgb = np.zeros(len(df_train))
best_score = []
full_index = [] 
c_score = []
for i, (train_idx, valid_idx) in enumerate(kf.split(df_X)):
    print(25*"#")
    print(f"FOLD {i}\n")
    
    # Set up data
    Xtrain = df_X.loc[train_idx, FEATURES].copy()
    ytrain = df_y.loc[train_idx].copy()
    Xvalid = df_X.loc[valid_idx, FEATURES].copy()
    yvalid = df_y.loc[valid_idx].copy()
    full_index = full_index + list(valid_idx)
    
    # Make model, train
    trn_data = lgb.Dataset(Xtrain, ytrain, categorical_feature=cats)
    lgb_eval = [lgb.Dataset(Xvalid, yvalid)]
    
    lgb_model = lgb.train(params1, trn_data, NUM_ROUNDS,
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
    
    val_list = list(valid_idx)
    val_list.sort 
    m = calc_score2(df_train, y_pred, val_list)
    c_score.append(m)
    
    if i>0:
        preds += lgb_model.predict(Xtest)
    else:
        preds = lgb_model.predict(Xtest)
    
    
preds /= KFOLD   
print(f"Mean Score {np.mean(best_score)}")
full_index.sort()


#GET SCORE
m = calc_score2(df_train, oof_lgb, full_index)
print(m)
print(c_score)

# 
# %% -------------------------------------------------------------------------

def calc_score(df, y_pred_, check_index):
    y_true = df.loc[check_index, ["ID","efs","efs_time","race_group"]].copy()
    y_pred = df.loc[check_index, ["ID"]].copy()
    y_pred["prediction"] = y_pred_
    m = score(y_true, y_pred, "ID")
    #print(f"\nOverall CV for lgb =",m)
    return m

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

FEATURES = nums + cats   
NUM_ROUNDS = 3000
KFOLD = 5
kf = KFold(n_splits=KFOLD, shuffle=True, random_state=42)
df_y = df_target[['km']].copy()

def lgb_pred(df_X, df_y, df_train):
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
        
    # Make model, train
        trn_data = lgb.Dataset(Xtrain, ytrain, categorical_feature=cats)
        lgb_eval = [lgb.Dataset(Xvalid, yvalid)]
    
        lgb_model = lgb.train(params, trn_data, NUM_ROUNDS,
                      valid_sets=lgb_eval, #feval=rmsle_lgbm, 
                      callbacks = [
                          #lgb.log_evaluation(1000),
                          lgb.early_stopping(stopping_rounds=25, verbose=0)
                      ])
    
    # Eval, predict, save
        #print('\nBest Fit: ', lgb_model.best_score['valid_0']['l1'], '\n')

        y_pred = lgb_model.predict(Xvalid)
        oof_lgb[valid_idx] = y_pred
        
        val_list = list(valid_idx)
        val_list.sort 
        c_index_score = calc_score(df_train, y_pred, val_list)
        c_list.append(c_index_score)

    full_index.sort()
    
    full_score = calc_score(df_train, oof_lgb, full_index)
    return c_list, full_score


# %%
import time 

df_X2 = df_X.loc[:, FEATURES].copy()
c_list, full_score = lgb_pred(df_X2, df_y, df_train)
best_score = full_score
mean_score = np.mean(c_list)
print(f'oof score {best_score}, Mean score: {mean_score}')


# def add_cat_feature(df, new_feature):
#     fname = "_".join(new_feature)
#     # Make new feature
#     df[fname] = df[new_feature[0]].astype(str) + df[new_feature[1]].astype(str)
    
#     # Ordinal encode new features
#     train_categories = df[fname].value_counts().sort_values().index.get_level_values(0).to_list()
#     train_dict = {cname:i for i, cname in enumerate(train_categories)}    
#     df[fname] = df[fname].map(train_dict).astype(np.int16)
#     return df


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
for k in range(10):
    if k % 25 == 0:
        print(f"Number of tests run: {k}, Number of features: {df_X2.shape[1]}")
    ct = np.random.choice([2, 3, 4], 1)[0]
    new_feature = list( np.random.choice(cats, ct, replace=False))
    df_X2 = add_cat_feature(df_X2, new_feature)
    
    c_list, new_score = lgb_pred(df_X2, df_y, df_train)
    mean_score = np.mean(c_list)
    
    if new_score > best_score:
        MORE.append(new_feature)
        best_score = new_score
        print(f'{new_feature}, oof score {new_score:.5f}, mean score {mean_score:.5f}')
        
    else:
        fname = "_".join(new_feature)
        df_X2 = df_X2.drop(columns=[fname])

end_time = time.time()
print(f'Total time: {end_time - start_time}')
print("DONE")

#%%

print(df_X2.shape)

print(MORE)
#%%
'''
score 0.67803
[['prod_type', 'vent_hist'], ['melphalan_dose', 'in_vivo_tcd', 'cyto_score', 'prim_disease_hct'], ['obesity', 'hepatic_severe', 'rheum_issue'], ['mrd_hct', 'melphalan_dose', 'ethnicity'], ['ethnicity', 'mrd_hct', 'prior_tumor'], ['tce_match', 'vent_hist', 'pulm_moderate'], ['in_vivo_tcd', 'arrhythmia']]
'''