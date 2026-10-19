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
add_list = [['prior_tumor', 'graft_type', 'peptic_ulcer'], ['melphalan_dose', 'donor_related'], ['in_vivo_tcd', 'peptic_ulcer', 'mrd_hct'], ['diabetes', 'in_vivo_tcd']]
add_col = []

def make_cats_features(df_train, df_test, add_list):
    for lst in add_list:
        fname = "_".join(lst)
        if len(lst) == 2:
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


# %% -------------------------------------------------------------------------

def calc_score(df, y_pred_, check_index):
    y_true = df.loc[check_index, ["ID","efs","efs_time","race_group"]].copy()
    y_pred = df.loc[check_index, ["ID"]].copy()
    y_pred["prediction"] = y_pred_
    m = score(y_true, y_pred, "ID")
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

def lgb_pred(df_X, df_y, df_train, cats_opt):
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
        trn_data = lgb.Dataset(Xtrain, ytrain, categorical_feature=cats_opt)
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


#%% Standard Run

c_list, new_score = lgb_pred(df_X.loc[:, FEATURES], df_y, df_train, cats)
print(new_score, '\n', c_list)


# %% Optimization, check random cross features

df_X2 = df_X.loc[:, FEATURES].copy()
c_list, full_score = lgb_pred(df_X2, df_y, df_train, cats)
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

MORE = []
start_time = time.time()
cats_opt = cats
for k in range(400):
    if k % 25 == 0:
        print(f"Number of tests run: {k}, Number of features: {df_X2.shape[1]}")
        
    ct = np.random.choice([2, 3, 4], 1)[0]
    new_feature = list( np.random.choice(cats, ct, replace=False))
    fname = "_".join(new_feature)
    df_X2 = add_cat_feature(df_X2, new_feature)
    cats_opt = cats_opt + [fname]
    
    c_list, new_score = lgb_pred(df_X2, df_y, df_train, cats_opt)
    mean_score = np.mean(c_list)
    
    if new_score > best_score:
        MORE.append(new_feature)
        best_score = new_score
        print(f'{new_feature}, oof score {new_score:.5f}, mean score {mean_score:.5f}')   
    else:
        df_X2 = df_X2.drop(columns=[fname])
        cats_opt.remove(fname)

end_time = time.time()
print(f'Total time {end_time-start_time}')      
print("DONE")

#%%

c_list, full_score = lgb_pred(df_X2, df_y, df_train, cats_opt)
mean_score = np.mean(c_list)
print(f'oof score {full_score}, Mean score: {mean_score}')

print(MORE)

#%%

df_X2.to_parquet('m_save/df_X2_0.67448.csv', index=False)
#print(df_X2.shape)
df_check = pd.read_parquet('m_save/df_X2_0.67448.csv')
print(df_check.shape)

c_list, full_score = lgb_pred(df_check, df_y, df_train, cats)
mean_score = np.mean(c_list)
print(f'oof score {full_score}, Mean score: {mean_score}')

#%%
print(MORE)

#%%
'''
['mrd_hct', 'hepatic_mild', 'ethnicity', 'donor_related'] 0.6768258195594198 0.6742049437240524


['hepatic_mild', 'ethnicity'], oof score 0.67674, mean score 0.67416
['renal_issue', 'obesity', 'cmv_status'], oof score 0.67708, mean score 0.67455
[['hepatic_mild', 'ethnicity'], ['renal_issue', 'obesity', 'cmv_status']]

['rituximab', 'obesity'], oof score 0.67672, mean score 0.67441
['conditioning_intensity', 'graft_type'], oof score 0.67673, mean score 0.67423
['peptic_ulcer', 'tce_match'], oof score 0.67699, mean score 0.67449
['peptic_ulcer', 'in_vivo_tcd'], oof score 0.67714, mean score 0.67431
['vent_hist', 'tce_match'], oof score 0.67716, mean score 0.67417
Number of tests run: 25, Number of features: 62
['donor_related', 'in_vivo_tcd'], oof score 0.67747, mean score 0.67486
[['rituximab', 'obesity'], ['conditioning_intensity', 'graft_type'], ['peptic_ulcer', 'tce_match'], ['peptic_ulcer', 'in_vivo_tcd'], ['vent_hist', 'tce_match'], ['donor_related', 'in_vivo_tcd']]

['prior_tumor', 'graft_type', 'peptic_ulcer'], oof score 0.67711, mean score 0.67469
['melphalan_dose', 'donor_related'], oof score 0.67724, mean score 0.67492
['in_vivo_tcd', 'peptic_ulcer', 'mrd_hct'], oof score 0.67751, mean score 0.67539
['diabetes', 'in_vivo_tcd'], oof score 0.67769, mean score 0.67500
[['prior_tumor', 'graft_type', 'peptic_ulcer'], ['melphalan_dose', 'donor_related'], ['in_vivo_tcd', 'peptic_ulcer', 'mrd_hct'], ['diabetes', 'in_vivo_tcd']]

['pulm_severe', 'peptic_ulcer', 'arrhythmia', 'prior_tumor'], oof score 0.67659, mean score 0.67384
['in_vivo_tcd', 'rituximab', 'prior_tumor'], oof score 0.67687, mean score 0.67451
['peptic_ulcer', 'arrhythmia'], oof score 0.67723, mean score 0.67475
['graft_type', 'mrd_hct'], oof score 0.67745, mean score 0.67471
['prod_type', 'vent_hist'], oof score 0.67751, mean score 0.67483
[['pulm_severe', 'peptic_ulcer', 'arrhythmia', 'prior_tumor'], ['in_vivo_tcd', 'rituximab', 'prior_tumor'], ['peptic_ulcer', 'arrhythmia'], ['graft_type', 'mrd_hct'], ['prod_type', 'vent_hist']]



'''