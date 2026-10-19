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
# nums = ['donor_age', 'age_at_hct'] 
# cats = [name for name in df_train.columns if name not in (nums + target + ['ID'])]

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

# Feature Engineering
for cname in nums:
    df_train[cname] = df_train[cname].fillna(-1)
    df_test[cname] = df_test[cname].fillna(-1)


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
    df['log_age'] = np.log1p(df['age_at_hct'])
    df['log_donor_age'] = np.log1p(df['donor_age'])
    df['log_comorbidity'] = np.log1p(df['comorbidity_score'])
    
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

df_train, add_cols = add_features(df_train)
df_test, _ = add_features(df_test)
#nums = nums + add_cols


#
df_target = df_train[target].copy()
df_X = df_train.drop(columns=target)
df_X[cats] = df_X[cats].astype('category')
df_test[cats] = df_test[cats].astype('category')
FEATURES = nums + cats 

#%% Kaplan-Meier by race

# def transform_survival_probability(df, time_col='efs_time', event_col='efs'):
#     kmf = KaplanMeierFitter()
#     kmf.fit(df[time_col], event_observed=df[event_col])
#     survival_probabilities = kmf.survival_function_at_times(df[time_col]).values.flatten()
#     return survival_probabilities

# def update_target_with_survival_probabilities(df, time_col='efs_time', event_col='efs'):
#     race_group = sorted(df['race_group'].unique())
#     survival_probs_dict = {}
#     for race in race_group:
#         race_df = df[df['race_group'] == race]
#         survival_probs_dict[race] = transform_survival_probability(race_df, time_col, event_col)
#     for race in race_group:
#         df.loc[df['race_group'] == race, 'km'] = survival_probs_dict[race]
#     df.loc[df[event_col] == 0, 'km'] -= 0.18
    
#     return df

# # # will make "target" the label 
# df_train = update_target_with_survival_probabilities(df_train, time_col='efs_time', event_col='efs')
# df_train.drop(columns='km')
# df_target['km'] = df_train['km']


# naf = NelsonAalenFitter()
# naf.fit(durations=df_train['efs_time'], event_observed=df_train['efs'])
# df_target['na_hazard'] = naf.cumulative_hazard_at_times(df_train['efs_time']).values
# df_target['na_hazard'] = df_target['na_hazard'] * -1




KFOLD = 5
oof_pred = np.zeros(len(df_target))
cv = KFold(n_splits=KFOLD, shuffle=True, random_state=42)
for train_index, valid_index in cv.split(df_target):
    train_data = df_target.iloc[train_index]
    valid_data = df_target.iloc[valid_index]
    
    kmf = KaplanMeierFitter()
    kmf.fit(durations=train_data['efs_time'], event_observed=train_data['efs'])
    oof_pred[valid_index] = kmf.survival_function_at_times(valid_data['efs_time']).values

df_target['km'] = oof_pred
m = c_index(df_train, df_target['km'])
print('km', m)

oof_pred = np.zeros(len(df_target))
for train_index, valid_index in cv.split(df_target):

    train_data = df_target.iloc[train_index]
    valid_data = df_target.iloc[valid_index]
            
    # Make NA
    naf = NelsonAalenFitter()
    naf.fit(durations=train_data['efs_time'], event_observed=train_data['efs'])
    oof_pred[valid_index] = - naf.cumulative_hazard_at_times(valid_data['efs_time']).values

df_target['na_hazard'] = oof_pred
m = c_index(df_train, df_target['na_hazard'])
print('na', m)



# Try fitting the function and plotting
oof_pred = np.zeros(len(df_target))
for train_index, valid_index in cv.split(df_target):

    train_data = df_train.iloc[train_index]
    valid_data = df_train.iloc[valid_index]
            

    cph = CoxPHFitter(penalizer=0.01) #, l1_ratio=0.01)
    cph.fit(train_data.loc[:,FEATURES + ['efs', 'efs_time']], duration_col='efs_time', event_col='efs', strata='race_group')
    #cph.print_summary()
    oof_pred[valid_index] = cph.predict_partial_hazard(valid_data.loc[:,FEATURES + ['efs', 'efs_time']])

df_target['cox_ph'] = oof_pred
m = c_index(df_train, df_target['cox_ph'])
print('cox_ph', m)


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



#%% Fitting Kaplan-Meier
print('KM')

FEATURES = nums + cats   
NUM_ROUNDS = 3000
KFOLD = 5
kf = KFold(n_splits=KFOLD, shuffle=True, random_state=42)

df_y = df_target[['km']].copy() #'na_hazard']].copy() #'cox_hazard']].copy() #
Xtest = df_test.loc[:,FEATURES]

oof_lgb1 = np.zeros(len(df_train))
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
    
    lgb_model = lgb.train(params, trn_data, NUM_ROUNDS,
                      valid_sets=lgb_eval, #feval=rmsle_lgbm, 
                      callbacks = [
                          lgb.log_evaluation(1000),
                          lgb.early_stopping(stopping_rounds=25)
                      ])
    
    # Eval, predict, save
    print('\nBest Fit: ', lgb_model.best_score['valid_0']['l1'], '\n')
    best_score.append(lgb_model.best_score['valid_0']['l1'])
        
    
    
    y_pred = lgb_model.predict(Xvalid)
    oof_lgb1[valid_idx] = y_pred
    
    if i>0:
        preds += lgb_model.predict(Xtest)
    else:
        preds = lgb_model.predict(Xtest)
    
preds /= KFOLD   
print(f"Mean Score {np.mean(best_score)}")
#GET SCORE
y_true = df_train[["ID","efs","efs_time","race_group"]].copy()
y_pred = df_train[["ID"]].copy()
y_pred["prediction"] = oof_lgb1
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"\nOverall CV for XGBoost =",m)

#%%

# df_sub = pd.read_csv('data/ra/sample_submission.csv')
# df_sub['prediction'] = preds
# df_sub.to_csv('submissions/xgb1.csv', index=False)

# df_check = pd.read_csv('submissions/xgb1.csv')
# display(df_check)
 
# %% # Nelson Aalen model
print('NelsonAalen')

FEATURES = nums + cats   
NUM_ROUNDS = 3000
KFOLD = 5
kf = KFold(n_splits=KFOLD, shuffle=True, random_state=42)
df_y = df_target[['na_hazard']].copy()
Xtest = df_test.loc[:,FEATURES]
oof_lgb2 = np.zeros(len(df_train))
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
    oof_lgb2[valid_idx] = y_pred
    
    if i>0:
        preds += lgb_model.predict(Xtest)
    else:
        preds = lgb_model.predict(Xtest)
    
preds /= KFOLD   
print(f"Mean Score {np.mean(best_score)}")
#GET SCORE
y_true = df_train[["ID","efs","efs_time","race_group"]].copy()
y_pred = df_train[["ID"]].copy()
y_pred["prediction"] = oof_lgb2
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"\nOverall CV for XGBoost =",m)

#%% Out of fold cox ph
print('Cox PH')

FEATURES = nums + cats   
NUM_ROUNDS = 3000
KFOLD = 5
kf = KFold(n_splits=KFOLD, shuffle=True, random_state=42)
df_y = df_target[['cox_ph']].copy()
Xtest = df_test.loc[:,FEATURES]
oof_lgb3 = np.zeros(len(df_train))
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
    oof_lgb3[valid_idx] = y_pred
    
    if i>0:
        preds += lgb_model.predict(Xtest)
    else:
        preds = lgb_model.predict(Xtest)
    
preds /= KFOLD   
print(f"Mean Score {np.mean(best_score)}")
#GET SCORE
y_true = df_train[["ID","efs","efs_time","race_group"]].copy()
y_pred = df_train[["ID"]].copy()
y_pred["prediction"] = oof_lgb3
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"\nOverall CV for XGBoost =",m)

# %%


oof = rankdata(oof_lgb2) + rankdata(oof_lgb1) #+  rankdata(oof_lgb3) 

#GET SCORE
y_true = df_train[["ID","efs","efs_time","race_group"]].copy()
y_pred = df_train[["ID"]].copy()
y_pred["prediction"] = oof
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"\nEnsemble oof =",m)

#%%
# %%

with open('oof/lgb_km.npy', 'wb') as f:
    np.save(f, oof_lgb1)
with open('oof/lgb_na.npy', 'wb') as f:
    np.save(f, oof_lgb2)
with open('oof/lgb_cox.npy', 'wb') as f:
    np.save(f, oof_lgb3)
    
    
#%%

with open('oof/lgb_km.npy', 'rb') as f:
    lgb_km = np.load(f)
with open('oof/lgb_na.npy', 'rb') as f:
    lgb_na = np.load(f)
with open('oof/lgb_cox.npy', 'rb') as f:
    lgb_cox = np.load(f)

oof = rankdata(lgb_km) + rankdata(lgb_na) #+ rankdata(lgb_cox)  

#GET SCORE
y_true = df_train[["ID","efs","efs_time","race_group"]].copy()
y_pred = df_train[["ID"]].copy()
y_pred["prediction"] = oof
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"\nEnsemble oof =",m)



# %%
