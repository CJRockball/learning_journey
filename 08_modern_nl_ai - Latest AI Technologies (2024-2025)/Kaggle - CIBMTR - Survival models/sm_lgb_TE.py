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
from sklearn.preprocessing import TargetEncoder
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


#Define categories
target =['efs', 'efs_time']
# nums = ['donor_age', 'age_at_hct'] 
# cats = [name for name in df_train.columns if name not in (nums + target + ['ID'])]

cats = df_train.select_dtypes(include=["object_", 'bool']).columns
cats = [cat for cat in cats if cat not in target+['ID']]
nums = df_train.select_dtypes(include=['number']).columns
nums = [num for num in nums if num not in target+['ID']]

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

for cname in nums:
    df_train[cname] = df_train[cname].fillna(-1)
    df_test[cname] = df_test[cname].fillna(-1)


df_target = df_train[target].copy()
FEATURES = nums + cats 
#df_X = df_train.iloc[:,FEATURES] #.drop(columns=target)
df_train[cats] = df_train[cats].astype('category')
df_test[cats] = df_test[cats].astype('category')


#%%

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

naf = NelsonAalenFitter()
naf.fit(durations=df_train['efs_time'], event_observed=df_train['efs'])
df_target['na_hazard'] = naf.cumulative_hazard_at_times(df_train['efs_time']).values
df_target['na_hazard'] = df_target['na_hazard'] * -1

#%% Target encoding

add_col = []
for cname in cats:
    ne_name = cname + "_TE"
    enc_auto = TargetEncoder(smooth="auto", random_state=42)        
    df_train[ne_name] = enc_auto.fit_transform(df_train[[str(cname)]], df_target['km'].values.ravel())
    df_test[ne_name] = enc_auto.transform(df_test[[str(cname)]])
    add_col.append(ne_name)

nums = nums + add_col
FEATURES = nums + cats

df_X = df_train.loc[:,FEATURES] #.drop(columns=target)
df_X[cats] = df_X[cats].astype('category')
df_test[cats] = df_test[cats].astype('category')

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


FEATURES = nums + cats   
NUM_ROUNDS = 3000
KFOLD = 5
kf = KFold(n_splits=KFOLD, shuffle=True, random_state=42)
df_y = df_target[['km']].copy() #'na_hazard']].copy() #
Xtest = df_test.loc[:,FEATURES]
oof_lgb = np.zeros(len(df_train))
pred_lgb = np.zeros(len(df_test))
best_score = []
    
for i, (train_idx, valid_idx) in enumerate(kf.split(df_X)):
    print(25*"#")
    print(f"FOLD {i}")
    
    # Set up data
    Xtrain = df_X.loc[train_idx, FEATURES].copy()
    ytrain = df_y.loc[train_idx].copy()
    Xvalid = df_X.loc[valid_idx, FEATURES].copy()
    yvalid = df_y.loc[valid_idx].copy()
    
    # Make model, train
    trn_data = lgb.Dataset(Xtrain, ytrain, categorical_feature=cats)
    lgb_eval = [lgb.Dataset(Xvalid, yvalid)]
    
    lgb_model = lgb.train(params, trn_data, NUM_ROUNDS,
                      valid_sets=lgb_eval,
                      callbacks = [
                          #lgb.log_evaluation(1000),
                          lgb.early_stopping(stopping_rounds=25, verbose=0)
                      ])
    
    # Eval, predict, save
    print('Best Fit: ', lgb_model.best_score['valid_0']['l1'], '\n')
    best_score.append(lgb_model.best_score['valid_0']['l1'])
        
    
    
    y_pred = lgb_model.predict(Xvalid)
    oof_lgb[valid_idx] = y_pred
    
    if i>0:
        preds += lgb_model.predict(Xtest)
    else:
        preds = lgb_model.predict(Xtest)
    
preds /= KFOLD   
print(f"Mean Score {np.mean(best_score)}")


m = c_index(df_train, oof_lgb)
print(m)


# %%

display(df_target)


# %%

plt.figure()
df_target.loc[df_target['efs'] < 1, 'na_hazard'].plot(kind='hist', bins=40, label='0')
#df_target.loc[df_target['efs'] > 0,'na_hazard'].plot(kind='hist', bins=40, label='1')
plt.show()


# %%
