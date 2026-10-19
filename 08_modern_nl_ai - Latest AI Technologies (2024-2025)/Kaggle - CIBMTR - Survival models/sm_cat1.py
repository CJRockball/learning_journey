#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, train_test_split
from catboost import CatBoostRegressor, Pool
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

# Feature Engineering
for cname in nums:
    df_train[cname] = df_train[cname].fillna(-1)
    df_test[cname] = df_test[cname].fillna(-1)

# Set up dfs
df_target = df_train[target].copy()
df_X = df_train.drop(columns=target)
df_X[cats] = df_X[cats].astype('category')
df_test[cats] = df_test[cats].astype('category')
FEATURES = nums + cats 

#%% ------------------------- CHOOSE SURVIVAL MODEL --------------------- 

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

def make_params():
    params={
    'iterations': 5000,
    'depth': 5, #5, #
    'learning_rate': 0.1, #0.022, #
    #'l2_leaf_reg': 0.7357305158548999, 
    #'border_count': 246, 
    #'bagging_temperature': 0.6896908609682827,
    #'eta': 0.28901888228959255, 
    #'reg_lambda': 41.0642500499563, 
    #'colsample_bylevel': 0.6,
    #'subsample': 0.8,
    'loss_function': 'MAE',
    'eval_metric': 'MAE',
    'cat_features': cats, 
    'random_state': 42,
    #'min_data_in_leaf': 51,
    'early_stopping_rounds': 50,
    #'max_bin': 5000, # Can try to blow out max_bins. Didn't work here
    'task_type': "GPU",
    'verbose':100,
    #'random_strength': 1.5,
    #'bootstrap_type': 'Bernoulli',
    }
    return params

# C-index 0.674
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
    cat_model = CatBoostRegressor(grow_policy='Lossguide',task_type='GPU',
                                  verbose=0) #**cat_params,)
        
    cat_model.fit(train_pool, eval_set=valid_pool)
        
    # Eval, predict, save
    y_pred = cat_model.predict(Xvalid)
    mae_ = mean_absolute_error(yvalid, y_pred)
    
    print('\nBest Fit: ', mae_, '\n')
    best_score.append(mae_)
    oof_cat[valid_idx] = y_pred
    
    if i>0:
        preds += cat_model.predict(Xtest)
    else:
        preds = cat_model.predict(Xtest)
    
preds /= KFOLD   
print(f"Mean Score {np.mean(best_score)}")

# Get c-index
y_true = df_train[["ID","efs","efs_time","race_group"]].copy()
y_pred = df_train[["ID"]].copy()
y_pred["prediction"] = oof_cat
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"\nOverall C-index for Catboost =",m)

# %% # ------------------------------  Nelson Aalen model

naf = NelsonAalenFitter()
naf.fit(durations=df_train['efs_time'], event_observed=df_train['efs'])
df_target['na_hazard'] = naf.cumulative_hazard_at_times(df_train['efs_time']).values
df_target['na_hazard'] = df_target['na_hazard'] * -1
df_target.loc[df_train['efs'] == 0, 'na_hazard'] *= -1




df_y = df_target[['na_hazard']].copy() 
Xtest = df_test.loc[:,FEATURES]
oof_cat2 = np.zeros(len(df_train))
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
    cat_model = CatBoostRegressor(grow_policy='Lossguide',task_type='GPU',
                                  learning_rate=0.1, verbose=0) #**cat_params,)
        
    cat_model.fit(train_pool, eval_set=valid_pool)
        
    # Eval, predict, save
    y_pred = cat_model.predict(Xvalid)
    mae_ = mean_absolute_error(yvalid, y_pred)
    
    print('\nBest Fit: ', mae_, '\n')
    best_score.append(mae_)
    oof_cat2[valid_idx] = y_pred
    
    if i>0:
        preds += cat_model.predict(Xtest)
    else:
        preds = cat_model.predict(Xtest)
    
preds /= KFOLD   
print(f"Mean Score {np.mean(best_score)}")

# Get c-index
y_true = df_train[["ID","efs","efs_time","race_group"]].copy()
y_pred = df_train[["ID"]].copy()
y_pred["prediction"] = oof_cat2 
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"\nOverall C-index for Catboost =",m)

#%% Add cox metric on catboost

df_target['target4'] = df_train['efs_time'].copy()
df_target.loc[df_train['efs'] == 0, 'target4'] *= -1

print(df_target['target4'].isnull())


#%%
# Cox model params
cox1_params = {
        'grow_policy': 'Depthwise',
        'min_child_samples': 8,
        'loss_function': 'Cox',
        'learning_rate': 0.03,
        'random_state': 42,
        'task_type': 'CPU',
        'num_trees': 6000,
        'subsample': 0.6,  
        'reg_lambda': 8.0,
        'depth': 8,
    }


df_y = df_target[['target4']].copy() 
Xtest = df_test.loc[:,FEATURES]
oof_cat3 = np.zeros(len(df_train))
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
        
    cat_model = CatBoostRegressor(grow_policy='Lossguide', loss_function= 'Cox',
                                  learning_rate=0.1, task_type='CPU', 
                                  verbose=0) #**cat_params,)
        
    cat_model.fit(train_pool, eval_set=valid_pool)
        
    # Eval, predict, save
    y_pred = cat_model.predict(Xvalid)
    mae_ = mean_absolute_error(yvalid, y_pred)
    
    print('\nBest Fit: ', mae_, '\n')
    best_score.append(mae_)
    oof_cat3[valid_idx] = y_pred
    
    if i>0:
        preds += cat_model.predict(Xtest)
    else:
        preds = cat_model.predict(Xtest)
    
preds /= KFOLD   
print(f"Mean Score {np.mean(best_score)}")

# Get c-index
y_true = df_train[["ID","efs","efs_time","race_group"]].copy()
y_pred = df_train[["ID"]].copy()
y_pred["prediction"] = oof_cat3 # Use -00f_xgb for Chris' solution
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"\nOverall C-index for Catboost =",m)



#%%


oof = rankdata(oof_cat) + rankdata(oof_cat2) + rankdata(oof_cat3) 

#GET SCORE
y_true = df_train[["ID","efs","efs_time","race_group"]].copy()
y_pred = df_train[["ID"]].copy()
y_pred["prediction"] = oof
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"\nEnsemble oof =",m)



# %%

with open('oof/cat_km.npy', 'wb') as f:
    np.save(f, oof_cat)
with open('oof/cat_na.npy', 'wb') as f:
    np.save(f, oof_cat2)
with open('oof/cat_cox.npy', 'wb') as f:
    np.save(f, oof_cat3)
    
    
#%%

with open('oof/cat_km.npy', 'rb') as f:
    cat_km = np.load(f)
with open('oof/cat_na.npy', 'rb') as f:
    cat_na = np.load(f)
with open('oof/cat_cox.npy', 'rb') as f:
    cat_cox = np.load(f)

oof = rankdata(cat_km) + rankdata(cat_na) + rankdata(cat_cox)  

#GET SCORE
y_true = df_train[["ID","efs","efs_time","race_group"]].copy()
y_pred = df_train[["ID"]].copy()
y_pred["prediction"] = oof
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"\nEnsemble oof =",m)

# %%
