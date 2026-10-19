#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, train_test_split
from xgboost import XGBRegressor, XGBClassifier
import xgboost
print("Using XGBoost version",xgboost.__version__)
#from catboost import CatBoostRegressor, Pool

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
# cats = [name for name in df_train.columns if name not in (nums + target+["ID"])]
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

#Num features
for cname in nums:
    df_train[cname] = df_train[cname].fillna(-1)
    df_test[cname] = df_test[cname].fillna(-1)


# Set up dfs
df_target = df_train[target].copy()
df_target['efs_time2'] = df_target['efs_time']
df_target.loc[df_target['efs'] == 0, 'efs_time2'] *= -1

df_X = df_train.drop(columns=target)
df_X[cats] = df_X[cats].astype('category')
FEATURES = nums + cats  

#%%  -------------------- Choose a survival model -----------

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


#%% Fitting Kaplan-Meier
print('KM')
params = {'tree_method': 'hist',
                'n_estimators': 5000,
                'objective': 'reg:squarederror', 
                'random_state': 42,
                'enable_categorical': True,
                #'verbosity': 0,
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
df_y = df_target[['km']].copy() #'na_hazard']].copy() #'cox_hazard']].copy() #
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
        preds += xgb_model.predict(df_test.loc[:,FEATURES])
    else:
        preds = xgb_model.predict(df_test.loc[:,FEATURES])
    
preds /= KFOLD   
print(f"Mean Score {np.mean(best_score)}")

y_true = df_train[["ID","efs","efs_time","race_group"]].copy()
y_pred = df_train[["ID"]].copy()
y_pred["prediction"] = oof_xgb # Use -00f_xgb for Chris' solution
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"\nOverall CV for XGBoost =",m)

#%% Nelson Aalen model ----------------------------------
print('NelsonAalen')

FEATURES = nums + cats   
KFOLD = 5
kf = KFold(n_splits=KFOLD, shuffle=True, random_state=42)
df_y = df_target[['na_hazard']].copy()
oof_xgb2 = np.zeros(len(df_train))
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
    oof_xgb2[valid_idx] = y_pred
    
    if i>0:
        preds += xgb_model.predict(df_test.loc[:,FEATURES])
    else:
        preds = xgb_model.predict(df_test.loc[:,FEATURES])
    
preds /= KFOLD   
print(f"Mean Score {np.mean(best_score)}")

y_true = df_train[["ID","efs","efs_time","race_group"]].copy()
y_pred = df_train[["ID"]].copy()
y_pred["prediction"] = oof_xgb2
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"\nOverall CV for XGBoost =",m)

#%% XGB survival

params = {'tree_method': 'hist',
                'n_estimators': 5000,
                'objective': 'survival:cox', #'reg:squarederror', #'reg:absoluteerror', #'reg:logistic', #  'reg:squaredlogerror', #
                'random_state': 42,
                'enable_categorical': True,
                #'verbosity': 0,
                'early_stopping_rounds': 100,
                'eval_metric': 'cox-nloglik', # "mae", # 
                # Using Fisher grouping of categorical features. None=One hot
                "max_cat_to_onehot" : None,
                
                "max_depth":3,  
                "colsample_bytree":0.5, 
                "subsample":0.8, 
                "learning_rate":0.2,
                "min_child_weight": 80,
                }


FEATURES = nums + cats   
KFOLD = 5
kf = KFold(n_splits=KFOLD, shuffle=True, random_state=42)
df_y = df_target[['efs_time2']].copy() #'na_hazard']].copy() #'cox_hazard']].copy() #
oof_xgb_cox = np.zeros(len(df_train))
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
    oof_xgb_cox[valid_idx] = y_pred
    
    if i>0:
        preds += xgb_model.predict(df_test.loc[:,FEATURES])
    else:
        preds = xgb_model.predict(df_test.loc[:,FEATURES])
    
preds /= KFOLD   
print(f"Mean Score {np.mean(best_score)}")

y_true = df_train[["ID","efs","efs_time","race_group"]].copy()
y_pred = df_train[["ID"]].copy()
y_pred["prediction"] = oof_xgb_cox # Use -00f_xgb for Chris' solution
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"\nOverall CV for XGBoost =",m)


#%%

oof = rankdata(oof_xgb) + rankdata(oof_xgb2) + rankdata(oof_xgb_cox) 

#GET SCORE
y_true = df_train[["ID","efs","efs_time","race_group"]].copy()
y_pred = df_train[["ID"]].copy()
y_pred["prediction"] = oof
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"\nEnsemble oof =",m)

#%%

# %%

with open('oof/xgb_km.npy', 'wb') as f:
    np.save(f, oof_xgb)
with open('oof/xgb_na.npy', 'wb') as f:
    np.save(f, oof_xgb2)
with open('oof/xgb_cox.npy', 'wb') as f:
    np.save(f, oof_xgb_cox)
    
    
#%%

with open('oof/xgb_km.npy', 'rb') as f:
    xgb_km = np.load(f)
with open('oof/xgb_na.npy', 'rb') as f:
    xgb_na = np.load(f)
with open('oof/xgb_cox.npy', 'rb') as f:
    xgb_cox = np.load(f)

oof = rankdata(xgb_km) + rankdata(xgb_na) + rankdata(xgb_cox)  

#GET SCORE
y_true = df_train[["ID","efs","efs_time","race_group"]].copy()
y_pred = df_train[["ID"]].copy()
y_pred["prediction"] = oof
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"\nEnsemble oof =",m)



# %%
