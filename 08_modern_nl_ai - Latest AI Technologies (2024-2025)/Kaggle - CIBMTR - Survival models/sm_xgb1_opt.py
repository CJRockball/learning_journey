#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, train_test_split
from xgboost import XGBRegressor, XGBClassifier
import xgboost
print("Using XGBoost version",xgboost.__version__)
#from catboost import CatBoostRegressor, Pool

"""
To evaluate the equitable prediction of transplant survival outcomes,
we use the concordance index (C-index) between a series of event
times and a predicted score across each race group.
 
It represents the global assessment of the model discrimination power:
this is the model’s ability to correctly provide a reliable ranking
of the survival times based on the individual risk scores.
 
The concordance index is a value between 0 and 1 where:
 
0.5 is the expected result from random predictions,
1.0 is perfect concordance (with no censoring, otherwise <1.0),
0.0 is perfect anti-concordance (with no censoring, otherwise >0.0)

"""

import pandas.api.types
from lifelines import KaplanMeierFitter, CoxPHFitter, NelsonAalenFitter
from lifelines.utils import concordance_index

class ParticipantVisibleError(Exception):
    pass


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    """
    >>> import pandas as pd
    >>> row_id_column_name = "id"
    >>> y_pred = {'prediction': {0: 1.0, 1: 0.0, 2: 1.0}}
    >>> y_pred = pd.DataFrame(y_pred)
    >>> y_pred.insert(0, row_id_column_name, range(len(y_pred)))
    >>> y_true = { 'efs': {0: 1.0, 1: 0.0, 2: 0.0}, 'efs_time': {0: 25.1234,1: 250.1234,2: 2500.1234}, 'race_group': {0: 'race_group_1', 1: 'race_group_1', 2: 'race_group_1'}}
    >>> y_true = pd.DataFrame(y_true)
    >>> y_true.insert(0, row_id_column_name, range(len(y_true)))
    >>> score(y_true.copy(), y_pred.copy(), row_id_column_name)
    0.75
    """
    
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

#%% Make all combinations

new_columns = {}
new_columns2 = {}
COLS2 = []
for i, c1 in enumerate(cats):
    for j, c2 in enumerate(cats[i+1:]):
        name = f"{c1}-{c2}"
        new_columns[name] = df_train[c1].astype("str") + "_" + df_train[c2].astype("str")
        new_columns2[name] = df_test[c1].astype("str") + "_" + df_test[c2].astype("str")
        COLS2.append(name)
        print(f"{i}-{i+j+1}, ", end='')
df_train = pd.concat([df_train, pd.DataFrame(new_columns)], axis=1)
df_test = pd.concat([df_test, pd.DataFrame(new_columns2)], axis=1)
print()
print(len(COLS2),"bi-grams generated")

#%%

cats = cats + COLS2

#%%
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
    
    df_train[cname] = df_train[cname].map(train_dict).astype(np.int32)
    df_test[cname] = df_test[cname].map(test_dict).astype(np.int32)       

# Feature Engineering

for cname in nums:
    df_train[cname] = df_train[cname].fillna(-1)
    df_test[cname] = df_test[cname].fillna(-1)


# FIx dfs
df_target = df_train[target].copy()
df_X = df_train.drop(columns=target).copy()
df_X[cats] = df_X[cats].astype('category')
FEATURES = nums + cats 


# -------------------- Choose a survival model ----------- 
# Try fitting the function and plotting
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

# # will make "target" the label 
df_train = update_target_with_survival_probabilities(df_train, time_col='efs_time', event_col='efs')
df_target = pd.DataFrame()
df_target['km'] = df_train['km']

#%%
params = {'tree_method': 'hist',
                'n_estimators': 5000,
                'objective': 'reg:squarederror', #'reg:absoluteerror', #'reg:logistic', #  'reg:squaredlogerror', #
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
pred_xgb = np.zeros(len(df_test))
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
        preds += np.exp(xgb_model.predict(df_test.loc[:,FEATURES])) - 1
    else:
        preds = np.exp(xgb_model.predict(df_test.loc[:,FEATURES])) - 1
    
preds /= KFOLD   
print(f"Mean Score {np.mean(best_score)}")

y_true = df_train[["ID","efs","efs_time","race_group"]].copy()
y_pred = df_train[["ID"]].copy()
y_pred["prediction"] = oof_xgb
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"\nC=index for XGBoost =",m)

#%% Check prediction

df_sub = pd.read_csv('data/ra/sample_submission.csv')
df_sub['prediction'] = preds
df_sub.to_csv('submissions/xgb1.csv', index=False)

df_check = pd.read_csv('submissions/xgb1.csv')
display(df_check)

#%%

print(Xtrain.shape)


# %%

imp_data = xgb_model.get_booster().get_score(importance_type='gain')

df_imp = pd.DataFrame.from_dict(imp_data, orient='index', columns=['gain']).astype(float)
df_imp.sort_values(by='gain', ascending=False).iloc[:25,:]['gain'].plot(kind='barh')

display(df_imp.sort_values(by='gain', ascending=False))

# %%
