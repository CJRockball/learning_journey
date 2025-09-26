""" Simplest possible run XGB with cats and copy the one prediction to all 3
"""
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import auc, log_loss

import xgboost as xgb
from xgboost import XGBClassifier
import time 


#%% import data

df_train = pd.read_csv('data/ra/train.csv')
df_org = pd.read_csv('data/ra/Fertilizer Prediction.csv')
df_test = pd.read_csv('data/ra/test.csv')

df_train = df_train.drop(columns=['id'])
df_test = df_test.drop(columns=['id'])

df_train = pd.concat([df_train, df_org], axis=0)

target = ['Fertilizer Name']
cats = ['Soil Type', 'Crop Type', 'Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
nums = []

df_train[['Soil Type', 'Crop Type']] = df_train[['Soil Type', 'Crop Type']].astype('category')
df_test[['Soil Type', 'Crop Type']] = df_test[['Soil Type', 'Crop Type']].astype('category')

#%% add feature

def soil_crop_compatibility_flag(df: pd.DataFrame,
                                 soil_col: str = "Soil Type",
                                 crop_col: str = "Crop Type",
                                 flag_name: str = "soil_crop_compatibility"
) -> pd.DataFrame:
    """
    Add an agronomy-based compatibility score (0–1) for every row.

    Parameters
    ----------
    df        : DataFrame with two categorical columns,
                e.g. “Soil Type” and “Crop Type”.
    soil_col  : name of the soil column  (default “Soil Type”)
    crop_col  : name of the crop column  (default “Crop Type”)
    flag_name : name of the new compatibility column.

    Returns
    -------
    DataFrame  —  original df plus one extra float column.
    """

    # -------------------------------- knowledge table ------------------------------
    compat = {
        # Clayey soil
        ("Clayey", "Paddy"): 1.0,  ("Clayey", "Sugarcane"): 0.9,
        ("Clayey", "Cotton"): 0.7, ("Clayey", "Wheat"): 0.6,
        ("Clayey", "Maize"): 0.6,  ("Clayey", "Millets"): 0.4,
        ("Clayey", "Barley"): 0.5, ("Clayey", "Tobacco"): 0.5,
        ("Clayey", "Ground Nuts"): 0.3, ("Clayey", "Pulses"): 0.6,
        ("Clayey", "Oil seeds"): 0.5,

        # Sandy soil
        ("Sandy", "Ground Nuts"): 1.0, ("Sandy", "Millets"): 0.9,
        ("Sandy", "Cotton"): 0.8,      ("Sandy", "Oil seeds"): 0.8,
        ("Sandy", "Tobacco"): 0.7,     ("Sandy", "Barley"): 0.7,
        ("Sandy", "Pulses"): 0.7,      ("Sandy", "Wheat"): 0.6,
        ("Sandy", "Maize"): 0.6,       ("Sandy", "Sugarcane"): 0.4,
        ("Sandy", "Paddy"): 0.2,

        # Loamy soil
        ("Loamy", "Wheat"): 1.0,   ("Loamy", "Maize"): 1.0,
        ("Loamy", "Cotton"): 0.9,  ("Loamy", "Tobacco"): 0.9,
        ("Loamy", "Barley"): 0.9,  ("Loamy", "Pulses"): 0.8,
        ("Loamy", "Ground Nuts"): 0.8, ("Loamy", "Oil seeds"): 0.8,
        ("Loamy", "Sugarcane"): 0.8,   ("Loamy", "Millets"): 0.7,
        ("Loamy", "Paddy"): 0.7,

        # Red soil
        ("Red", "Millets"): 0.9,  ("Red", "Cotton"): 0.8,
        ("Red", "Ground Nuts"): 0.8, ("Red", "Oil seeds"): 0.7,
        ("Red", "Tobacco"): 0.7,  ("Red", "Maize"): 0.7,
        ("Red", "Pulses"): 0.7,   ("Red", "Wheat"): 0.6,
        ("Red", "Barley"): 0.6,   ("Red", "Sugarcane"): 0.6,
        ("Red", "Paddy"): 0.5,

        # Black soil
        ("Black", "Cotton"): 1.0, ("Black", "Sugarcane"): 0.8,
        ("Black", "Pulses"): 0.8, ("Black", "Wheat"): 0.7,
        ("Black", "Maize"): 0.7,  ("Black", "Oil seeds"): 0.7,
        ("Black", "Tobacco"): 0.6, ("Black", "Barley"): 0.6,
        ("Black", "Ground Nuts"): 0.6, ("Black", "Paddy"): 0.6,
        ("Black", "Millets"): 0.5,
    }

    # ----------------------------- compute score -----------------------------------
    def score(row):
        return compat.get((row[soil_col], row[crop_col]), 0.5)

    out = df.copy()
    out[flag_name] = out.apply(score, axis=1)
    return out

df_train = soil_crop_compatibility_flag(df_train)
df_test = soil_crop_compatibility_flag(df_test)

nums = nums + ['soil_crop_compatibility']

#%% change classes for target

def ord_trans(df, features):
    train_len = len(df)
    
    for c in features:
        df[c], list_cat_names = df[c].factorize()

    df = df.iloc[:train_len,:].copy()
    target_dict = {i:name for i, name in enumerate(list_cat_names)}
    return df, target_dict

df_train, target_dict = ord_trans(df_train, target)

display(df_train)
print(target_dict)

#%%

params = {'max_depth': 7,
    'learning_rate': 0.05635134330984224,
    'subsample': 0.5605235929333594,
    'colsample_bytree': 0.5594578346445631,
    'min_child_weight': 6,
    'gamma': 0.35819323772520817,
    'reg_alpha': 0.9747714669120731,
    'reg_lambda': 0.7061465594372847,
    'objective': 'multi:softprob',
    'num_class': 7,
    'eval_metric': 'mlogloss',
    #'tree_method': 'gpu_hist',
    #'verbosity': 0,
    'n_estimators': 10000,
    'enable_categorical': True,
    'early_stopping_rounds':25,
    }


# MAP@3 Metric
def mapk(actual, predicted, k=3):
    def apk(a, p, k):
        if len(p) > k:
            p = p[:k]
        score = 0.0
        num_hits = 0.0
        for i, pred in enumerate(p):
            if pred == a and pred not in p[:i]:
                num_hits += 1.0
                score += num_hits / (i + 1.0)
        return score
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])



#%% run cv training

df_y = df_train[target].copy()
num_targets = df_y.nunique().values[0]
df_X = df_train[cats+nums].copy()
Xtest = df_test[cats+nums].copy()

# set up data
FOLDS = 10
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=1337)

oof = np.zeros((len(df_X), num_targets))
preds = np.zeros((len(Xtest), num_targets))

oof_true = []
oof_top3_preds = []
fold_loglosses = []
fold_map3s = []
for i,(train_index, valid_index) in enumerate(kf.split(df_X)):
    print(f"### FOLD {i} ###")
    # split data
    Xtrain = df_X.iloc[train_index]
    ytrain = df_y.iloc[train_index]
    Xvalid = df_X.iloc[valid_index]
    yvalid = df_y.iloc[valid_index]
    
    # xgb.train
    model = XGBClassifier(**params)
        
                        #   n_estimators=2000, 
                        #   objective='multi:softmax', #'multi:softprob',
                        #   early_stopping_rounds=100, 
                        #   enable_categorical=True, 
                        #   eval_metric='mlogloss',
                        #   n_jobs=4,
                        #   random_state=1337, 
                        #   #colsample_bytree=0.7, 
                        #   #max_delta_step = 0.5, 
                        #   #gamma = 0.001, 
                        #   max_depth = 5, 
                        #   device="cuda")
    
    model = model.fit(Xtrain, ytrain, 
                      eval_set=[(Xvalid, yvalid)], 
                      verbose=250)

    
    
    # predict
    ypred = model.predict_proba(Xvalid)
    fold_logloss = log_loss(yvalid, ypred)
    oof[valid_index] = ypred
    
    # save
    fold_loglosses.append(fold_logloss)
    
    top3_preds = np.argsort(oof[valid_index], axis=1)[:, -3:][:, ::-1]
    fold_map3 = mapk(yvalid[target[0]].tolist(), [list(p) for p in top3_preds], k=3)
    fold_map3s.append(fold_map3)
    
    print(f"  Fold {i+1} Log Loss: {fold_logloss:.5f}")
    print(f"  Fold {i+1} MAP@3: {fold_map3:.5f}")

    oof_true.extend(yvalid)
    oof_top3_preds.extend(top3_preds)
    preds += model.predict_proba(Xtest) / FOLDS

# Overall MAP@3
map3_score = mapk(oof_true, [list(p) for p in oof_top3_preds], k=3)
print(f"\nOverall OOF MAP@3 Score: {map3_score:.5f}")

#%%

fname = 'xgb5'
test_top3_preds = np.argsort(preds, axis=1)[:, -3:][:, ::-1]

submission_labels = []
for row in test_top3_preds:
    labels = [target_dict[i] for i in row]
    submission_labels.append(' '.join(labels))

df_sub = pd.read_csv('data/ra/sample_submission.csv')
df_sub['Fertilizer Name'] = submission_labels
df_sub.to_csv(f'submissions/{fname}.csv', index=False)

df_check = pd.read_csv(f'submissions/{fname}.csv')
display(df_check)
print("\nSubmission file saved!")


#%%
preds2 = preds.copy().astype(object)

for i in range(3):
    preds2[:,i] = np.vectorize(target_dict.get)(preds2[:,i])

print(preds2)

#%%

df_preds = pd.DataFrame(data=preds, columns=['First', 'Second', 'Third'])

for name in df_preds.columns:
 df_preds[name] = df_preds[name].map(target_dict).astype('str')

df_preds['sub'] = df_preds.First + ' ' + df_preds.Second + ' ' + df_preds.Third

display(df_preds)

# %%

fname = 'xgb1'

df_sub = pd.read_csv('data/ra/sample_submission.csv')
df_sub['Fertilizer Name'] = df_preds['sub']
df_sub.to_csv(f'submissions/{fname}.csv', index=False)

df_check = pd.read_csv(f'submissions/{fname}.csv')
display(df_check)

# %%

display(df_test)





# %%
