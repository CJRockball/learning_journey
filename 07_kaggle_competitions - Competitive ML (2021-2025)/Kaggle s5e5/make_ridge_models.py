#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import root_mean_squared_log_error, root_mean_squared_error
from itertools import combinations, combinations_with_replacement

import lightgbm as lgb
import xgboost as xgb
from xgboost import XGBRegressor
import time 


#%%

df_train = pd.read_csv('data/ra/train.csv').drop(columns=['id'])
df_train_org = pd.read_csv('data/ra/calories.csv').drop(columns=['User_ID'])
df_train_org['Sex'] = df_train_org['Gender']
df_train_org = df_train_org.drop(columns=['Gender'])
df_train = pd.concat([df_train, df_train_org], axis=0).reset_index(drop=True)
df_train = df_train.drop_duplicates().reset_index(drop=True)
df_test =  pd.read_csv('data/ra/test.csv').drop(columns=['id'])

sex_transform = {'male':0, 'female':1}
df_train['Sex'] = df_train['Sex'].replace(sex_transform).astype('category')
df_test['Sex'] = df_test['Sex'].replace(sex_transform).astype('category')

display(df_train)
print(df_train.shape)
# display(df_train_org)

target = ['Calories']
nums = [name for name in df_train.columns if name not in [target[0], 'Sex']]
cats = ['Sex']
features = nums + cats

# %% make features, mix all 234x cross and 2x div

ncols = range(len(nums))
comb_list2 = list(combinations_with_replacement(ncols, 2))
print('adding: ', len(comb_list2))
for nr1, nr2 in comb_list2:
    name1, name2 = nums[nr1], nums[nr2]
    df_train[f'{name1}-m-{name2}'] = df_train[name1] * df_train[name2]
    df_test[f'{name1}-m-{name2}'] = df_test[name1] * df_test[name2]

    df_train[f'{name1}-d-{name2}'] = df_train[name1] / (df_train[name2] + 1e-5)
    df_test[f'{name1}-d-{name2}'] = df_test[name1] / (df_test[name2] + 1e-5)


print(df_train.memory_usage(deep=True).sum() / 1024**2)
print(df_train.shape)

#NO 3x CROSS IMPROVED THE RESULT
comb_list3 = list(combinations_with_replacement(ncols, 3))
print('adding: ', len(comb_list3))

for nr1, nr2, nr3 in comb_list3:
    name1, name2, name3 = nums[nr1], nums[nr2], nums[nr3]
    df_train[f'{name1}_{name2}_{name3}'] = df_train[name1] * df_train[name2] * df_train[name3]
    df_test[f'{name1}_{name2}_{name3}'] = df_test[name1] * df_test[name2] * df_test[name3]
    

print(df_train.memory_usage(deep=True).sum() / 1024**2)
print(df_train.shape)


ncols = range(len(nums))
comb_list4 = list(combinations_with_replacement(ncols, 4))
print('adding: ', len(comb_list4))

for nr1, nr2, nr3, nr4 in comb_list4:
    name1, name2, name3, name4 = nums[nr1], nums[nr2], nums[nr3], nums[nr4]
    df_train[f'{name1}_{name2}_{name3}_{name4}'] = df_train[name1] * df_train[name2] + df_train[name3] * df_train[name4]
    df_test[f'{name1}_{name2}_{name3}_{name4}'] = df_test[name1] * df_test[name2] * df_test[name3] * df_test[name4]

print(df_train.memory_usage(deep=True).sum() / 1024**2)
print(df_train.shape)  

#%% Make puposefull features

def feature_engineering(df):
    df['bmi'] = df['Weight'] / ((df['Height'] / 100) ** 2)    
    df['exercise_intensity'] = df['Heart_Rate'] / df['Duration']
    df['heart_rate_duration'] = df['Heart_Rate'] * df['Duration']
    df['temp_duration'] = df['Body_Temp'] * df['Duration']
    df['hr_to_temp'] = df['Heart_Rate'] / df['Body_Temp']
    df['hr_to_age'] = df['Heart_Rate'] / df['Age']
    df['age_bmi'] = df['Age'] * df['bmi']
    df['max_heart_rate'] = 220 - df['Age']
    df['heart_rate_intensity'] = df['Heart_Rate'] / df['max_heart_rate']


    # # add log-transform of skewed features
    # skewed_feats = ['Age', 'Weight', 'Body_Temp', 'Height', 'Duration', 'Heart_Rate']
    # for feat in skewed_feats:
    #     df[f'log_{feat}'] = np.log1p(df[feat])
    return df


df_train = feature_engineering(df_train)
df_test = feature_engineering(df_test)
print(df_train.memory_usage(deep=True).sum() / 1024**2)
print(df_train.shape)  

# Normalize
nums = [cname for cname in df_train.columns if cname not in target + cats ]
m = df_train[nums].mean()
s = df_train[nums].std()
df_train[nums] = (df_train[nums] - m)/s
df_test[nums] = (df_train[nums] - m)/s

display(df_train)

# %%
from sklearn.linear_model import Ridge

KFOLD = 5
kf = KFold(n_splits=KFOLD, shuffle=True)

df_y = np.log1p(df_train[target])
df_X = df_train.drop(columns=target).copy(deep=True)
df_Xtest = df_test.copy(deep=True)

ridge_oof = np.zeros(len(df_X))
metric_list = []
for i, (trn_idx, vld_idx) in enumerate(kf.split(df_X)):

    Xtrain = df_X.loc[trn_idx].copy()
    ytrain = df_y.loc[trn_idx].copy()
    Xvalid = df_X.loc[vld_idx].copy()
    yvalid = df_y.loc[vld_idx].copy()
    Xtest = df_Xtest.copy()
    
    # Make model, predict
    l_model = Ridge()
    l_model.fit(Xtrain, ytrain)

    # Metrics
    y_pred = l_model.predict(Xvalid)
    rmsle_ = root_mean_squared_error(yvalid, y_pred)
    metric_list.append(rmsle_)
    
    ridge_oof[vld_idx] = y_pred
    # Make train file
    if i>0:
        ridge_preds += l_model.predict(Xtest)
    else:
        ridge_preds = l_model.predict(Xtest)
    

ridge_preds /= KFOLD
print(f'Metric List: {metric_list}')
print(f'Mean: {np.mean(metric_list)}')

#%%

y_preds_ridge = np.expm1(ridge_preds)
print('predict mean :',y_preds_ridge.mean())
print('predict median :',np.median(y_preds_ridge))

y_preds_ridge = np.clip(y_preds_ridge,1,314)
print('predict mean after clip:',y_preds_ridge.mean())
print('predict median after clip:',np.median(y_preds_ridge))

fname = 'ridge4'
df_sub = pd.read_csv('data/ra/sample_submission.csv')
df_sub['Calories'] = y_preds_ridge
df_sub.to_csv(f'submissions/{fname}.csv', index=False)

df_check = pd.read_csv(f'submissions/{fname}.csv')
display(df_check)

ridge_oof_conv = np.expm1(ridge_oof)
df_ridge_oof = pd.DataFrame(data=ridge_oof_conv, index=df_train.index, columns=[fname])
df_ridge_oof.to_csv(f'oof/{fname}.csv', index=False)

df_check = pd.read_csv(f'oof/{fname}.csv')
display(df_check)

# %%
