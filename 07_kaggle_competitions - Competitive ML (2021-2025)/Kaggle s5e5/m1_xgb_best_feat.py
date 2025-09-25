#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import root_mean_squared_log_error, root_mean_squared_error
from itertools import combinations, combinations_with_replacement

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
df_train['Sex'] = df_train['Sex'].replace(sex_transform)
df_test['Sex'] = df_test['Sex'].replace(sex_transform)

display(df_train)
print(df_train.shape)
# display(df_train_org)

target = ['Calories']
nums = [name for name in df_train.columns if name not in [target[0], 'Sex']]
cats = ['Sex']
features = nums + cats

# %% check data distribution, plot histograms

# Normalize all num features
# mx = df_train[nums].max()
# mn = df_train[nums].min()

# df_train[nums] = (df_train[nums] - mn) / (mx - mn)
# df_test[nums] = (df_test[nums] - mn) / (mx - mn)


# Make max,min,mean,std features
def stat_feat(df):
    add_cols = []
    for name in nums:
        df[f'{name}_max'] = df[name].max()
        df[f'{name}_min'] = df[name].min()
        df[f'{name}_mean'] = df[name].mean()
        df[f'{name}_std'] = df[name].std()
        add_cols = add_cols + [f'{name}_std', f'{name}_mean', f'{name}_min', f'{name}_max']
    return df, add_cols

df_train, add_cols = stat_feat(df_train)
df_test, _ = stat_feat(df_test)


ncols = range(len(nums))
comb_list2 = list(combinations_with_replacement(ncols, 2))
print('adding: ', len(comb_list2))
cross_cols = []
for nr1, nr2 in comb_list2:
    name1, name2 = nums[nr1], nums[nr2]
    df_train[f'{name1}_{name2}'] = df_train[name1] * df_train[name2]
    df_test[f'{name1}_{name2}'] = df_test[name1] * df_test[name2]
    cross_cols += [f'{name1}_{name2}']

# cross_cols = cross_cols + add_cols
features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'Sex', 
 'Height_Height', 'Height_Duration', 'Height_Heart_Rate', 'Height_Body_Temp', 
 'Weight_Body_Temp', 'Heart_Rate_Heart_Rate']

df_train = df_train[features + target]
df_test = df_test[features]
print(df_train.memory_usage(deep=True).sum() / 1024**2)
print(df_train.shape)

#%%
ncols = range(len(nums))
comb_list3 = list(combinations_with_replacement(ncols, 3))
print('adding: ', len(comb_list3))
cross_cols = []

for nr1, nr2, nr3 in comb_list3:
    name1, name2, name3 = nums[nr1], nums[nr2], nums[nr3]
    df_train[f'{name1}_{name2}_{name3}'] = df_train[name1] * df_train[name2] * df_train[name3]
    df_test[f'{name1}_{name2}_{name3}'] = df_test[name1] * df_test[name2] * df_test[name3]
    cross_cols += [f'{name1}_{name2}_{name3}']
    
features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'Sex', 
            'Height_Height', 'Height_Duration', 'Height_Heart_Rate', 'Height_Body_Temp', 
            'Weight_Body_Temp', 'Heart_Rate_Heart_Rate', 'Age_Weight_Weight', 
            'Age_Heart_Rate_Body_Temp', 'Height_Height_Weight']

df_train = df_train[features + target]
df_test = df_test[features]
print(df_train.memory_usage(deep=True).sum() / 1024**2)
print(df_train.shape)

#%%
ncols = range(len(nums))
comb_list4 = list(combinations_with_replacement(ncols, 4))
print('adding: ', len(comb_list4))
cross_cols = []

for nr1, nr2, nr3, nr4 in comb_list4:
    name1, name2, name3, name4 = nums[nr1], nums[nr2], nums[nr3], nums[nr4]
    df_train[f'{name1}_{name2}_{name3}_{name4}'] = df_train[name1] * df_train[name2] + df_train[name3] * df_train[name4]
    df_test[f'{name1}_{name2}_{name3}_{name4}'] = df_test[name1] * df_test[name2] * df_test[name3] * df_test[name4]
    cross_cols += [f'{name1}_{name2}_{name3}_{name4}']

print(df_train.memory_usage(deep=True).sum() / 1024**2)
print(df_train.shape)      

#%%

KFOLD = 5
kf = KFold(n_splits=KFOLD, shuffle=True)

start_time = time.time()
df_y = df_train[target]
df_X = df_train.drop(columns=target).copy(deep=True)
df_Xtest = df_test.copy(deep=True)

kf = KFold(n_splits=KFOLD, shuffle=True, random_state=42)
oof = np.zeros(len(df_train))
total_features = features
best_rmsle = 0.0591696 # 0.059255 fastest 2x cross  # 0.05958 basic data set
print(f'Starting features: {len(total_features)}, df size: {df_X.shape}, best rmsle: {best_rmsle}')
rmsle_score_list = []
start = time.time()
for j, new_feature in enumerate(cross_cols):
    total_features = total_features + [new_feature]
    
    for i, (train_index, valid_index) in enumerate(kf.split(df_X)):
        #print(f'Training round {i}')
        # Split data
        Xtrain = df_X.loc[train_index,total_features].copy()
        ytrain = df_y.loc[train_index].copy()
        Xvalid = df_X.loc[valid_index,total_features].copy()
        yvalid = df_y.loc[valid_index].copy()
        #Xtest = df_Xtest.loc[:,total_features].copy()

        model = XGBRegressor(
                    max_depth=9, #6,            
                    colsample_bytree=0.8, 
                    subsample=0.9, 
                    n_estimators=10000, 
                    learning_rate=0.01, 
                    early_stopping_rounds=25,  
                    eval_metric="rmsle",
                    reg_alpha = 0.98,
                    reg_lambda = 0.12,
                    device="cuda:0",
                    nthread=4,
                    enable_categorical=True,
                )
        model.fit(
                    Xtrain, ytrain,
                    eval_set=[(Xtrain, ytrain), (Xvalid, yvalid)],   
                    verbose=0
                )    
        
        # Predict and estimate
        # XGB automatically recovers the best fit hen using the early-stopping
        y_pred = model.predict(Xvalid)
        rmsle = root_mean_squared_log_error(yvalid, y_pred)
        break
    if rmsle < best_rmsle:
        best_rmsle = rmsle
        print(f'feature {j}/{len(cross_cols)}, feature {new_feature}, rmsle: {rmsle}, num of features {len(total_features)}')
    else:
        total_features.remove(new_feature)
        df_X.drop(columns=[new_feature])
        df_Xtest.drop(columns=[new_feature])
        print(f'feature {j}/{len(cross_cols)}, feature {new_feature}, rmsle: {rmsle}')


print(total_features)
end = time.time()
print('Total time: ', end-start)

 #%%

# Get feature importance scores
importance_scores = model.get_score(importance_type='gain')
df_imp = pd.DataFrame.from_dict(importance_scores, orient='index', columns=['Importance'])

display(df_imp)
df_imp.plot(kind='barh')


#%%

fname = 'xgb5'

df_sub = pd.read_csv('data/ra/sample_submission.csv')
df_sub['Calories'] = preds #np.exp(preds) - 1
df_sub.to_csv(f'submissions/{fname}.csv', index=False)

df_check = pd.read_csv(f'submissions/{fname}.csv')
display(df_check)

# %%

['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'Sex', 
 'Height_Height', 'Height_Duration', 'Height_Heart_Rate', 'Height_Body_Temp',
 'Weight_Body_Temp', 'Heart_Rate_Heart_Rate', 'Age_Weight_Weight',
 'Age_Heart_Rate_Body_Temp', 'Height_Height_Weight', 'Age_Age_Heart_Rate_Heart_Rate',
 'Age_Age_Body_Temp_Body_Temp', 'Age_Height_Height_Height', 'Age_Height_Height_Body_Temp',
 'Age_Duration_Heart_Rate_Body_Temp', 'Age_Duration_Body_Temp_Body_Temp',
 'Age_Heart_Rate_Heart_Rate_Body_Temp', 'Height_Height_Height_Body_Temp', 
 'Duration_Duration_Duration_Duration', 'Heart_Rate_Heart_Rate_Heart_Rate_Body_Temp',
 'Body_Temp_Body_Temp_Body_Temp_Body_Temp']

