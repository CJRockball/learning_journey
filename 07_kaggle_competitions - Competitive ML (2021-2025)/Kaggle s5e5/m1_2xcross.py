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

# %% check data distribution, plot histograms


ncols = range(len(nums))
comb_list2 = list(combinations_with_replacement(ncols, 2))
print('adding: ', len(comb_list2))
cross_cols = []
for nr1, nr2 in comb_list2:
    name1, name2 = nums[nr1], nums[nr2]
    df_train[f'{name1}-m-{name2}'] = df_train[name1] * df_train[name2]
    df_test[f'{name1}-m-{name2}'] = df_test[name1] * df_test[name2]
    #cross_cols += [f'{name1}-m-{name2}']
    # df_train[f'{name1}-d-{name2}'] = df_train[name1] / (df_train[name2] + 1e-5)
    # df_test[f'{name1}-d-{name2}'] = df_test[name1] / (df_test[name2] + 1e-5)
    # cross_cols += [f'{name1}-d-{name2}']


print(df_train.memory_usage(deep=True).sum() / 1024**2)
print(df_train.shape)
# chosen_features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'Sex', 
# 'Age-m-Age', 'Age-m-Height', 'Age-m-Duration', 'Age-m-Body_Temp', 
# 'Height-m-Duration', 'Weight-m-Weight']

# df_train = df_train[chosen_features + target]
# df_test = df_test[chosen_features]

df_train[cats] = df_train[cats].astype('category')
df_test[cats] = df_test[cats].astype('category')

#%% XGB model

KFOLD = 5
start_time = time.time()
df_y = np.log1p(df_train[target])
df_X = df_train.drop(columns=target).copy(deep=True)
df_Xtest = df_test.copy(deep=True)

kf = KFold(n_splits=KFOLD, shuffle=True, random_state=42)
xgb_oof = np.zeros(len(df_train))
rmsle_score_list = []

for i, (train_index, valid_index) in enumerate(kf.split(df_X)):
    print(f'Training round {i}')
    # Split data
    Xtrain = df_X.loc[train_index].copy()
    ytrain = df_y.loc[train_index].copy()
    Xvalid = df_X.loc[valid_index].copy()
    yvalid = df_y.loc[valid_index].copy()
    Xtest = df_Xtest.copy()
   
    
    model = XGBRegressor(
                max_depth=8, #9, #6,            
                colsample_bytree=0.7, 
                subsample= 0.6, #0.9, 
                n_estimators=10000, 
                learning_rate=0.015, 
                early_stopping_rounds=100,  
                eval_metric="rmse",
                reg_alpha = 0.98,
                reg_lambda = 0.12,
                device="cuda:0",
                enable_categorical=True,
            )
    model.fit(
                Xtrain, ytrain,
                eval_set=[(Xtrain, ytrain), (Xvalid, yvalid)],   
                verbose=500
            )    
    
    # Predict and estimate
    # XGB automatically recovers the best fit hen using the early-stopping
    y_pred = model.predict(Xvalid)
    rmsle = root_mean_squared_error(yvalid, y_pred)
    print(rmsle)
    rmsle_score_list.append(rmsle)
    xgb_oof[valid_index] = y_pred
    
    # Make train file
    if i>0:
        xgb_preds += model.predict(Xtest)
    else:
        xgb_preds = model.predict(Xtest)
    
xgb_preds /= KFOLD
end_time = time.time()
print(f'Total time: {end_time - start_time}')
print(f'Ind fold errors: {np.round(rmsle_score_list,4)}')
print(f'Mean rmsle: {np.mean(rmsle_score_list)}, std: {np.std(rmsle_score_list)}')

#%%

# Get feature importance scores
importance_scores = model.get_booster().get_score(importance_type='gain')
df_imp = pd.DataFrame.from_dict(importance_scores, orient='index', columns=['Importance'])

display(df_imp)
df_imp.plot(kind='barh')



#%%

y_preds_xgb = np.expm1(xgb_preds)
print('predict mean :',y_preds_xgb.mean())
print('predict median :',np.median(y_preds_xgb))

y_preds_xgb = np.clip(y_preds_xgb,1,314)
print('predict mean after clip:',y_preds_xgb.mean())
print('predict median after clip:',np.median(y_preds_xgb))



fname = 'xgb16'
df_sub = pd.read_csv('data/ra/sample_submission.csv')
df_sub['Calories'] = y_preds_xgb
df_sub.to_csv(f'submissions/{fname}.csv', index=False)

df_check = pd.read_csv(f'submissions/{fname}.csv')
display(df_check)

xgb_oof_conv = np.expm1(xgb_oof)
df_xgb_oof = pd.DataFrame(data=xgb_oof_conv, index=df_train.index, columns=[fname])
df_xgb_oof.to_csv(f'oof/{fname}.csv', index=False)

df_check = pd.read_csv(f'oof/{fname}.csv')
display(df_check)
                  

#%%
params = {
    "boosting_type": "gbdt",
    "objective": "regression",
    "metric": 'rmse', #'mae', #'custom',
    "verbose": -1,
    'random_state': 42,
    'n_jobs':-1,
    "feature_pre_filter" : False,
    
    'learning_rate': 0.02,
    'num_leaves': 1024,
    #'min_child_weight':1,
    'subsample': 0.6,
    'colsample_bytree': 0.75,
    'reg_alpha':0.001, 
    'reg_lambda':0.001
        }

NUM_ROUNDS = 5000
KFOLD = 5
kf = KFold(n_splits=KFOLD, shuffle=True)

start_time = time.time()
df_y = np.log1p(df_train[target].copy())
df_X = df_train.drop(columns=target).copy(deep=True)
df_Xtest = df_test.copy(deep=True)

kf = KFold(n_splits=KFOLD, shuffle=True, random_state=42)
lgb_oof = np.zeros(len(df_train))
rmsle_score_list = []

for i, (train_index, valid_index) in enumerate(kf.split(df_X)):
    print(f'Training round {i}')
    # Split data
    Xtrain = df_X.loc[train_index].copy()
    ytrain = df_y.loc[train_index].copy()
    Xvalid = df_X.loc[valid_index].copy()
    yvalid = df_y.loc[valid_index].copy()
    Xtest = df_Xtest.copy()


    # Make model, train
    trn_data = lgb.Dataset(Xtrain, ytrain, categorical_feature=cats)
    lgb_eval = [lgb.Dataset(Xvalid, yvalid)]
    
    model = lgb.train(params, trn_data, NUM_ROUNDS,
                      valid_sets=lgb_eval,
                      callbacks = [
                          lgb.log_evaluation(100),
                          lgb.early_stopping(stopping_rounds=100) #, verbose=0)
                      ])
    

    # Predict and estimate
    # XGB automatically recovers the best fit hen using the early-stopping
    y_pred = model.predict(Xvalid)
    rmsle = root_mean_squared_error(yvalid, y_pred)
    print(rmsle)
    rmsle_score_list.append(rmsle)
    lgb_oof[valid_index] = y_pred
    
    # Make train file
    if i>0:
        lgb_preds += model.predict(Xtest)
    else:
        lgb_preds = model.predict(Xtest)
    
lgb_preds /= KFOLD
end_time = time.time()
print(f'Total time: {end_time - start_time}')
print(f'Ind fold errors: {np.round(rmsle_score_list,4)}')
print(f'Mean rmsle: {np.mean(rmsle_score_list)}, std: {np.std(rmsle_score_list)}')

#%%

y_preds_lgb = np.expm1(lgb_preds)
print('predict mean :',y_preds_lgb.mean())
print('predict median :',np.median(y_preds_lgb))

y_preds_lgb = np.clip(y_preds_lgb,1,314)
print('predict mean after clip:',y_preds_lgb.mean())
print('predict median after clip:',np.median(y_preds_lgb))

fname = 'lgb4'
df_sub = pd.read_csv('data/ra/sample_submission.csv')
df_sub['Calories'] = y_preds_lgb
df_sub.to_csv(f'submissions/{fname}.csv', index=False)

df_check = pd.read_csv(f'submissions/{fname}.csv')
display(df_check)


lgb_oof_conv = np.expm1(lgb_oof)
df_lgb_oof = pd.DataFrame(data=lgb_oof_conv, index=df_train.index, columns=[fname])
df_lgb_oof.to_csv(f'oof/{fname}.csv', index=False)

df_check = pd.read_csv(f'oof/{fname}.csv')
display(df_check)


#%%
from catboost import CatBoostRegressor, Pool
# Cat params
random_state = 1337
cat_list = cats #+ nums

cat_params={
    'iterations': 1500,
    'depth': 9,
    'learning_rate': 0.02,
    #'eta': 1, 
    'reg_lambda': 1, 
    'colsample_bylevel': 0.6,
    'subsample': 0.8,
    'loss_function': 'RMSE',
    'eval_metric': 'RMSE',
    'cat_features': cat_list, 
    'random_state': random_state,
    #'min_data_in_leaf': 51,
    'early_stopping_rounds': 50,
    #'max_bin': 5000, # Can try to blow out max_bins. Didn't work here
    #'task_type': "GPU",
    'verbose':100,
    #'random_strength': 1.5,
    'bootstrap_type': 'Bernoulli',
}

NUM_ROUNDS = 5000
KFOLD = 5
kf = KFold(n_splits=KFOLD, shuffle=True)

start_time = time.time()
df_y = np.log1p(df_train[target].copy())
df_X = df_train.drop(columns=target).copy(deep=True)
df_Xtest = df_test.copy(deep=True)

kf = KFold(n_splits=KFOLD, shuffle=True, random_state=42)
cat_oof = np.zeros(len(df_train))
rmsle_score_list = []

for i, (train_index, valid_index) in enumerate(kf.split(df_X)):
    print(f'Training round {i}')
    # Split data
    Xtrain = df_X.loc[train_index].copy()
    ytrain = df_y.loc[train_index].copy()
    Xvalid = df_X.loc[valid_index].copy()
    yvalid = df_y.loc[valid_index].copy()
    Xtest = df_Xtest.copy()


    # Make model, train
    # Fit model
    train_pool = Pool(data=Xtrain, label=ytrain, cat_features=['Sex'])
    valid_pool = Pool(data=Xvalid, label=yvalid, cat_features=['Sex'])
        
    model = CatBoostRegressor(**cat_params,)
        
    model.fit(train_pool, eval_set=valid_pool)
        
    # Predict and estimate
    # XGB automatically recovers the best fit hen using the early-stopping
    y_pred = model.predict(Xvalid)
    rmsle = root_mean_squared_error(yvalid, y_pred)
    print(rmsle)
    rmsle_score_list.append(rmsle)
    cat_oof[valid_index] = y_pred
    
    # Make train file
    if i>0:
        cat_preds += model.predict(Xtest)
    else:
        cat_preds = model.predict(Xtest)
    
cat_preds /= KFOLD
end_time = time.time()
print(f'Total time: {end_time - start_time}')
print(f'Ind fold errors: {np.round(rmsle_score_list,4)}')
print(f'Mean rmsle: {np.mean(rmsle_score_list)}, std: {np.std(rmsle_score_list)}')

#%%

y_preds_cat = np.expm1(cat_preds)
print('predict mean :',y_preds_cat.mean())
print('predict median :',np.median(y_preds_cat))

y_preds_cat = np.clip(y_preds_cat,1,314)
print('predict mean after clip:',y_preds_cat.mean())
print('predict median after clip:',np.median(y_preds_cat))



fname = 'cat2'
df_sub = pd.read_csv('data/ra/sample_submission.csv')
df_sub['Calories'] = y_preds_cat
df_sub.to_csv(f'submissions/{fname}.csv', index=False)

df_check = pd.read_csv(f'submissions/{fname}.csv')
display(df_check)


cat_oof_conv = np.expm1(cat_oof)
df_cat_oof = pd.DataFrame(data=cat_oof_conv, index=df_train.index, columns=[fname])
df_cat_oof.to_csv(f'oof/{fname}.csv', index=False)

df_check = pd.read_csv(f'oof/{fname}.csv')
display(df_check)


# %%

eq_mix_preds = 1/3*y_preds_cat + 1/3*y_preds_lgb + 1/3*y_preds_xgb

fname = 'stack1'
df_sub = pd.read_csv('data/ra/sample_submission.csv')
df_sub['Calories'] = eq_mix_preds
df_sub.to_csv(f'submissions/{fname}.csv', index=False)

df_check = pd.read_csv(f'submissions/{fname}.csv')
display(df_check)


# %%
