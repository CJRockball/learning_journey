#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import root_mean_squared_log_error, root_mean_squared_error
from itertools import combinations, combinations_with_replacement

import lightgbm as lgb
import xgboost as xgb
from xgboost import XGBRegressor
from catboost import CatBoostRegressor, Pool
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
df_train['Sex'] = df_train['Sex'].replace(sex_transform) #.astype('category')
df_test['Sex'] = df_test['Sex'].replace(sex_transform) #.astype('category')

display(df_train)
print(df_train.shape)
# display(df_train_org)

target = ['Calories']
nums = [name for name in df_train.columns if name not in [target[0], 'Sex']]
cats = ['Sex']
features = nums + cats

#%%
# Take duplicates, where the features are the same, and use only the lowest value as target
df_train = df_train.groupby(['Sex', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp'])['Calories'].min().reset_index()

# %% check data distribution, plot histograms


ncols = range(len(nums))
comb_list2 = list(combinations_with_replacement(ncols, 2))
print('adding: ', len(comb_list2))
# cross_cols = []
for nr1, nr2 in comb_list2:
    name1, name2 = nums[nr1], nums[nr2]
    df_train[f'{name1}-m-{name2}'] = df_train[name1] * df_train[name2]
    df_test[f'{name1}-m-{name2}'] = df_test[name1] * df_test[name2]
    #cross_cols += [f'{name1}-m-{name2}']

df_train['bmi'] = df_train['Weight'] /(df_train['Height'] / 100)**2
df_train['intensity'] = df_train['Heart_Rate'] / df_train['Duration']
df_test['bmi'] = df_test['Weight'] /(df_test['Height'] / 100)**2
df_test['intensity'] = df_test['Heart_Rate'] / df_test['Duration']

df_train[cats] = df_train[cats].astype('category')
df_test[cats] = df_test[cats].astype('category')

#%% XGB model

KFOLD = 5
start_time = time.time()
df_y = np.log1p(df_train[target])
df_X = df_train.drop(columns=target).copy(deep=True)
df_Xtest = df_test.copy(deep=True)

bins = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
duration_bins = bins.fit_transform(df_train[['Duration']]).astype(int).flatten()
skf = StratifiedKFold(n_splits=KFOLD, shuffle=True, random_state=42)

xgb_oof = np.zeros(len(df_train))
cat_oof = np.zeros(len(df_train))
xgb_preds = np.zeros(len(df_test))
cat_preds = np.zeros(len(df_test))
rmsle_score_list = []

for i, (train_index, valid_index) in enumerate(skf.split(df_X, duration_bins)):
    print(f'Training round {i}')
    # Split data
    Xtrain = df_X.loc[train_index].copy()
    ytrain = df_y.loc[train_index].copy()
    Xvalid = df_X.loc[valid_index].copy()
    yvalid = df_y.loc[valid_index].copy()
    Xtest = df_Xtest.copy()
   
    # XGB    
    model = XGBRegressor(
                max_depth=10, #9, #6,            
                colsample_bytree=0.7, 
                subsample= 0.9, #0.9, 
                n_estimators=10000, 
                learning_rate=0.02, 
                early_stopping_rounds=100,  
                eval_metric="rmse",
                reg_alpha=0.98,
                reg_lambda=0.12,
                device="cuda:0",
                gamma=0.01,
                max_delta_step=2,
                enable_categorical=True,
                random_state = 42,
            )
    model.fit(
                Xtrain, ytrain,
                eval_set=[(Xtrain, ytrain), (Xvalid, yvalid)],   
                verbose=500
            )    
    
    # Predict and estimate
    # XGB automatically recovers the best fit hen using the early-stopping
    xgb_oof[valid_index] = model.predict(Xvalid)
    xgb_preds += model.predict(Xtest)/KFOLD


    # CAT
    train_pool = Pool(data=Xtrain, label=ytrain, cat_features=['Sex'])
    valid_pool = Pool(data=Xvalid, label=yvalid, cat_features=['Sex'])
    
    cat_model = CatBoostRegressor(verbose=500, random_state=42)
    cat_model.fit(train_pool, eval_set=valid_pool, early_stopping_rounds=50)
    # Predict and estimate
    
    cat_oof[valid_index] = cat_model.predict(Xvalid)    
    cat_preds += cat_model.predict(Xtest)/ KFOLD


final_pred = 0.5 * xgb_preds + 0.5 * cat_preds
final_oof = 0.5 * xgb_oof + 0.5 * cat_oof
final_rmsle = root_mean_squared_error(final_oof, df_y)
print(f'OOF rmsle: {final_rmsle}')

end_time = time.time()
print(f'Total time: {end_time - start_time}')
# print(f'Ind fold errors: {np.round(rmsle_score_list,4)}')
# print(f'Mean rmsle: {np.mean(rmsle_score_list)}, std: {np.std(rmsle_score_list)}')

#%%

print(np.expm1(xgb_oof[-5:]))

print(np.expm1(df_y.iloc[-5:]))

#%%

# Get feature importance scores
importance_scores = model.get_booster().get_score(importance_type='gain')
df_imp = pd.DataFrame.from_dict(importance_scores, orient='index', columns=['Importance'])

display(df_imp)
df_imp.plot(kind='barh')



#%%

y_preds_xgb = np.expm1(cat_preds)
print('predict mean :',y_preds_xgb.mean())
print('predict median :',np.median(y_preds_xgb))

y_preds_xgb = np.clip(y_preds_xgb,1,314)
print('predict mean after clip:',y_preds_xgb.mean())
print('predict median after clip:',np.median(y_preds_xgb))



fname = 'xgb17'
df_sub = pd.read_csv('data/ra/sample_submission.csv')
df_sub['Calories'] = y_preds_xgb
df_sub.to_csv(f'submissions/{fname}.csv', index=False)

df_check = pd.read_csv(f'submissions/{fname}.csv')
display(df_check)

xgb_oof_conv = np.expm1(cat_oof)
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

old_oof = pd.read_csv('oof/knn.csv')

display(old_oof)



# %%
