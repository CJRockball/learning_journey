#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import root_mean_squared_log_error, root_mean_squared_error

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

# %% check data distribution, plot histograms

# Normalize all num features
# mx = df_train[nums].max()
# mn = df_train[nums].min()

# df_train[nums] = (df_train[nums] - mn) / (mx - mn)
# df_test[nums] = (df_test[nums] - mn) / (mx - mn)


# Make max,min,mean,std features
def stat_feat(df):
    for name in nums:
        df[f'{name}_max'] = df[name].max()
        df[f'{name}_min'] = df[name].min()
        df[f'{name}_mean'] = df[name].mean()
        df[f'{name}_std'] = df[name].std()
    return df

stat_feat(df_train)
stat_feat(df_test)




#%%
from itertools import combinations, combinations_with_replacement

ncols = range(len(nums))
comb_list2 = list(combinations_with_replacement(ncols, 2))
print('adding: ', len(comb_list2))
for nr1, nr2 in comb_list2:
    name1, name2 = nums[nr1], nums[nr2]
    df_train[f'{name1}_{name2}'] = np.log1p(df_train[name1] * df_train[name2])
    df_test[f'{name1}_{name2}'] = np.log1p(df_test[name1] * df_test[name2])



ncols = range(len(nums))
comb_list3 = list(combinations_with_replacement(ncols, 3))
print('adding: ', len(comb_list3))
for nr1, nr2, nr3 in comb_list3:
    name1, name2, name3 = nums[nr1], nums[nr2], nums[nr3]
    df_train[f'{name1}_{name2}_{name3}'] = np.log1p(df_train[name1] * df_train[name2] * df_train[name3])
    df_test[f'{name1}_{name2}_{name3}'] = np.log1p(df_test[name1] * df_test[name2] * df_test[name3])



ncols = range(len(nums))
comb_list4 = list(combinations_with_replacement(ncols, 4))
print('adding: ', len(comb_list4))
for nr1, nr2, nr4, nr4 in comb_list4:
    name1, name2, name3, name4 = nums[nr1], nums[nr2], nums[nr3], nums[nr4]
    df_train[f'{name1}_{name2}_{name3}_{name4}'] = np.log1p(df_train[name1] * df_train[name2] + df_train[name3] * df_train[name4])
    df_test[f'{name1}_{name2}_{name3}_{name4}'] = np.log1p(df_test[name1] * df_test[name2] * df_test[name3] * df_test[name4])

print(df_train.shape)
print(df_test.shape)
print(df_train.memory_usage(deep=True).sum() / 1024**2)

# %%

def default_settings():
    params = {'booster':'gbtree',
                'tree_method': 'hist',
                'objective': 'reg:squaredlogerror', #'reg:squarederror', #'reg:squaredlogerror', #
                'eval_metric': 'rmse',
                'random_state': 1337,
                }
    return params

def params1():
    params = {'max_depth':6,  
            'colsample_bytree':0.8, 
            'subsample':0.96, 
            'learning_rate':0.01, 
            'alpha':0.96,
            'lambda':0.12,}
    return params



#%%

KFOLD = 5
kf = KFold(n_splits=KFOLD, shuffle=True)

start_time = time.time()
df_y = df_train[target]
df_X = df_train.drop(columns=target)

kf = KFold(n_splits=KFOLD, shuffle=True, random_state=42)
oof = np.zeros(len(df_train))
model_list = []
rmsle_score_list = []
start = time.time()
for i, (train_index, valid_index) in enumerate(kf.split(df_X)):
    print(f'Training round {i}')
    # Split data
    Xtrain = df_X.iloc[train_index].copy()
    ytrain = df_y.iloc[train_index].copy()
    Xvalid = df_X.iloc[valid_index].copy()
    yvalid = df_y.iloc[valid_index].copy()
    Xtest = df_test.copy()

    # #Create regression matrices
    # dtrain = xgb.DMatrix(Xtrain, ytrain, enable_categorical=True)
    # dvalid = xgb.DMatrix(Xvalid, yvalid, enable_categorical=True)
    # dtest  = xgb.DMatrix(Xtest, enable_categorical=True)

    # p1 = default_settings()
    # p2 = params1()
    # p3 = p1 | p2
    
    # results = {}
    # #Fit model
    # # model = xgb.train(params=p3,
    # #             dtrain=dtrain,
    # #             evals=[(dtrain, 'dtrain'), (dvalid, 'dvalid')],
    # #             num_boost_round=5000,
    # #             maximize=True,
    # #             early_stopping_rounds=5000,
    # #             verbose_eval=500,
    # #             evals_result=results,
    # #             )
    
    model = XGBRegressor(
                max_depth=12, #6, 
                colsample_bytree=0.5, 
                subsample=0.3, 
                n_estimators=10000, 
                learning_rate=0.01, 
                early_stopping_rounds=50,  
                eval_metric="rmsle",
                reg_alpha = 1, #0.98,
                reg_lambda = 1, # 0.12,
                device='cpu', #"cuda:0",
                nthread=4,
                enable_categorical=True,
            )
    model.fit(
                Xtrain, ytrain,
                eval_set=[(Xtrain, ytrain), (Xvalid, yvalid)],   
                verbose=100
            )    
    
    
    
    
    # Predict and estimate
    # XGB automatically recovers the best fit hen using the early-stopping
    y_pred = model.predict(Xvalid)
    rmsle = root_mean_squared_log_error(yvalid, y_pred)
    rmsle_score_list.append(rmsle)
    oof[valid_index] = y_pred
    print(rmsle)
    
    # Make train file
    if i>0:
        preds += model.predict(Xtest)
    else:
        preds = model.predict(Xtest)
    
preds /= KFOLD
end_time = time.time()
print(f'Total time: {end_time - start_time}')
print(f'Ind fold errors: {np.round(rmsle_score_list,4)}')
print(f'Mean rmsle: {np.mean(rmsle_score_list)}, std: {np.std(rmsle_score_list)}')

# plt.figure()
# plt.plot(range(len(results['dtrain']['rmse'])), results['dtrain']['rmse'], label='Train')
# plt.plot(range(len(results['dtrain']['rmse'])), results['dvalid']['rmse'], label='Valid')
# plt.legend()
# plt.grid()
# plt.title('Train/Validation RMSE')
# plt.show()

end = time.time()
print('Total time: ', end-start)

#%%

# Get feature importance scores
importance_scores = model.get_score(importance_type='gain')
df_imp = pd.DataFrame.from_dict(importance_scores, orient='index', columns=['Importance'])

display(df_imp)
df_imp.plot(kind='barh')


#%%

fname = 'lgb4'

df_sub = pd.read_csv('data/ra/sample_submission.csv')
df_sub['Calories'] = preds #np.exp(preds) - 1
df_sub.to_csv(f'submissions/{fname}.csv', index=False)

df_check = pd.read_csv(f'submissions/{fname}.csv')
display(df_check)

#lgb4 cv

# %%
