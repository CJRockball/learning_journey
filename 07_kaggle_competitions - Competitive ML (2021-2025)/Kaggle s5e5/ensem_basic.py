#%%
import pandas as pd

#name_list = ['xgb17', 'lgb5', 'knn2']
name_list = ['xgb13', 'lgb3', 'cat2', 'ridge2', 'knn', 'xgb_lgb3', 'nn5_cv0.0616']
ens_name = 'ens11'

#%%


df_train = pd.read_csv('data/ra/train.csv').drop(columns=['id'])
df_train_org = pd.read_csv('data/ra/calories.csv').drop(columns=['User_ID'])
df_train_org['Sex'] = df_train_org['Gender']
df_train_org = df_train_org.drop(columns=['Gender'])
df_train = pd.concat([df_train, df_train_org], axis=0).reset_index(drop=True)
df_train = df_train.drop_duplicates().reset_index(drop=True)


df_oof = pd.DataFrame(index=df_train.index)

for i, fname in enumerate(name_list):
    df = pd.read_csv(f'oof/{fname}.csv')
    print(fname, df.shape)
    df_oof = pd.concat([df_oof, df], axis=1)

df_test = pd.read_csv('data/ra/test.csv')
df_test2 = pd.DataFrame(index=df_test.index)

for i, fname in enumerate(name_list):
    df = pd.read_csv(f'submissions/{fname}.csv').drop(columns=['id'])
    df_test2 = pd.concat([df_test2, df], axis=1)
df_test2.columns = name_list

# display(df_oof)
# display(df_test2)

# %%
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_log_error


df_y = df_train[['Calories']]
df_X = df_oof.copy()

model = Ridge()
model.fit(df_X,df_y)

y_pred = model.predict(df_X)
rmse_ = root_mean_squared_log_error(df_y, y_pred)
print('RMSE: ', rmse_)

test_pred = model.predict(df_test2)

# %%

df_sub = pd.read_csv('data/ra/sample_submission.csv')
df_sub["Calories"] = test_pred
df_sub.to_csv(f'submissions/{ens_name}.csv', index=False)

df_check = pd.read_csv(f'submissions/{ens_name}.csv')
display(df_check)


#%%

