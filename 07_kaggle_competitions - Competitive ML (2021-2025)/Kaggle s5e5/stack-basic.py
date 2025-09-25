#%%
import pandas as pd
import numpy as np

name_list = ['ens6', 'stack6'] #['xgb_lgb1', 'xgb_lgb2', 'xgb_lgb3'] #,'xgb13', 'lgb3', 'cat2', 'ridge2', 'xgb17', 'lgb5', 'knn2']
ens_name = 'stack7'

#%%


df_test = pd.read_csv('data/ra/test.csv')
df_test2 = pd.DataFrame(index=df_test.index)

for i, fname in enumerate(name_list):
    df = pd.read_csv(f'submissions/{fname}.csv').drop(columns=['id'])
    df = np.log1p(df)
    df_test2 = pd.concat([df_test2, df], axis=1)
df_test2.columns = name_list

# display(df_oof)
# display(df_test2)

# %%
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_log_error

num_feat = len(name_list)
df_test2['mean_pred'] = 0

for cname in name_list:
    df_test2['mean_pred'] += 1/num_feat * df_test2[cname]

display(df_test2)


# %%

df_sub = pd.read_csv('data/ra/sample_submission.csv')
df_sub["Calories"] = np.expm1(df_test2['mean_pred'].to_numpy())
df_sub.to_csv(f'submissions/{ens_name}.csv', index=False)

df_check = pd.read_csv(f'submissions/{ens_name}.csv')
display(df_check)


#%%


# %%
