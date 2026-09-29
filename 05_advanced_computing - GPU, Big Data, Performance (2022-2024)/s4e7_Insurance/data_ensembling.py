#%%
import pandas as pd

df1 = pd.read_csv('data/sub_nn1.csv')
df1.columns = ['id', 'Response_nn1']
display('df1', df1.head(3))

df2 = pd.read_parquet('data/sub_xgb_full_train.parquet')
display('df2', df2.head(3))

df3 = pd.read_parquet('data/sub_nn_emb_fulldata.parquet')
df3.columns = ['id', 'Response_emb']
display('df3', df3.head(3))

# %%

df = pd.concat([df1, df2.Response_xgb, df3.Response_emb], axis=1)
display(df.head())


# %%

df['Response'] = (0.6*df.Response_emb + 0.2*df.Response_nn1 + 0.2* df.Response_xgb)
display(df.head())


# %%
import numpy as np

df.drop(columns=['Response_nn1', 'Response_xgb', 'Response_emb'], inplace=True)
df["Response"] = df['Response'].astype(np.float32)
print(df.info())


#%%
df.to_parquet('data/sub_tri_comb.parquet')


# %%
