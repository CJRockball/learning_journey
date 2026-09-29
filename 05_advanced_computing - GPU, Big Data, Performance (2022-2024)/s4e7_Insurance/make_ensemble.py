#%%

import pandas as pd
import numpy as np

df_nn = pd.read_parquet('data/submissions/sub_nn_emb_fd_base.parquet')
df_nn.columns = ['id', 'base_response']
df_pi = pd.read_parquet('data/submissions/nn_emb_fd_PIsplit.parquet')
df_pi.columns = ['id', 'pi_response']
df_vd = pd.read_parquet('data/submissions/nn_emb_fd_VDsplit.parquet')
df_vd.columns = ['id', 'vd_response']

#%%

display(df_nn)

#%%

df_comb = pd.concat([df_nn, df_pi.pi_response, df_vd.vd_response], axis=1)

display(df_comb.head())

# %%

df_comb['Response'] =  0.6*df_comb.base_response + 0.2*df_comb.pi_response + 0.2*df_comb.vd_response
display(df_comb.head())

# %%

sub_comb = df_comb[['id', 'Response']]

display(sub_comb.head())

sub_comb.to_parquet('data/submissions/sub_comb_base_pi_vd.parquet')

# %%

df_check = pd.read_parquet('data/submissions/sub_comb_base_pi_vd.parquet')
display(df_check.head())

# %%
