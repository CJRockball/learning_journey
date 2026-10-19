#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os
#from tqdm import tqdm
import random
from IPython.display import display

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import gc
torch.cuda.empty_cache()
gc.collect()

import ctypes
# Function to clean RAM & vRAM
def clean_memory():
    gc.collect()
    ctypes.CDLL("libc.so.6").malloc_trim(0)
    torch.cuda.empty_cache()

clean_memory()


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_everything()

import time
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from catboost import CatBoostRegressor, Pool
import lightgbm as lgb
import xgboost

from scipy.stats import rankdata 
import pandas.api.types
from lifelines import KaplanMeierFitter, CoxPHFitter, NelsonAalenFitter
from lifelines.utils import concordance_index



class ParticipantVisibleError(Exception):
    pass


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float: 
    del solution[row_id_column_name]
    del submission[row_id_column_name]
    
    event_label = 'efs'
    interval_label = 'efs_time'
    prediction_label = 'prediction'
    for col in submission.columns:
        if not pandas.api.types.is_numeric_dtype(submission[col]):
            raise ParticipantVisibleError(f'Submission column {col} must be a number')
    # Merging solution and submission dfs on ID
    merged_df = pd.concat([solution, submission], axis=1)
    merged_df.reset_index(inplace=True)
    merged_df_race_dict = dict(merged_df.groupby(['race_group']).groups)
    metric_list = []
    for race in merged_df_race_dict.keys():
        # Retrieving values from y_test based on index
        indices = sorted(merged_df_race_dict[race])
        merged_df_race = merged_df.iloc[indices]
        # Calculate the concordance index
        c_index_race = concordance_index(
                        merged_df_race[interval_label],
                        -merged_df_race[prediction_label],
                        merged_df_race[event_label])
        metric_list.append(c_index_race)
    return float(np.mean(metric_list)-np.sqrt(np.var(metric_list)))

def get_cindex(df, oof):
    # Get c-index
    y_true = df[["ID","efs","efs_time","race_group"]].copy()
    y_pred = df[["ID"]].copy()
    y_pred["prediction"] = oof 
    m = score(y_true.copy(), y_pred.copy(), "ID")
    return m

#%% Get data

df_train = pd.read_csv('data/ra/train.csv')
df_test = pd.read_csv('data/ra/test.csv')

# Define categories
target =['efs', 'efs_time']
nums = ['donor_age', 'age_at_hct'] #, 'efs_time']
cats = [name for name in df_train.columns if name not in (nums + target+['ID'])]

# Replace missing
for cname in cats:
    df_train[cname] = df_train[cname].fillna('Missing')
    df_test[cname] = df_test[cname].fillna('Missing')

for cname in nums:
    df_train[cname] = df_train[cname].fillna(-1)
    df_test[cname] = df_test[cname].fillna(-1)

# # Fill na for nums. Put mean.    
# add_cols = []
# for cname in nums:
#     print(f'fillna for {cname}')
#     # Mark all cols ith nans
#     if df_train[cname].isnull().values.any():
#         print('Add nan col')
#         df_train[f"is_{cname}_na"] = df_train[cname].isna().astype(int)
#         df_test[f"is_{cname}_na"] = df_test[cname].isna().astype(int)
#         add_cols.append(f"is_{cname}_na")

#     # Fill all nans ith mean
#     m = df_train[cname].mean()
#     df_train[cname] = df_train[cname].fillna(m)
#     df_test[cname]  = df_test[cname].fillna(m)
# cats = cats + add_cols


# Normalize nums
num_m = df_train[nums].mean()
num_s = df_train[nums].std()
df_train[nums] = (df_train[nums] - num_m) / num_s
df_test[nums] = (df_test[nums] - num_m) / num_s


# Ordinal encoding
def ord_trans(df, df1, features):
    for cname in features:
        train_categories = df[cname].value_counts().sort_values().index.get_level_values(0).to_list()
        train_dict = {fname:i for i, fname in enumerate(train_categories)}

        test_categories = df1[cname].unique()
        test_dict = {}
        for fname in test_categories:
            if fname in train_categories:
                test_dict[fname] = train_dict[fname]
            elif df[cname].isnull().any(): 
                test_dict[fname] = train_dict['Missing']
            else:
                test_dict[fname] = 0    
    
        df[cname] = df[cname].map(train_dict).astype(np.int16)
        df1[cname] = df1[cname].map(test_dict).astype(np.int16)       
    return df, df1

df_train, df_test = ord_trans(df_train, df_test, cats)

# Set up dfs
FEATURES = nums + cats 
df_target = df_train[target].copy()
#df_test[cats] = df_test[cats].astype(np.int32)


#%% Kaplan-Meier by race

def transform_survival_probability(df, time_col='efs_time', event_col='efs'):
    kmf = KaplanMeierFitter()
    kmf.fit(df[time_col], event_observed=df[event_col])
    survival_probabilities = kmf.survival_function_at_times(df[time_col]).values.flatten()
    return survival_probabilities

def update_target_with_survival_probabilities(df, time_col='efs_time', event_col='efs'):
    race_group = sorted(df['race_group'].unique())
    survival_probs_dict = {}
    for race in race_group:
        race_df = df[df['race_group'] == race]
        survival_probs_dict[race] = transform_survival_probability(race_df, time_col, event_col)
    for race in race_group:
        df.loc[df['race_group'] == race, 'km'] = survival_probs_dict[race]
    df.loc[df[event_col] == 0, 'km'] -= 0.18
    
    return df

# # will make "target" the label 
df_train = update_target_with_survival_probabilities(df_train, time_col='efs_time', event_col='efs')
df_train.drop(columns='km')
df_target['km'] = df_train['km']

# Set up dfs for torch
df_y = df_target['km'].copy()

df_X = df_train.copy()
df_X[cats] = df_X[cats].astype(np.int32)

Xtrain, Xvalid, ytrain, yvalid = train_test_split(df_X, df_y, test_size=0.2, random_state=42)

Xtrain_tens = torch.tensor(Xtrain.loc[:,cats].values, dtype=torch.long)
ytrain_tens = torch.tensor(ytrain.values, dtype=torch.float32)
Xvalid_tens = torch.tensor(Xvalid.loc[:,cats].values, dtype=torch.long)
yvalid_tens = torch.tensor(yvalid.values, dtype=torch.float32)

Xtrain_num = torch.tensor(Xtrain.loc[:,nums].values, dtype=torch.float32)
Xvalid_num = torch.tensor(Xvalid.loc[:,nums].values, dtype=torch.float32)



#%%
"""
Make tabtransformer
- embedd, num - scale/batchnorm
- Multi headed attention
multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
attn_output, attn_output_weights = multihead_attn(query, key, value)
- Add/Norm
x1 = torch.cat([x1, x2], 1)
x1 = bn(x1)
- Forard
nn.Linear(multi-head in, 8xmultihead-in)
GEGLU()
nn.Linear(8xmultihead-in, multihead-in)
- Add/Norm
x1 = torch.cat([x1, x2], 1)
- Concat ith numerical

x1 = bn(x1)
x1 = torch.cat([x1, x2], 1)
MLP



"""

class NumIn(nn.Module):
    def __init__(self, num_features=2, ):
        super().__init__()
        self.net = nn.Sequential(
                torch.nn.LayerNorm(num_features),
                )

    def forward(self, x):
        return self.net(x)
       

class Embedding(nn.Module):
    def __init__(self, emb_sizes, emb_dropout=0.1):
        super().__init__()
        # Get embedding
        self.embedding_d = nn.ModuleList([nn.Embedding(car,siz) for car,siz in emb_sizes])
        for emb in self.embedding_d:
            emb.weight.data.uniform_(-0.01, 0.01)
            
        # Embedding dropout
        self.emb_dropout = nn.Dropout(emb_dropout)

    
    def forward(self, cat_fields):
        # Initialize embedding for respective cat fields
        x1 = [e(cat_fields[:,i]) for i,e in enumerate(self.embedding_d)]
        # Concatenate all embeddings on axis 1
        x1 = torch.cat(x1,1)
        # Dropout for embeddings
        x1 = self.emb_dropout(x1)
        return x1


class FeedForward(nn.Module):
    def __init__(self, emb_sizes, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_sizes, emb_sizes * mult * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_sizes * mult * 2, emb_sizes)
        )

    def forward(self, x):
        return self.net(x)

class MLP(nn.Module):
    def __init__(self, emb_sizes, num_features):
        super().__init__()
        self.in_features = emb_sizes + num_features
        self.mlp_ = nn.Sequential(
            nn.Linear(self.in_features, self.in_features),
            nn.ReLU(),
            nn.Linear(self.in_features, self.in_features),
            nn.ReLU(),
            nn.Linear(self.in_features, 1),
        )
 
    def forward(self,x):
        return self.mlp_(x)


class Attention(nn.Module):
    def __init__(self, embed_sizes, num_heads=5, d=0.1):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(embed_sizes, num_heads, dropout=d)
        self.feed_forward = FeedForward(embed_sizes)
        self.norm1 = nn.LayerNorm(embed_sizes)
        self.norm2 = nn.LayerNorm(embed_sizes)
        self.dropout = nn.Dropout(d)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src1, _ = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        src2 = src + self.dropout(src1)
        src3 = self.norm1(src2)
        
        src3 = self.feed_forward(src3)
        src4 =  src2 + self.dropout(src2)
        src4 = self.norm2(src4)
        
        return src4


class Tabtransformer(nn.Module):
    def __init__(self, emb_data, embed_sizes, num_features):
        super().__init__()
        self.embed = Embedding(emb_data)
        self.num = NumIn(num_features)
        self.atten = Attention(embed_sizes)
        self.mlp_ = MLP(embed_sizes, num_features)
        
    def forward(self,x_cat, x_num):
        emb_data = self.embed(x_cat)
        att_data = self.atten(emb_data)
        num_data = self.num(x_num)
        x = torch.cat([att_data, num_data], 1)    
        out = self.mlp_(x)
        
        return out

#%% Training function


def train_(Xtrain_, Xvalid_, ytrain_, yvalid_, model):
    n_epochs = 10
    batch_size = 500

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_loss_num = 0.0
    train_loss = []
    for epoch in range(n_epochs):
        for i in range(0, len(Xtrain_), batch_size):
          
            Xbatch = Xtrain_[i:i+batch_size]
            y_pred = model(Xbatch)
            ybatch = ytrain_[i:i+batch_size]
            loss = loss_fn(y_pred, ybatch)
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            train_loss_num += loss.item()
        
        train_loss_batch = train_loss_num/batch_size
        train_loss.append(train_loss_batch)
        print(f'Finished epoch {epoch}, latest loss {train_loss_batch}')
        train_loss_num = 0.0
    
    plt.figure()
    plt.plot(list(range(n_epochs)), train_loss)
    plt.show()

#%% Test ind sequences
# 1. Test embedding

def get_embedding_data(Xtrain, cats_):
    '''Embedding cardinality is a list of two-tuples. First is no of unique values in a cat,
        the second is the number dimensions used to embedd'''
    # embedding_cardinality = {n: len(c.unique()) for n,c in Xtrain[meta_data['CATS']].items()}
    # emb_sizes = [(size, min(50, (size+1) // 2 )) for item, size in embedding_cardinality.items()]

    embedding_cardinality = {}
    for cname in Xtrain[cats_].columns:
        embedding_cardinality[cname] = Xtrain[cname].nunique()
    
    emb_sizes = []
    count_embs = 0
    for cname,value in embedding_cardinality.items():
        emb_size = min(50, (value+1) // 2 )
        emb_sizes.append((value, emb_size))
        count_embs += emb_size
        
    return emb_sizes, count_embs


emb_data, count_embs = get_embedding_data(Xtrain, cats)
num_features = 2   

#%% Train.
# Get the embedding layer and add a linear on the output to test backprop


class EMB(nn.Module):
    def __init__(self, emb_data, count_embs):
      super().__init__()
      
      self.first_model = nn.Sequential(
                                      Embedding(emb_data),
                                      nn.Linear(count_embs, 1)
                                    )

    def forward(self, x):
      out = self.first_model(x)
      return out


emb_model = EMB(emb_data, count_embs)

train_(Xtrain_tens, Xvalid_tens, ytrain_tens, yvalid_tens, emb_model)

#%% Try Feed Forard

class TestFF(nn.Module):
    def __init__(self, emb_data, count_embs):
        super().__init__()
        self.ff = nn.Sequential(
                                Embedding(emb_data),
                                FeedForward(count_embs),
                                nn.Linear(count_embs, 1)
        )
    
    def forward(self, x):
        out = self.ff(x)
        return out

testff_model = TestFF(emb_data, count_embs)

train_(Xtrain_tens, Xvalid_tens, ytrain_tens, yvalid_tens, testff_model)


# %% Test attention

class TestAttn(nn.Module):
    def __init__(self, emb_data, count_embs):
        super().__init__()
        self.ff = nn.Sequential(
                                Embedding(emb_data),
                                Attention(count_embs),
                                nn.Linear(count_embs, 1)
        )
    
    def forward(self, x):
        out = self.ff(x)
        return out

testattn_model = TestAttn(emb_data, count_embs)

train_(Xtrain_tens, Xvalid_tens, ytrain_tens, yvalid_tens, testattn_model)

# %% Test numerical features input

class TestNum(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.num_ = nn.Sequential(
                                  NumIn(num_features),
                                  nn.Linear(num_features, 1)
        )
    
    def forward(self, x):
        out = self.num_(x)
        return out

testnum_model = TestNum(num_features)

train_(Xtrain_num, Xvalid_num, ytrain_tens, yvalid_tens, testnum_model)

# %% Test MLP


class TestMLP(nn.Module):
    def __init__(self, emb_data, embed_sizes, num_features):
        super().__init__()
        self.emb = Embedding(emb_data)
        self.num_ = NumIn(num_features)
        self.mlp_ = MLP(embed_sizes, num_features)
        
    def forward(self, x_cat, x_num):
        x1 = self.emb(x_cat)
        x2 = self.num_(x_num)
        x3 = torch.cat([x1, x2], axis=1)
        out = self.mlp_(x3)
        return out


def train2_(Xtrain_, Xtrain_num,  Xvalid_, ytrain_, yvalid_, model):
    n_epochs = 10
    batch_size = 500

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_loss_num = 0.0
    train_loss = []
    for epoch in range(n_epochs):
        for i in range(0, len(Xtrain_), batch_size):
          
            Xbatch = Xtrain_[i:i+batch_size]
            Xbatch_num = Xtrain_num[i:i+batch_size]
            y_pred = model(Xbatch, Xbatch_num)
            ybatch = ytrain_[i:i+batch_size]
            loss = loss_fn(y_pred, ybatch)
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            train_loss_num += loss.item()
        
        train_loss_batch = train_loss_num/batch_size
        train_loss.append(train_loss_batch)
        print(f'Finished epoch {epoch}, latest loss {train_loss_batch}')
        train_loss_num = 0.0
    
    plt.figure()
    plt.plot(list(range(n_epochs)), train_loss)
    plt.show()


mlp_model = TestMLP(emb_data, count_embs, num_features)

train2_(Xtrain_tens, Xtrain_num,  Xvalid_tens, ytrain_tens, yvalid_tens, mlp_model)    

# %% Run TabTransformer

class TabTransformer(nn.Module):
    def __init__(self, emb_data, embed_size, num_features):
        super().__init__()
        self.tab = Tabtransformer(emb_data, embed_size, num_features)
        
    def forward(self, x_cat, x_num):
        out = self.tab(x_cat, x_num)
        return out
    


tab_model = TabTransformer(emb_data, count_embs, num_features)

train2_(Xtrain_tens, Xtrain_num,  Xvalid_tens, ytrain_tens, yvalid_tens, tab_model)    



