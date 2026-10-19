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

# cats = df_train.select_dtypes(include=["object_", 'bool']).columns
# cats = [cat for cat in cats if cat not in target+['ID']]
# nums = df_train.select_dtypes(include=['number']).columns
# nums = [num for num in nums if num not in target+['ID']]


# Replace missing
for cname in cats:
    df_train[cname] = df_train[cname].fillna('Missing')
    df_test[cname] = df_test[cname].fillna('Missing')
for cname in nums:
    df_train[cname] = df_train[cname].fillna(-1)
    df_test[cname] = df_test[cname].fillna(-1)


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
df_X = df_train.loc[:, FEATURES] 
df_X[cats] = df_X[cats].astype(np.int32)
df_test[cats] = df_test[cats].astype(np.int32)


# Kaplan-Meier by race
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

# Calculate km by race, add to df_target 
df_train = update_target_with_survival_probabilities(df_train, time_col='efs_time', event_col='efs')
df_train.drop(columns='km')
df_target['km'] = df_train['km']

KFOLD = 5
kf = KFold(n_splits=KFOLD, shuffle=True, random_state=42)

#%%  Make embedding
# Prep embedding vector
# Make embedding
# visualize t-sne
# try in lgbm

#%%

class Model(nn.Module):
    def __init__(self, emb_data, no_in):
        super().__init__()
        # Get embedding
        self.embedding_d = nn.ModuleList([nn.Embedding(car,siz) for car,siz in emb_data])
        for emb in self.embedding_d:
            emb.weight.data.uniform_(-0.01, 0.01)
        
        no_out = 4*no_in
        self.dense1 = nn.Linear(no_in, no_out)
        self.dense2 = nn.Linear(no_out, no_out)
        self.out = nn.Linear(no_out, 1)
        

    def forward(self, cat_fields):
        # Initialize embedding for respective cat fields
        x1 = [e(cat_fields[:,i]) for i,e in enumerate(self.embedding_d)]
        # Concatenate all embeddings on axis 1
        x1 = torch.cat(x1,1)        
        # Dropout for embeddings
        x1 = F.relu(self.dense1(x1))
        x1 = F.relu(self.dense2(x1))
        out = self.out(x1)
        
        return out



#%%

def get_postsplit_meta(Xtrain, meta_data):
    '''Embedding cardinality is a list of two-tuples. First is no of unique values in a cat,
        the second is the number dimensions used to embedd'''
    embedding_cardinality = {n: len(c.unique()) for n,c in Xtrain[meta_data['CATS']].items()}
    emb_sizes = [(size, min(50, (size+1) // 2 )) for item, size in embedding_cardinality.items()]
    meta_data['emb_sizes'] = emb_sizes
    return meta_data


# Make embeddings
# Test on limited number of features
df_X2 = df_X.iloc[:,2:]
display(df_X2)

embedding_cardinality = {}
for cname in df_X2.columns:
    embedding_cardinality[cname] = df_X2[cname].nunique()
    
emb_data = []
in_data = 0
for cname,value in embedding_cardinality.items():
    emb_size = min(50, (value+1) // 2 )
    emb_data.append((value, emb_size))
    in_data += emb_size


# print(embedding_cardinality)
# print(emb_data)
# print(in_data)

# meta_data = {}
# meta_data['CATS'] = list(df_X2.columns)
# meta_data = get_postsplit_meta(df_X2, meta_data)
# print(meta_data['emb_sizes'])

#%%

df_y = df_target['km']

Xtrain, Xvalid, ytrain, yvalid = train_test_split(df_X, df_y, test_size=0.2, random_state=42)

Xtrain_tens = torch.tensor(Xtrain.values, dtype=torch.long)
ytrain_tens = torch.tensor(ytrain.values, dtype=torch.float32)
Xvalid_tens = torch.tensor(Xvalid.values, dtype=torch.long)
yvalid_tens = torch.tensor(yvalid.values, dtype=torch.float32)
# %%

epochs = 10
batch_size = 500
LR = 1e-3

emb_model = Model(emb_data, in_data)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(emb_model.parameters(), lr=LR) #, weight_decay=1e-5)


for i in range(epochs):
    train_loss = 0.0
    emb_model.train()
    for j in range(0, len(Xtrain), batch_size):
        Xbatch = Xtrain_tens[j:j+batch_size, cats]
        ybatch = ytrain_tens[j:j+batch_size]
        
        output = emb_model(Xbatch)
        loss = criterion(output.flatten(), ybatch)
        train_loss += loss.item()
        
        loss.backward()
        optimizer.step()
    train_mse = (train_loss/batch_size)
    
    
    emb_model.eval()
    y_pred = emb_model(Xvalid_tens)
    mse_v = criterion(y_pred, yvalid_tens)    
    print(f" Train mse: {train_mse}, valid mse: {mse_v}")
    



# %%


