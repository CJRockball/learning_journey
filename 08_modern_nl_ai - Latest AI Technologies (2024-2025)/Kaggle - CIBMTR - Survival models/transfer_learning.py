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

# Set up dfs
FEATURES = nums + cats 
df_target = df_train[target].copy()
# # will make "target" the label 
df_train = update_target_with_survival_probabilities(df_train, time_col='efs_time', event_col='efs')
df_train.drop(columns='km')
df_target['km'] = df_train['km']

# Mod efs
df_train['efs_mod'] = df_train['efs']
df_train['efs_mod'] = df_train['efs_mod'].replace({0:-1})


# Set up dfs for torch
df_y = df_target['km'].copy()  # df_train['efs_time'].copy() #
df_X = df_train.copy()
df_X[cats] = df_X[cats].astype(np.int32)
df_test[cats] = df_test[cats].astype(np.int32)

print(df_X.loc[:,FEATURES].shape)



#%%
from torchvision import models

# Load ResNet model
model = models.resnet18(pretrained=True)

# Modify architecture for tabular task
number_of_tab_features = 1
model.fc = nn.Linear(model.fc.in_features, number_of_tab_features)

print(model)
#%%

Xtrain, Xvalid, ytrain, yvalid = train_test_split(df_X.loc[:,FEATURES], df_y, test_size=0.2, random_state=42)

tensor_x = torch.tensor(Xtrain.values, dtype=torch.float32)
tensor_y = torch.tensor(ytrain.values, dtype=torch.float32)

trainloader = DataLoader(list(zip(tensor_x, tensor_y)), batch_size=32)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# %%

for epoch in range(2):
    for x_batch, y_batch in trainloader:
        optimizer.zero_grad()
        output = model(x_batch)
        loss = criterion(output.flatten(),y_batch)
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch}, Loss: {loss.item()}')



# %%
