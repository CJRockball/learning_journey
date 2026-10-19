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
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
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

# Set up dfs
FEATURES = nums + cats 
df_target = df_train[target].copy()
df_X = df_train.loc[:,FEATURES] #.drop(columns=target)
df_X[cats] = df_X[cats].astype('category')
df_test[cats] = df_test[cats].astype('category')


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

# # will make "target" the label 
df_train = update_target_with_survival_probabilities(df_train, time_col='efs_time', event_col='efs')
df_train.drop(columns='km')
df_target['km'] = df_train['km']

#%%

class StdDataset(Dataset):
    def __init__(self, dfX, dfy, num_cols, cat_cols):
        self.cat_features = torch.tensor(dfX.loc[:,cat_cols].values, dtype=torch.long)
        self.num_features = torch.tensor(dfX.loc[:,num_cols].values, dtype=torch.float32)
        self.dfy = torch.tensor(dfy.values, dtype=torch.float32)
        
    def __len__(self):
        return len(self.dfy)
    
    def __getitem__(self, idx):
        cat = self.cat_features[idx]
        num = self.num_features[idx]
        y = self.dfy[idx]
        return [num, cat, y]

class Model(nn.Module):
    def __init__(self, emb_data, num_embeds, num_nums):
        super().__init__()
        # Get embedding
        self.embedding_d = nn.ModuleList([nn.Embedding(car,siz) for car,siz in emb_data])
        for emb in self.embedding_d:
            emb.weight.data.uniform_(-0.01, 0.01)
        
        self.batchnorm = nn.BatchNorm1d(num_nums)
        
        no_in = num_embeds + num_nums
        self.dense1 = nn.Linear(no_in, 256)
        self.dense2 = nn.Linear(256, 256)
        self.dense3 = nn.Linear(256,64)
        self.dp1 = nn.Dropout(0.1)
        self.dp2 = nn.Dropout(0.1)
        self.out = nn.Linear(64, 1)
        

    def forward(self, num_fields, cat_fields):
        # Initialize embedding for respective cat fields
        x1 = [e(cat_fields[:,i]) for i,e in enumerate(self.embedding_d)]
        # Concatenate all embeddings on axis 1
        x1 = torch.cat(x1,1)     
        # Norm the nums
        x2 = self.batchnorm(num_fields)
        # Concat nums and cats
        x1 = torch.cat([x1,x2], axis=1)
        # Dropout for embeddings
        x1 = F.relu(self.dense1(x1))
        x1 = self.dp1(x1)
        x1 = F.relu(self.dense2(x1))
        x1 = self.dp2(x1)
        x1 = F.relu(self.dense3(x1))
        out = self.out(x1)
        
        return out



#%%

def get_embedd_data(df, cats):
    '''Embedding cardinality is a list of two-tuples. First is no of unique values in a cat,
        the second is the number dimensions used to embedd'''
    
    embedding_cardinality = {}
    for cname in df[cats].columns:
        embedding_cardinality[cname] = df[cname].nunique()
    
    emb_sizes = []
    num_embeds = 0
    for cname,value in embedding_cardinality.items():
        emb_size = 1  # min(50, (value+1) // 2 )
        emb_sizes.append((value, emb_size))
        num_embeds += emb_size
  
    return emb_sizes, num_embeds

emb_data, num_embeds = get_embedd_data(df_X, cats)


#%%

df_y = df_target['km']
Xtrain, Xvalid, ytrain, yvalid = train_test_split(df_X, df_y, test_size=0.2, random_state=42)

epochs = 10
BATCH_SIZE = 500
LR = 1e-3
DEVICE = 'cpu'
trainset = StdDataset(Xtrain, ytrain, nums, cats)
validset = StdDataset(Xvalid, yvalid, nums, cats)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE)
validloader = DataLoader(validset, batch_size=BATCH_SIZE)

# %%

num_nums = len(nums)
emb_model = Model(emb_data, num_embeds, num_nums)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(emb_model.parameters(), lr=LR) #, weight_decay=1e-5)

train_list = []
valid_list = []
for epoch in range(epochs):
    train_loss = 0.0
    emb_model.train()
    for data in trainloader:
        Xbatch_nums, Xbatch_cats, y = data[0].to(DEVICE), data[1].to(DEVICE), data[2].to(DEVICE)         

        output = emb_model(Xbatch_nums, Xbatch_cats)
        loss = criterion(output.flatten(), y)
        train_loss += loss.item()
        
        optimizer.zero_grad()    
        loss.backward()
        optimizer.step()
        
    train_mse = (train_loss/len(trainloader))
    train_list.append(train_mse)
    train_loss = 0.0
    
    valid_loss = 0.0
    emb_model.eval()
    for data in validloader:
        Xbatch_nums, Xbatch_cats, y = data[0].to(DEVICE), data[1].to(DEVICE), data[2].to(DEVICE)
        
        y_pred = emb_model(Xbatch_nums, Xbatch_cats)
        loss = criterion(y_pred.flatten(), y)  
        valid_loss += loss.item()

    valid_mse = valid_loss/len(validloader)
    valid_list.append(valid_mse)      
    valid_loss = 0.0
    
    print(f"Epoch: {epoch} Train loss: {train_mse:.5f}, Valid loss: {valid_mse:.5f}")

plt.figure()
plt.plot(range(epochs), train_list, label='Train')
plt.plot(range(epochs), valid_list, label='Valid')
plt.legend()
plt.show()

#%%

print(emb_model.embedding_d[])



# %% Make converter

conv_list = []
conv_dict = {}
for i, cname in enumerate(cats):
    emb_eights = emb_model.embedding_d[i].weight.cpu().detach().tolist()
    temp_dict = {i:dim_[0] for i,dim_ in enumerate(emb_eights)}
    conv_list.append(temp_dict)
    conv_dict[cname] = temp_dict
    

# %%

df_train2 = df_train.copy()
df_test2 = df_test.copy()

for cname in cats:
    converter = conv_dict[cname]
    df_train2[cname] = df_train2[cname].map(converter).astype(np.float32)
    df_test[cname] = df_test[cname].map(converter).astype(np.float32)

# %% Make LGBM model and predict

#Xtrain_L, Xvalid_L, ytrain_L, yvalid_L = train_test_split(df_train2[FEATURES], df_y, test_size=0.2, random_state=42)



params = {
    "boosting_type": "gbdt",
    "objective": "regression",
    "metric": 'mae', #'rmse', #'custom',
    "verbose": -1,
    'random_state': 42,
    'n_jobs':4,
    "feature_pre_filter" : False,
    
    'learning_rate': 0.02,
    'num_leaves': 32,
    'min_child_weight':1,
    'subsample': 0.502742,
    'colsample_bytree': 0.373231,
    'reg_alpha':1, 
    'reg_lambda':1
        }



NUM_ROUNDS = 3000
KFOLD = 5
skf = StratifiedKFold(n_splits=KFOLD, shuffle=True, random_state=42)

for i, (train_index, valid_index) in enumerate(skf.split(df_train2, df_train['race_group'])):
    
    Xtrain_L = df_train2.loc[train_index, FEATURES].copy()    
    ytrain_L = df_y[train_index].copy()
    Xvalid_L = df_train2.loc[valid_index, FEATURES].copy()
    yvalid_L = df_y.loc[valid_index].copy()

    # Make model, train
    
    trn_data = lgb.Dataset(Xtrain_L, ytrain_L) #, categorical_feature=cats)
    lgb_eval = [lgb.Dataset(Xvalid_L, yvalid_L)]

    lgb_model = lgb.train(params, trn_data, NUM_ROUNDS,
                      valid_sets=lgb_eval,
                      callbacks = [
                          lgb.log_evaluation(1000),
                          lgb.early_stopping(stopping_rounds=25) #, verbose=0)
                      ])

    print('Best Fit: ', lgb_model.best_score['valid_0']['l1'], '\n')

    oof = lgb_model.predict(Xvalid_L)
    c_ = get_cindex(df_train2.loc[valid_index,:], oof)
    print(c_)

    if i>0:
        preds += lgb_model.predict(df_test.loc[:,FEATURES]) 
    if i == 0:
        preds = lgb_model.predict(df_test.loc[:,FEATURES])


preds /= KFOLD

# %%


# %%
