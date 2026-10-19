""" Make nn using train/test split
KM target

"""

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
df_X = df_train.loc[:,FEATURES] #.drop(columns=target)
df_X[cats] = df_X[cats].astype('category')
df_test[cats] = df_test[cats].astype('category')


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
KFOLD = 5
kf = KFold(n_splits=KFOLD, shuffle=True, random_state=42)


#%%  ----------------------- MAKE NN --------------------------------
# Torch classes and model

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


class EarlyStopping:
    def __init__(self, patience=1):
        self.patience = patience
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None
        
    def __call__(self, val_loss, model):
        score = val_loss
        if self.best_score is None:
            self.best_score = score
            #self.best_model_state = model.state_dict() # State dict doesn't ork
            torch.save(model.state_dict(), 'models/best.pt')
        elif score >= self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            #self.best_model_state = model.state_dict()
            torch.save(model.state_dict(), 'models/best.pt')
            self.counter = 0
            
    def load_best_model(self, model):
#        model.load_state_dict(self.best_model_state)           
        model_data = torch.load('models/best.pt', weights_only=False)
        model.load_state_dict(model_data)     


# Model

class Model(nn.Module):
    def __init__(self, meta_data, emb_dropout, fc_in_out, dropout_perc, d_out=1):
        super().__init__()
        n_num_cols = meta_data['num_nums']
        emb_sizes = meta_data['emb_sizes']
        # Get embedding
        self.embedding_d = nn.ModuleList([nn.Embedding(car,siz) for car,siz in emb_sizes])
        for emb in self.embedding_d:
            emb.weight.data.uniform_(-0.01, 0.01)
            #nn.init.kaiming_normal_(emb.weight.data)
            
        # Embedding dropout
        self.emb_dropout = nn.Dropout(emb_dropout)
        # Calculate in_features to linear layer
        emb_vector_sum = sum([e.embedding_dim for e in self.embedding_d])
        # Add in_feature to list
        linear_szs = [emb_vector_sum + n_num_cols] + fc_in_out
        
        # Initialize fc layers
        self.fc_layers = nn.ModuleList([nn.Linear(linear_szs[i],linear_szs[i+1])
                                        for i in range(len(linear_szs) - 1)])
        # Output layer
        self.out = nn.Linear(linear_szs[-1],d_out)
        # Initialize Batch Norm 
        self.batchnorm = nn.ModuleList([nn.BatchNorm1d(s) for s in linear_szs[1:]])
        # Batch for num in
        self.batchnorm_num = nn.BatchNorm1d(n_num_cols)
        # Dropout
        self.dropout = nn.ModuleList([nn.Dropout(p) for p in dropout_perc])
    
    
    def forward(self, num_fields, cat_fields):
        # Initialize embedding for respective cat fields
        x1 = [e(cat_fields[:,i]) for i,e in enumerate(self.embedding_d)]
        # Concatenate all embeddings on axis 1
        x1 = torch.cat(x1,1)
        # Dropout for embeddings
        x1 = self.emb_dropout(x1)
        
        # Input normalization for cont fields
        x2 = self.batchnorm_num(num_fields)
        # Concat inputs
        x1 = torch.cat([x1, x2], 1)
        
        for fc, bn, drop in zip(self.fc_layers, self.batchnorm, self.dropout):
            x1 = F.relu(fc(x1))
            x1 = bn(x1)
            x1 = drop(x1)
        
        out = self.out(x1)
        #out = F.sigmoid(x1) #sigmoid as we use BCELoss
        return out

#%% util, train, valid fcn


def get_postsplit_meta(Xtrain, meta_data):
    '''Embedding cardinality is a list of two-tuples. First is no of unique values in a cat,
        the second is the number dimensions used to embedd'''
    # embedding_cardinality = {n: len(c.unique()) for n,c in Xtrain[meta_data['CATS']].items()}
    # emb_sizes = [(size, min(50, (size+1) // 2 )) for item, size in embedding_cardinality.items()]

    embedding_cardinality = {}
    for cname in Xtrain[meta_data['CATS']].columns:
        embedding_cardinality[cname] = Xtrain[cname].nunique()
    
    emb_sizes = []
    in_data = 0
    for cname,value in embedding_cardinality.items():
        emb_size = min(50, (value+1) // 2 )
        emb_sizes.append((value, emb_size))
        in_data += emb_size
        print(cname, value, emb_size)

    meta_data['emb_sizes'] = emb_sizes
    
    return meta_data


def train(model, loader, optimizer, criterion, DEVICE):
    running_loss = 0.0
    model.train()
    for data in loader: #tqdm(loader):
        in1, in2, label = data[0].to(DEVICE), data[1].to(DEVICE), data[2].to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        
        output = model.forward(in1, in2)
        loss = criterion(output.flatten(), label.float()) #torch.flatten(output)
        #loss = torch.sqrt(mse_)
        
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
    training_loss = running_loss/len(loader) #np.sqrt(running_loss/len(loader))
    return training_loss


def valid(model, loader, criterion, DEVICE):
    y_prediction = []
    y_true = []
    running_loss = 0.0
    model.eval()
    for data in loader: #tqdm(loader):
        in1, in2, label = data[0].to(DEVICE), data[1].to(DEVICE), data[2].to(DEVICE)
        
        output = model.forward(in1, in2)
        loss = criterion(output.flatten(), label.float())
        running_loss += loss.item()
        
        y_prediction.append(output.detach().cpu().tolist())
        y_true.append(label.detach().cpu().tolist())
    
    
    # Flatten prediction and labels    
    y_true1 = np.array([v for lst in y_true for v in lst])
    y_prediction1 = np.array([v for lst in y_prediction for v in lst])
    
    validation_loss = running_loss/len(loader) #np.sqrt(running_loss/len(loader))
    validation_rmsle = 1 #mean_absolute_error( y_true1, y_prediction1 )
    return validation_loss, validation_rmsle 


#%% Set up data
def plot_data(train_d, valid_d):
    xx = np.arange(len(train_d))
    plt.figure()
    plt.plot(xx, train_d, label='Train')
    plt.plot(xx, valid_d, label='Validation')
    plt.legend()
    plt.show()
    return


# Make meta-data
CATS = cats
NUM = nums
meta_data = {}
meta_data['NUM'] = NUM
meta_data['CATS'] = CATS
meta_data['num_cats'] = len(CATS)
meta_data['num_nums'] = len(NUM)
# Use category for embedding
# Made sure there are no "new" features in Xvalid. use the full df for meta data
meta_data = get_postsplit_meta(df_X, meta_data)


EPOCHS = 20
LR = 1e-3
BATCH_SIZE = 128
PATIENCE = 3
DEVICE = torch.device('cpu') # 'cuda:0') #

# dfs
df_y = df_target['km'].copy()

#%%
KFOLD = 5
kf = KFold(n_splits=KFOLD, shuffle=True, random_state=42)
start_time = time.time()

oof_nn = np.zeros(len(df_X))
for i, (train_index, val_index) in enumerate(kf.split(df_X)):
    # Set up a ne model every loop
    model = Model(meta_data, 0.1, [256, 256, 64], 3*[0.1]).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR) #, weight_decay=1e-5)
    early_stopping = EarlyStopping(patience=PATIENCE)

    # Set up data
    Xtrain = df_X.loc[train_index].copy()
    ytrain = df_y.loc[train_index].copy()
    Xvalid = df_X.loc[val_index].copy()
    yvalid = df_y.loc[val_index].copy()
    # SET UP DATA standard dataset, dataloader functions
    traindataset = StdDataset(Xtrain, ytrain, meta_data['NUM'], meta_data['CATS'])
    validdataset = StdDataset(Xvalid, yvalid, meta_data['NUM'], meta_data['CATS'])
    trainloader = DataLoader(traindataset, batch_size=BATCH_SIZE)
    validloader = DataLoader(validdataset, batch_size=BATCH_SIZE)

    #Training loop
    train_epoch_list = []
    valid_epoch_list = []
    for epoch in range(EPOCHS):
        train_data = train(model, trainloader, optimizer, criterion, DEVICE)
        validation_loss, validation_rmsle = valid(model, validloader, criterion, DEVICE) #, oof, val_idx)
        print(f'Epoch: {epoch}/{EPOCHS}, Train loss: {train_data:.6f}, Validation loss: {validation_loss:.6f}, Validation mae: {validation_rmsle:.6f}')
        train_epoch_list.append(train_data)
        valid_epoch_list.append(validation_loss)

        early_stopping(validation_loss, model)
    
        if early_stopping.early_stop:
            print("Early stopping")
            break
    early_stopping.load_best_model(model)
    
    # Make oof prediction on validation data
    cat_features = torch.tensor(Xvalid.loc[:,cats].values, dtype=torch.long)
    num_features = torch.tensor(Xvalid.loc[:,nums].values, dtype=torch.float32)    
    oof_nn[val_index] = model(num_features.to(DEVICE), cat_features.to(DEVICE)).cpu().detach().numpy().flatten()
    # Calculate c-index
    m = get_cindex(df_train.iloc[val_index], oof_nn[val_index])
    print(m)

    # Calculate test prediction
    cat_features = torch.tensor(df_test.loc[:,cats].values, dtype=torch.long)
    num_features = torch.tensor(df_test.loc[:,nums].values, dtype=torch.float32)    
    test_pred = model(num_features.to(DEVICE), cat_features.to(DEVICE)).cpu().detach().numpy().flatten()
    
    # Make train file
    if i>0:
        preds += test_pred
    else:
        preds = test_pred
    
preds /= KFOLD
        
#plot_data(train_epoch_list, valid_epoch_list)
# validation_loss, validation_rmsle = valid(model, validloader, criterion, DEVICE)
# print(f'RMSLE: {validation_rmsle}')
m = get_cindex(df_train, oof_nn)
print('Final result',m)
end_time = time.time()
print(f'Total time: {end_time - start_time}')


# %%

print(Xtrain.columns)


# %%
