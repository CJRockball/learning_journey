#%%
import time
import sys
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import itertools


import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
import time

from sklearn.metrics import accuracy_score, confusion_matrix, \
    classification_report, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from captum.attr import IntegratedGradients, LayerConductance, NeuronConductance
from IPython.display import display

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('runs/kags4e7')

def seed_everything(seed=100):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_everything()

#%%

class InsData(Dataset):
    def __init__(self, X, y, X_transform=None):
        self.X = X
        self.y = y
        self.X_transform = X_transform
        print(type(self.X))
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        print(idx)
        print(type(self.X))
        print(self.X.columns)
        
        
        if self.X_transform is not None:
            self.X = self.X_transform.transform(self.X)
            X_data = torch.FloatTensor(self.X[idx,:])
            y_data = torch.LongTensor(self.y.iloc[idx])
            return X_data, y_data
        
        X_data = torch.tensor(self.X.iloc[idx,:].values, dtype=torch.float32)
        y_data = torch.tensor(self.y.iloc[idx], dtype=torch.int32)
        return X_data, y_data


#%%

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, stratify=y)

scaler = StandardScaler()
scaler.fit_transform(Xtrain)

trainset = InsData(Xtrain,ytrain)
testset = InsData(Xtest,ytest)

# %%

trainloader = DataLoader(trainset, batch_size=10)
testloader = DataLoader(testset, batch_size=10)

# %%

dataiter = iter(trainloader)
feature,label = next(dataiter)

print(feature)
print(label)




