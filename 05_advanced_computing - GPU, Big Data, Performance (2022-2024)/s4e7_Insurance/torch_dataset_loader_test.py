#%%
import time
import sys
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from line_profiler import profile

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder

from captum.attr import IntegratedGradients, LayerConductance, NeuronConductance
from IPython.display import display

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset, IterableDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('runs/kags4e7')

import gc
torch.cuda.empty_cache()
gc.collect()

def seed_everything(seed=100):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_everything()
os.chdir('/home/patrick/Python/timeseries/weather/kaggle/Insurance_s4e7')


#%% DataSet, DataLoader

class TestDataset(Dataset):
    def __init__(self, dfX, dfy):
        self.dfy = dfy.values
        self.dfX = dfX.values
         
    def __len__(self):
        return len(self.dfy)
    
    @profile
    def __getitem__(self,idx, batch_size):
        X = torch.tensor(self.dfX[idx:idx+batch_size,:], dtype=torch.float32)
        y = torch.tensor(self.dfy[idx:idx+batch_size,:], dtype=torch.long)
        return [X, y]

class FastDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    Source: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """
    def __init__(self, ds, batch_size=32):
        """
        Initialize a FastTensorDataLoader.

        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.

        :returns: A FastTensorDataLoader.
        """
        self.ds = ds
        self.dataset_len = ds.__len__()
        self.batch_size = batch_size

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches
        
    def __iter__(self):
        self.i = 0
        return self

    @profile
    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = self.ds.__getitem__(self.i, self.batch_size)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches

#%% Model class

class Model(nn.Module):
    def __init__(self, in_features, h1=20, h2=30,d_out=2):
        super().__init__()
        
        self.fc1 = nn.Linear(in_features, h1)
        self.relu1 = nn.SiLU()
        #self.bn1 = nn.BatchNorm1d(h1)
        self.dp1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(h1, h2)
        self.relu2 = nn.SiLU()
        #self.bn2 = nn.BatchNorm1d(h2)
        self.dp2 = nn.Dropout(0.1)

        self.out = nn.Linear(h2,d_out)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        #x = self.bn1(x)
        x = self.dp1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        #x = self.bn2(x)
        x = self.dp2(x)
        x = self.out(x)
        return x    
    

#%%
@profile
def load_data():
    
    list_int32 = ['Vintage','Policy_Sales_Channel','Region_Code']
    list_bool = ['Driving_License', 'Previously_Insured', 'Vehicle_Damage', 'Gender']
    list_float32 = ['Annual_Premium','Age']
    list_target = ['Response']

    # # Load, shuffle get sample
    # print('Read in')
    # df_in = pd.read_parquet('data/train_proc.parquet')
    # #X = pd.read_parquet('data/train_extdata.parquet')
    # print('shuffle')
    # #X = X.sample(frac=1)
    # df_in_y = df_in.pop('Response')
    # X, _,y,_ = train_test_split(df_in, df_in_y, stratify = df_in_y)

    # n = 10000
    BATCH_SIZE = 100
    # X = df_in.iloc[:n,:]
    # y = df_in_y.iloc[:n].to_frame()
    # del df_in, df_in_y
    # #y = y.to_frame()
    
    X = pd.read_parquet('data/train_proc_small.parquet')
    X[list_bool] = X[list_bool].astype(np.int8)
    y = pd.read_parquet('data/ytrain_proc_small.parquet')
    y = y.astype(np.int8)
    print('X shape', X.shape)
    
    # Split
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, train_size=0.9, random_state=27, stratify=y)
    in_feat = Xtrain.shape[1]
    class_no = ytrain.value_counts()/len(ytrain)
    del X, y
    
    # Normalize
    scaler = StandardScaler()
    Xtrain.loc[:,list_float32] = scaler.fit_transform(Xtrain.loc[:,list_float32])
    Xtest.loc[:,list_float32] = scaler.transform(Xtest.loc[:,list_float32])

    # Make dataset
    train_ds = TestDataset(Xtrain, ytrain)
    test_ds = TestDataset(Xtest, ytest)

    # Maket Loader
    trainloader = FastDataLoader(train_ds, batch_size=BATCH_SIZE)
    testloader = FastDataLoader(test_ds, batch_size=BATCH_SIZE)
    return trainloader, testloader, in_feat, class_no, Xtest, ytest

#%%
@profile
def train_nn(trainloader, testloader, in_feat, class_no, Xtest, ytest):
    epochs = 100
    lr = 0.003
    
    # Device selection
    device = torch.device('cpu') #('cuda:0' if torch.cuda.is_available() else 'cpu')
    Xtest = torch.tensor(Xtest.values, dtype=torch.float32).to(device)
    ytest = torch.tensor(ytest.values, dtype=torch.long).to(device)
    
    # Def model
    in_features, h1, h2 = in_feat, 2*in_feat, in_feat
    model = Model(in_features, h1, h2).to(device).to(device)
    #print(model.parameters)
    # Set metric (criterion) and choose optimizer
    
    class_weights = torch.FloatTensor(1 - class_no.values).to(device)
    criterion = nn.CrossEntropyLoss(weight = class_weights).to(device)
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) #, weight_decay=w_decay)

    # Training Loop
    train_loss_per_epoch = []
    test_loss_per_epoch = []   
    for epoch in range(epochs):

        model.train() 
        running_loss = 0.0
        for i, (features,label) in enumerate(trainloader):
            # Send data to gpu if used
            features = features.to(device)
            label = label.to(device)

            # Reset gradient 
            optimizer.zero_grad()

            # Forward, backward, optimize
            outputs = model.forward(features) 
            loss = criterion(outputs, torch.flatten(label))
            loss.backward()
            optimizer.step()
            
            # Metrics
            running_loss += loss.item()

        train_loss_per_epoch.append(running_loss / (i+1))    
        running_loss = 0.0

        
        model.eval()
        # running_test_loss = 0.0
        # for i, (features, label) in enumerate(testloader):
        #     features = features.to(device)
        #     label = label.to(device)
            
        #     y_pred = model.forward(features)
        #     loss = criterion(y_pred, torch.flatten(label)).detach()
        #     running_test_loss += loss.item()
            
        # Metric batch
        y_test_pred = model.forward(Xtest)
        test_loss = criterion(y_test_pred, torch.flatten(ytest)).detach()    
        test_loss_per_epoch.append(test_loss)
        # Metric loader
        # test_loss_per_epoch.append(running_test_loss / (i+1))
        # running_test_loss = 0.0

        # Print
        if epoch % 10 == 0:
            print(f"Ep: {epoch} Loss Training: {train_loss_per_epoch[-1]} Test: {test_loss_per_epoch[-1]}")
    return         
    
# %%

if __name__ == '__main__':
    s_time = time.time()
    print('Loading')
    trainloader, testloader, in_feat, class_no, Xtest, ytest = load_data()    
    m_time = time.time()
    print('Load time: ', m_time-s_time)
    print("Training")
    train_nn(trainloader, testloader, in_feat, class_no, Xtest, ytest)
    e_time = time.time()
    print('Train time: ', e_time - m_time)
    print('Total time', e_time - s_time)