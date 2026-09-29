#%%
import time
import sys
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from line_profiler import profile

import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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

#%%

# Good single line
# class InsData(Dataset):
#     def __init__(self, X, y):
#         self.X = torch.tensor(X, dtype=torch.float32)
#         self.y = torch.tensor(y, dtype=torch.long)
        
#     def __len__(self):
#         return len(self.y)
    
#     @profile
#     def __getitem__(self, idx):
#         features = self.X[idx]
#         target = self.y[idx]  
#         return features, target

#%%

class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    Source: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """
    def __init__(self, *tensors, batch_size=32, shuffle=False):
        """
        Initialize a FastTensorDataLoader.

        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.

        :returns: A FastTensorDataLoader.
        """
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches
    def __iter__(self):
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = tuple(t[self.i:self.i+self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches



class Model(nn.Module):
    def __init__(self, fc_in_out, dropout_perc, d_out=2):
                 #in_features, h1=20, h2=30,d_out=2):
        super().__init__()
        
        # self.fc1 = nn.Linear(in_features, h1)
        # self.relu1 = nn.SiLU()
        # #self.bn1 = nn.BatchNorm1d(h1)
        # self.dp1 = nn.Dropout(0.2)
        # self.fc2 = nn.Linear(h1, h2)
        # self.relu2 = nn.SiLU()
        # #self.bn2 = nn.BatchNorm1d(h2)
        # self.dp2 = nn.Dropout(0.1)

        # Initialize fc layers
        self.fc_layers = nn.ModuleList([nn.Linear(fc_in_out[i],fc_in_out[i+1])
                                        for i in range(len(fc_in_out) - 1)])
        # Output layer
        self.out = nn.Linear(fc_in_out[-1],d_out)
        # Initialize Batch Norm 
        self.batchnorm = nn.ModuleList([nn.BatchNorm1d(s) for s in fc_in_out[1:]])
        # Dropout
        self.dropout = nn.ModuleList([nn.Dropout(p) for p in dropout_perc])

    def forward(self, x):
        # x = self.fc1(x)
        # x = self.relu1(x)
        # #x = self.bn1(x)
        # x = self.dp1(x)
        # x = self.fc2(x)
        # x = self.relu2(x)
        # #x = self.bn2(x)
        # x = self.dp2(x)
        
        for fc, bn, drop in zip(self.fc_layers, self.batchnorm, self.dropout):
            x = F.silu(fc(x))
            x = bn(x)
            x = drop(x)
        
        x = self.out(x)
        return x       

#%%
# Load, reduce data
# Split data
# Create datasets with transformations

@profile
def load_data():
    # Load, shuffle get sample
    print('Read in')
    df_in = pd.read_parquet('data/train_proc.parquet')
    print('shuffle')
    #X = X.sample(frac=1)
    df_in_y = df_in.pop('Response')
    n = 1000000
    X, _,y,_ = train_test_split(df_in, df_in_y, train_size= n, 
                                random_state=27, stratify = df_in_y)

    BATCH_SIZE = 10000
    del df_in, df_in_y
    y = y.to_frame()


    # Add features
    list_int32 = ['Vintage','Policy_Sales_Channel','Region_Code']
    list_bool = ['Driving_License', 'Previously_Insured', 'Vehicle_Damage', 'Gender']
    list_float32 = ['Annual_Premium','Age']
    list_category = ['Vehicle_Age', 'Vintage','Policy_Sales_Channel','Region_Code']
    list_target = ['Response']
    
   
    # Calc imbalance
    class_no = pd.DataFrame(y.to_numpy()).value_counts()/len(y)
    #print(class_no)
    no_features_in = X.shape[1]
    
    # Split data
    print('Split data')
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.1, stratify=y)

    print('Scaler')
    scaler = StandardScaler()
    Xtrain_norm = scaler.fit_transform(Xtrain)
    Xtrain_norm = pd.DataFrame(Xtrain_norm, columns=Xtrain.columns)
    Xtest_norm = scaler.transform(Xtest)
    Xtest_norm = pd.DataFrame(Xtest_norm, columns=Xtest.columns)

    # trainset = InsData(Xtrain_norm.values,ytrain.values)
    # testset = InsData(Xtest_norm.values,ytest.values)
    
    # Make tensors for fasttensordataloader
    
    train_x = torch.tensor(Xtrain_norm.values, dtype=torch.float32) 
    test_x = torch.tensor(Xtest_norm.values, dtype=torch.float32) 
    train_y = torch.tensor(ytrain.values, dtype=torch.long)
    test_y = torch.tensor(ytest.values, dtype=torch.long)
    
    trainloader = FastTensorDataLoader(train_x, train_y, batch_size=BATCH_SIZE, shuffle=False)  
    testloader = FastTensorDataLoader(test_x, test_y, batch_size=BATCH_SIZE, shuffle=False)
    
    # trainloader = DataLoader(trainset, batch_size=1024, num_workers=4, \
    #     pin_memory=True, shuffle=False)
    # testloader = DataLoader(testset, batch_size=1024, num_workers=4, \
    #     pin_memory=True, shuffle=False)

    return no_features_in, trainloader, testloader, class_no, Xtest_norm, \
        Xtrain, Xtest, ytrain, ytest
    
# Test the dataloader
# dataiter = iter(trainloader)
# feature,label = next(dataiter)

# print('Features', feature.shape) #, '\n', feature)
# print('labels', label.shape) #, '\n', label)

# %%

@profile
def train(epochs, no_features_in, trainloader, testloader, class_no, lr, \
          Xtest_norm, ytest):
    
    Xtest_norm = torch.FloatTensor(Xtest_norm.values)
    ytest = torch.flatten(torch.LongTensor(ytest.values))
    # Device selection
    device = torch.device('cuda:0') #('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Def model
    in_features, h1, h2 = no_features_in, 2*no_features_in, no_features_in
    model = Model([in_features, h1, h2], [0.2,0.1], 2).to(device)
    # Set metric (criterion) and choose optimizer
    class_weights = torch.FloatTensor(1 - class_no.values).to(device)
    criterion = nn.CrossEntropyLoss(weight = class_weights).to(device)
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) #, weight_decay=w_decay)
    
    # Training Loop
    train_loss_per_epoch = []
    test_loss_per_epoch = []
    Xtest_norm = Xtest_norm.to(device)
    ytest = ytest.to(device)    
    
    for epoch in range(epochs):
        
        model.train() 
        running_loss = 0.0
        for i, (features, labels) in enumerate(trainloader):
            # Send data to gpu if used
            features = features.to(device)
            labels = labels.to(device)

            # Reset gradient 
            optimizer.zero_grad()

            # Forward, backward, optimize
            outputs = model(features) 
            loss = criterion(outputs, torch.flatten(labels))
            loss.backward()
            optimizer.step()
            
            # Metrics
            running_loss += loss.item()
        
        train_loss_per_epoch.append(running_loss / (i+1))    
        running_loss = 0.0
        
        #running_test_loss = 0.0
        model.eval()
        # for i, (features, labels) in enumerate(testloader):
        #     features = features.to(device)
        #     labels = labels.to(device)
            
        #     y_pred = model(features.float())
        #     loss = criterion(y_pred, torch.flatten(labels))
        #     running_test_loss += loss.item()

        y_test_pred = model.forward(Xtest_norm)
        test_loss = criterion(y_test_pred, ytest).detach()
        
        
        # Metric
        #test_loss_per_epoch.append(running_test_loss / (i+1))
        test_loss_per_epoch.append(test_loss)
        
        # Print
        if epoch % 10 == 0:
            print(f"Ep: {epoch} Loss Training: {train_loss_per_epoch[-1]} Test: {test_loss_per_epoch[-1]}")
            
    # plt.figure()
    # plt.plot(np.arange(epochs), train_loss_per_epoch, label="Train")
    # plt.plot(np.arange(epochs), test_loss_per_epoch, label='Test')
    # plt.legend()
    # plt.show()    
    return model


#%%

if __name__ == "__main__":
    start_t = time.time()
    print('Loading Data')
    no_features_in, trainloader, testloader, class_no, Xtest_norm, \
        Xtrain, Xtest, ytrain, ytest = load_data()
    mid_t = time.time()
    print(f'Loading time: {mid_t - start_t}')
    print ('Predicting')
    model = train(100, no_features_in, trainloader, testloader, class_no, 0.0003, Xtest_norm, ytest)
    end_t = time.time()
    print(f'Total time: {end_t - start_t}')

# %%

%who DataFrame




# %%
