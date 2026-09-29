#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import itertools
from line_profiler import profile

import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
import time

os.chdir('/home/patrick/Python/timeseries/weather/kaggle/Insurance_s4e7')

print(os.getcwd())
import time
import sys
import random

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, \
    classification_report, root_mean_squared_error
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

@profile
def feat_eng():
    df = pd.read_parquet('data/df_proc.parquet')

    df = df.iloc[:100000,:]

    #display(df.head(10))

    list_int32 = ['Vintage','Policy_Sales_Channel','Region_Code']
    list_bool = ['Driving_License', 'Previously_Insured', 'Vehicle_Damage', 'Gender']
    list_float32 = ['Annual_Premium','Age']
    list_category = ['Vehicle_Age', 'Vintage','Policy_Sales_Channel','Region_Code']
    list_target = ['Response']

    def cross_(df, features):
        cross_name_list = []
        cross_product = list(set(itertools.combinations(features,2)))

        col_dict = {}
        for name1, name2 in cross_product:
            col_dict[f'{name1}_{name2}'] = pd.factorize((df[name1].astype(str) + df[name2].astype(str)).to_numpy())[0]
            cross_name_list.append(f'{name1}_{name2}')
            
        df_new = pd.DataFrame(col_dict)
        df = pd.concat([df, df_new], axis=1).reset_index(drop=True)
        del df_new
        return df, cross_name_list
    
    # Data-prep
    # df['PI_AP'] = pd.factorize((df['Previously_Insured'].astype(str) + df['Annual_Premium'].astype(str)).to_numpy())[0]
    # df['PI_VA'] = pd.factorize((df['Previously_Insured'].astype(str) + df['Vehicle_Age'].astype(str)).to_numpy())[0]
    # df['PI_VD'] = pd.factorize((df['Previously_Insured'].astype(str) + df['Vehicle_Damage'].astype(str)).to_numpy())[0]
    # df['PI_V'] = pd.factorize((df['Previously_Insured'].astype(str) + df['Vintage'].astype(str)).to_numpy())[0]

    # df['AP_RC'] = pd.factorize((df['Annual_Premium'].astype(str) + df['Region_Code'].astype(str)).to_numpy())[0]
    # df['AP_PSC'] = pd.factorize((df['Annual_Premium'].astype(str) + df['Policy_Sales_Channel'].astype(str)).to_numpy())[0]
    # df['AP_V'] = pd.factorize((df['Annual_Premium'].astype(str) + df['Vintage'].astype(str)).to_numpy())[0]

    # df['PSC_RC'] = pd.factorize((df['Policy_Sales_Channel'].astype(str) + df['Region_Code'].astype(str)).to_numpy())[0]
    # df['PSC_V'] = pd.factorize((df['Policy_Sales_Channel'].astype(str) + df['Vintage'].astype(str)).to_numpy())[0]
    # df['PSC_G'] = pd.factorize((df['Policy_Sales_Channel'].astype(str) + df['Gender'].astype(str)).to_numpy())[0]
    # df['PSC_PI'] = pd.factorize((df['Policy_Sales_Channel'].astype(str) + df['Previously_Insured'].astype(str)).to_numpy())[0]


    # Feature eng float
    df['log_Annual_Premium'] = np.log(df.Annual_Premium+1)
    df['quad_Annual_Premium'] = df.Annual_Premium * df.Annual_Premium
    df['sq_Annual_Premium'] = np.sqrt(df.Annual_Premium+1)

    df['log_Age'] = np.log(df.Age+1)
    df['quad_Age'] = df.Age * df.Age
    df['sq_Age'] = np.sqrt(df.Age+1)

    cat_comb_list = ['Gender', 'Driving_License', 'Region_Code', 'Vehicle_Age', 'Vehicle_Damage', 'Annual_Premium',
                    'Policy_Sales_Channel', 'Vintage']

    df, cross_name_list = cross_(df, cat_comb_list)
    df[list_category+cross_name_list] = df[list_category+cross_name_list].astype('category')
    #print(df.info())

    # train,test split
    
    y = df.pop('Response')

    Xtrain, Xtest, ytrain, ytest = train_test_split(df, y, test_size=0.33, stratify = y)

    #%%Scale numerical features and split into X,y

    scaler = StandardScaler()
    Xtrain_norm = scaler.fit_transform(Xtrain)
    Xtest_norm = scaler.transform(Xtest)

    # Make Tensors 
    ytrain = torch.flatten(torch.LongTensor(ytrain.values))
    ytest = torch.flatten(torch.LongTensor(ytest.values))
    train_norm = torch.FloatTensor(Xtrain_norm)
    test_norm = torch.FloatTensor(Xtest_norm)
    
    
    return train_norm, test_norm, ytrain, ytest, Xtrain, Xtest, ytrain, ytest

# %%

class Model(nn.Module):
    def __init__(self, in_features, h1=20, h2=30,d_out=2):
        super().__init__()
        
        self.fc1 = nn.Linear(in_features, h1)
        self.relu1 = nn.SiLU()
        self.bn1 = nn.BatchNorm1d(h1)
        self.dp1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(h1, h2)
        self.relu2 = nn.SiLU()
        self.bn2 = nn.BatchNorm1d(h2)
        self.dp2 = nn.Dropout(0.1)

        self.out = nn.Linear(h2,d_out)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.bn1(x)
        x = self.dp1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.bn2(x)
        x = self.dp2(x)
        x = self.out(x)
        return x

# %%


class LRScheduler():
    """
    Learning rate scheduler. If the validation loss does not decrease for the 
    given number of `patience` epochs, then the learning rate will decrease by
    by given `factor`.
    """
    def __init__(
        self, optimizer, patience=5, min_lr=1e-6, factor=0.9):
        """
        new_lr = old_lr * factor

        :param optimizer: the optimizer we are using
        :param patience: how many epochs to wait before updating the lr
        :param min_lr: least lr value to reduce to while updating
        :param factor: factor by which the lr should be updated
        """
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor

        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( 
                self.optimizer,
                mode='min',
                patience=self.patience,
                factor=self.factor,
                min_lr=self.min_lr,
            )

    def __call__(self, val_loss):
        self.lr_scheduler.step(val_loss)

@profile
def train_fcn(n_epochs, samples_per_batch, lr, w_decay, Xtrain, ytrain, Xtest, 
              ytest, h1,h2,
              device=None, lr_scheduler_=False, verbose=False):
    if device is not None:
        device = torch.device('cuda:0') # if torch.cuda.is_available() else 'cpu')
        print(device)
    # num_batches = batches
    num_batches = len(Xtrain) // samples_per_batch
    # samples_per_batch = len(Xtrain) // num_batches

    train_loss_hist, train_acc_hist = [], []
    test_loss_hist, test_acc_hist = [], []
    
    # Create model class
    in_features = Xtrain.shape[1]
    model = Model(in_features, h1,h2).to(device)
    
    # Set metric (criterion) and choose optimizer
    #class weights for 2 weight = 1-(class no/total no)
    class_no = pd.DataFrame(ytrain.numpy()).value_counts()
    class_weights = torch.FloatTensor(1 - class_no.values/len(ytrain)).to(device)
    #loss function with class weights
    criterion = nn.CrossEntropyLoss(weight = class_weights) 
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=w_decay)
    # Adaptive learning rate
    if lr_scheduler_:
        lr_scheduler = LRScheduler(optimizer)
    
    # ############# TENSORBOARD #############
    # sample_data = Xtrain[:samples_per_batch]
    # writer.add_graph(model, sample_data)
    # writer.close()
    # ######################################

    Xtrain = Xtrain.to(device)
    ytrain = ytrain.to(device)
    Xtest = Xtest.to(device)
    ytest = ytest.to(device)
    for epoch in range(n_epochs):

        model.train()
        running_loss = 0.0
        running_acc = 0.0
        for batch in range(num_batches):

            # Reset
            optimizer.zero_grad()
            # Get a batch
            start = batch * samples_per_batch
            X_batch = Xtrain[start:start+samples_per_batch]
            y_batch = ytrain[start:start+samples_per_batch]
            # Forward pass
            y_batch_pred = model.forward(X_batch)
            loss = criterion(y_batch_pred, y_batch)
            acc = (torch.argmax(y_batch_pred, 1) == y_batch).float().mean().detach()
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            # Update weights
            optimizer.step()
            # Compute batch metrics
            running_loss += loss.item()
            running_acc += acc
            
        # Stor batch results
        train_loss_hist.append(running_loss / (batch+1))
        train_acc_hist.append(running_acc / (batch + 1))
        running_loss = 0.0
        running_acc = 0.0

        # Evaluate training on test set
        model.eval()
        y_test_pred = model.forward(Xtest)
        test_loss = criterion(y_test_pred, ytest).detach()
        test_acc = (torch.argmax(y_test_pred, 1) == ytest).float().mean().detach()
        
        # Store
        test_loss_hist.append(test_loss)
        test_acc_hist.append(test_acc)

        if lr_scheduler_:
            lr_scheduler(test_loss)

        if verbose:
            if epoch % 10 == 0:
                print(f'Epoch: {epoch} train loss {train_loss_hist[-1]:.4f}, test loss {test_loss:.4f}')
                #print(lr_scheduler.lr_scheduler.get_last_lr())
                
    return model, train_loss_hist, train_acc_hist, test_loss_hist, test_acc_hist
 
 
def plot_training(train_loss_hist, train_acc_hist, test_loss_hist, test_acc_hist):
    xx = list(range(len(train_loss_hist)))

    plt.figure()
    plt.plot(xx, train_loss_hist, label='Train')
    plt.plot(xx, test_loss_hist, label='Test')
    plt.title('Train/Test Loss')
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure()
    plt.plot(xx, train_acc_hist, label='Train')
    plt.plot(xx, test_acc_hist, label='Test')
    plt.title('Train/Test Accuracy')
    plt.legend()
    plt.grid()
    plt.show()       

#%% Make model

if __name__ == '__main__':
    start_time = time.time()
    print('Load Data')
    train_norm, test_norm, ytrain, ytest, Xtrain, Xtest, ytrain, ytest = feat_eng()

    n_nodes = Xtrain.shape[1]
    h1,h2 = 2*n_nodes, n_nodes
    n_epochs = 100
    samples_per_batch = 8*1024
    lr = 3e-4
    w_decay = 1e-4
    
    print('Run Training')
    modelx, train_loss_hist, train_acc_hist, test_loss_hist, test_acc_hist = \
        train_fcn(n_epochs, samples_per_batch, lr, w_decay, train_norm, ytrain, 
                test_norm, ytest, h1,h2,
                device=1, lr_scheduler_=False, verbose=True)
    end_time = time.time()
    print(f'Time: {end_time - start_time}')

