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

os.chdir('/home/patrick/Python/timeseries/weather/kaggle/Insurance_s4e7')

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


#class InsData(Dataset):
#     def __init__(self, X, y, X_transform=None):
#         self.X = torch.tensor(X, dtype=torch.float32)
#         self.y = torch.tensor(y, dtype=torch.float32)
#         self.X_transform = X_transform
        
#     def __len__(self):
#         return len(self.y)
    
#     @profile
#     def __getitem__(self, idx):

#         if self.X_transform is not None:
#             # Because it's only reading one line data will be T automatically
#             X_data = self.X.iloc[idx,:].to_frame().T
#             # Flatten the [1,C] to [C]
#             X_data = self.X_transform.transform(X_data).reshape(10)
#             X_data = torch.FloatTensor(X_data)
#             y_data = torch.LongTensor(self.y.iloc[idx,:].values)
#             # y_data will be [Rx1] no [R] flatten in training
#             return X_data, y_data
        
#         #X_data = torch.FloatTensor(self.X.iloc[idx,:].values)
#         #y_data = torch.LongTensor(self.y.iloc[idx,:].values)
#         X_data = self.X[idx] 
#         y_data = self.y[idx]
                
#         return X_data, y_data

# class InsData(Dataset):
#     def __init__(self, X, y):
#         self.X = X #torch.tensor(X, dtype=torch.float32)
#         self.y = y #torch.tensor(y, dtype=torch.float32)
        
#     def __len__(self):
#         return len(self.y)
    
#     def __getitem__(self, idx):
#         features = self.X.iloc[idx,:]
#         target = self.y.iloc[idx]
#         return torch.tensor(features, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)


# class InsData(Dataset):
#     def __init__(self, X, y):
#         self.X = torch.tensor(X, dtype=torch.float32)
#         self.y = torch.tensor(y, dtype=torch.float32)
        
#     def __len__(self):
#         return len(self.y)
    
#     def __getitem__(self, idx):
#         features = self.X[idx]
#         target = self.y[idx]
#         return features, target

# class InsData(Dataset):
#     def __init__(self, X, y, X_transform=None, y_transform=None):
#         self.X = X
#         self.y = y
        
#         self.Xtrain, self.Xtest, self.ytrain, self.ytest = \
#             train_test_split(self.X, self.y, test_size=0.33, stratify = self.y)
#         self.X_transform = X_transform
#         self.y_transform = y_transform

#         if X_transform is not None:
#             self.X_transform.fit(self.Xtrain)
#         if y_transform is not None:
#             self.y_transform.fit(self.ytrain)

#     def __len__(self):
#         if self.train is True:
#             return len(self.ytrain)
#         else:
#             return len(self.ytest)
        
#     def __getitem__(self, index):
#             X_data = self.Xtrain[index]
#             Y_data = self.ytrain[index]
#             if self.X_transform is not None:
#                 X_data = self.X_transform(X_data)
#             if self.Y_transform is not None:
#                 Y_data = self.y_transform(Y_data)
 
#             X_tdata = self.Xtest[index]
#             y_tdata = self.ytest[index]
#             if self.X_transform is not None:
#                 X_tdata = self.X_transform(X_tdata)
#             if self.y_transform is not None:
#                 y_tdata = self.y_transform(y_tdata) 
#             return X_data, Y_data, X_tdata, y_tdata
            

# class StandardScaler():
#     """Standardize data by removing the mean and scaling to unit variance.
#        This object can be used as a transform in PyTorch data loaders.

#     Args:
#         mean (FloatTensor): The mean value for each feature in the data.
#         scale (FloatTensor): Per-feature relative scaling.
#     """
#     def __init__(self, mean=None, scale=None):
#         if mean is not None:
#             #mean = torch.FloatTensor(mean)
#             mean = mean
#         if scale is not None:
#             #scale = torch.FloatTensor(scale)
#             scale = scale
#         self.mean_ = mean
#         self.scale_ = scale
               
#     def fit(self, sample):
#         """ Set the mean and scale values based on the sample data.
#         """
#         self.mean_ = sample.mean(axis=0, keepdims=True)
#         self.scale_ = sample.std(axis=0, keepdims=True)
#         return self
    
#     def __call__(self, sample):
#         return (sample - self.mean_)/self.scale_
    
#     def inverse_transform(self, sample):
#         """ Scale the data back to the original
#         """
#         return sample * self.scale_ + self.mean_

#%%
class EmbDataset(Dataset):
    def __init__(self, dfX, dfy):
        self.dfy = dfy.values
        self.features = dfX.values
         
    def __len__(self):
        return len(self.dfy)
    
    @profile
    def __getitem__(self,idx, batch_size):
        val = torch.tensor(self.features[idx:idx+batch_size,:], dtype=torch.float32)
        y       = torch.tensor(self.dfy[idx:idx+batch_size]   , dtype=torch.long)
        return [val, y]


class FastDataLoader:
    def __init__(self, ds, batch_size=32):

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
# Load, reduce data
# Split data
# Create datasets with transformations

@profile
def load_data(n=int(1e6), BATCH_SIZE =24000, full=False):
    # Load, shuffle get sample
    print('Read in')
    if full:
        print(f'load full dataset. Batch size: {BATCH_SIZE}')
        X = pd.read_parquet('data/artifacts/train.parquet')
        
        
        #X.drop(columns=['PI_AP'], inplace=True)
        # Split data on Previously_Insured (PI)
        #X = X.loc[X.Previously_Insured == 0]

        y= X.pop('Response')
        y = y.to_frame()

    else:
        df_in = pd.read_parquet('data/artifacts/train.parquet')
        print(f'load partial dataset. # data: {n}, batch: {BATCH_SIZE}')
        #X = X.sample(frac=1)
        df_in_y = df_in.pop('Response')
        X, _,y,_ = train_test_split(df_in, df_in_y, train_size= n, 
                                    random_state=27, stratify = df_in_y)
        del df_in, df_in_y
        
        #X.drop(columns=['PI_AP'], inplace=True)
        # X = pd.concat([X, y], axis=1)
        # X = X.loc[X.Previously_Insured == 0]
        #y = X.pop('Response')
        y = y.to_frame()
        print('X shape: ', X.shape)
        
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
    if full:
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.01, stratify=y)
    else: 
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.1, stratify=y)
    del X, y
    
    # Transforms
    print('Scaler')
    scaler = StandardScaler()
    Xtrain_norm = scaler.fit_transform(Xtrain)
    Xtrain_norm = pd.DataFrame(Xtrain_norm, columns=Xtrain.columns)
    Xtest_norm = scaler.transform(Xtest)
    Xtest_norm = pd.DataFrame(Xtest_norm, columns=Xtest.columns)

    print('Process combined')
    # Make Dataset
    train_ds = EmbDataset(Xtrain, ytrain)      
    test_ds = EmbDataset(Xtest, ytest)
    # Maket Loader
    trainloader = FastDataLoader(train_ds, batch_size=BATCH_SIZE)
    testloader = FastDataLoader(test_ds, batch_size=BATCH_SIZE)
    
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
    device = torch.device('cpu') #('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Def model
    in_features, h1, h2 = no_features_in, 2*no_features_in, no_features_in
    model = Model(in_features, h1, h2).to(device)
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
        
        running_test_loss = 0.0
        model.eval()
        for i, data in enumerate(testloader):
            in1, labels = data
            in1 = in1.to(device)
            labels = labels.to(device)
            
            y_pred = model.forward(in1)
            loss = criterion(y_pred, torch.flatten(labels)).detach()
            running_test_loss += loss.item()

        # y_test_pred = model.forward(Xtest_norm)
        # test_loss = criterion(y_test_pred, ytest).detach()
        # test_loss_per_epoch.append(test_loss)
        # Metric
        test_loss_per_epoch.append(running_test_loss / (i+1))
        running_test_loss = 0.0
        
        
        # Print
        if epoch % 10 == 0:
            print(f"Ep: {epoch} Loss Training: {train_loss_per_epoch[-1]} Test: {test_loss_per_epoch[-1]}")
 
    return model, train_loss_per_epoch, test_loss_per_epoch

def plot_training(epochs, train_loss_per_epoch, test_loss_per_epoch):
    plt.figure()
    plt.plot(np.arange(epochs), train_loss_per_epoch, label="Train")
    plt.plot(np.arange(epochs), test_loss_per_epoch, label='Test')
    plt.legend()
    plt.show()
    return
#%%
EPOCHS = 100
lr = 1e-4

if __name__ == "__main__":
    start_t = time.time()
    print('Loading Data')
    no_features_in, trainloader, testloader, class_no, Xtest_norm, \
        Xtrain, Xtest, ytrain, ytest = load_data(n=int(1e6), BATCH_SIZE =24000, full=False)
    mid_t = time.time()
    print ('Predicting')
    model, train_loss_per_epoch, test_loss_per_epoch = train(EPOCHS, no_features_in, trainloader, testloader, class_no, lr, Xtest_norm, ytest)
    end_t = time.time()
    print('Predict time: ', end_t - mid_t)
    print(f'Total time: {end_t - start_t}')
    plot_training(EPOCHS, train_loss_per_epoch, test_loss_per_epoch)

    
# %% Save model
#Save
torch.save(model.state_dict(), 'nn_ext.pth')

# Load
# model = TheModelClass(*args, **kwargs)
# model.load_state_dict(torch.load(PATH))
# model.eval()
#model.to(device)



#%% -------------- PREDICT --------------------------

#%whos DataFrame

#del Xtest, Xtest_norm, Xtrain, df_test, ytest, ytrain

# Clear GPU mem


# %% Get scaler

# _,_,_,_,_, Xtrain, Xtest,_,_ = load_data()

# df_org = pd.concat([Xtrain, Xtest], axis=1)

df_org = pd.read_parquet('data/train_extdata.parquet')
df_org.pop('Response')

scaler = StandardScaler()
scaler.fit(df_org) 

#%% Normalize prediction data

df_test = pd.read_parquet('data/test_extdata.parquet')
# print(df_test.shape)
# display(df_test.info())

X_pred_norm = scaler.transform(df_test)

Xpred = torch.tensor(X_pred_norm, dtype=torch.float32)
ypred = torch.zeros(Xpred.shape[0], 1)
print(Xpred.shape)
print(ypred.shape)

BATCH_SIZE = 10000
predloader = FastTensorDataLoader(Xpred, ypred, batch_size=BATCH_SIZE, shuffle=False)  

#%% Load model

no_features_in = Xpred.shape[1]
in_features, h1, h2 = no_features_in, 2*no_features_in, no_features_in
model = Model(in_features, h1, h2)

model.load_state_dict(torch.load('nn_ext.pth'))
model.to('cuda:0')

#%% Predict

y_pred_list = []
model.eval()
for i, (features, labels) in enumerate(predloader):
    features = features.to("cuda:0")
    #labels = labels.to("cuda:0")
    
    y_pred = model(features).detach().cpu().tolist()
    y_pred_list += y_pred

#%%
pred_data_tensor = torch.FloatTensor(y_pred_list)
print(pred_data_tensor.shape)

#%%


soft_pred = F.softmax(pred_data_tensor, dim=1) 

print(soft_pred[:5,1])


# %%
df_org_pred = pd.read_csv('data/sub_df_bare.csv')
display(df_org_pred.head())

#%%

df_sub = df_org_pred
df_pred = pd.DataFrame(soft_pred.detach().cpu().numpy(), columns=['pred0', 'Response'])

display(df_org_pred.head())
display(df_pred.head())

df_sub = pd.concat([df_org_pred, df_pred.Response], axis=1)

# df_sub.set_index('id', inplace=True)
display(df_sub)

# Change to parquet
df_sub.to_parquet('data/sub_nn_extend10.parquet')


#%% Check

df_2 = pd.read_parquet('data/sub_nn_extend10.parquet')
display(df_2.head())
# %%

# %%
