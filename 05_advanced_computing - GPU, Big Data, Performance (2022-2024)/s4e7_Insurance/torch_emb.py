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

#class FastTensorDataLoader:
#     """
#     A DataLoader-like object for a set of tensors that can be much faster than
#     TensorDataset + DataLoader because dataloader grabs individual indices of
#     the dataset and calls cat (slow).
#     Source: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
#     """
#     def __init__(self, *tensors, batch_size=32, shuffle=False):
#         """
#         Initialize a FastTensorDataLoader.

#         :param *tensors: tensors to store. Must have the same length @ dim 0.
#         :param batch_size: batch size to load.
#         :param shuffle: if True, shuffle the data *in-place* whenever an
#             iterator is created out of this object.

#         :returns: A FastTensorDataLoader.
#         """
#         assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
#         self.tensors = tensors

#         self.dataset_len = self.tensors[0].shape[0]
#         self.batch_size = batch_size
#         self.shuffle = shuffle

#         # Calculate # batches
#         n_batches, remainder = divmod(self.dataset_len, self.batch_size)
#         if remainder > 0:
#             n_batches += 1
#         self.n_batches = n_batches
        
#     def __iter__(self):
#         if self.shuffle:
#             r = torch.randperm(self.dataset_len)
#             self.tensors = [t[r] for t in self.tensors]
#         self.i = 0
#         return self

#     def __next__(self):
#         if self.i >= self.dataset_len:
#             raise StopIteration
#         batch = tuple(t[self.i:self.i+self.batch_size] for t in self.tensors)
#         self.i += self.batch_size
#         return batch

#     def __len__(self):
#         return self.n_batches

#%%

class EmbDataset(Dataset):
    def __init__(self, dfX, dfy, num_cols, cat_cols):
        self.dfy = dfy.values
        self.num_features = dfX.loc[:,num_cols].values
        self.cat_features = dfX.loc[:,cat_cols].values
         
    def __len__(self):
        return len(self.dfy)
    
    @profile
    def __getitem__(self,idx, batch_size):
        num_val = torch.tensor(self.num_features[idx:idx+batch_size,:], dtype=torch.float32)
        cat_val = torch.tensor(self.cat_features[idx:idx+batch_size,:], dtype=torch.long)
        y       = torch.tensor(self.dfy[idx:idx+batch_size]           , dtype=torch.long)
        return [cat_val, num_val, y]

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

class Model(nn.Module):
    def __init__(self, emb_sizes, emb_dropout, fc_in_out, dropout_perc, 
                 n_cat_cols, n_num_cols, d_out=2):
        super().__init__()
        
        # Get embedding
        self.embeddings = nn.ModuleList([nn.Embedding(car,siz) for car,siz in emb_sizes])
        for emb in self.embeddings:
            emb.weight.data.uniform_(-0.01, 0.01)
            #nn.init.kaiming_normal_(emb.weight.data)
            
        # Embedding dropout
        self.emb_dropout = nn.Dropout(emb_dropout)
        # Calculate in_features to linear layer
        emb_vector_sum = sum([e.embedding_dim for e in self.embeddings])
        # Add in_feature to list
        linear_szs = [emb_vector_sum + n_num_cols] + fc_in_out
        
        self.n_num_cols = n_num_cols
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
    
    @profile
    def forward(self, cat_fields, num_fields):
        # Initialize embedding for respective cat fields
        x1 = [e(cat_fields[:,i]) for i,e in enumerate(self.embeddings)]
        # Concatenate all embeddings on axis 1
        x1 = torch.cat(x1,1)
        # Dropout for embeddings
        x1 = self.emb_dropout(x1)
        
        # Input normalization for cont fields
        x2 = self.batchnorm_num(num_fields)
        # Concat inputs
        x1 = torch.cat([x1, x2], 1)
        
        for fc, bn, drop in zip(self.fc_layers, self.batchnorm, self.dropout):
            x1 = F.silu(fc(x1))
            x1 = bn(x1)
            x1 = drop(x1)
        
        x1 = self.out(x1)
        return x1       

#%%
# Load, reduce data
# Split data
# Create datasets with transformations

@profile
def load_data(n = int(1e6), BATCH_SIZE = 24000):
    # Load, shuffle get sample
    print('Read in')
    X = pd.read_parquet('data/train_proc.parquet')
    print('shuffle')
    #X = X.sample(frac=1)
    #df_in_y 
    y= X.pop('Response')
    
    # X, _,y,_ = train_test_split(df_in, df_in_y, train_size= n, 
    #                             random_state=27, stratify = df_in_y)

    # del df_in, df_in_y
    y = y.to_frame()

    # X = pd.read_parquet('data/train_proc_small.parquet')
    # y = pd.read_parquet('data/ytrain_proc_small.parquet')

    # Data prep
    list_int32 = ['Vehicle_Age', 'Vintage','Policy_Sales_Channel','Region_Code']
    list_bool = ['Driving_License', 'Previously_Insured', 'Vehicle_Damage', 'Gender']
    list_float32 = ['Annual_Premium','Age']
    list_target = ['Response']
    list_cat = list_int32 + list_bool
    len_cat_cols = len(list_cat)
    len_num_cols = len(list_float32)

    
    # Split data in train, test
    print('Split data')
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.01, stratify=y)
    del X, y
    
    print('Labels')
    
    oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)
    Xtrain[list_int32] = oe.fit_transform(Xtrain[list_int32])
    Xtest[list_int32] = oe.transform(Xtest[list_int32])
    Xtest[list_int32] = Xtest[list_int32].fillna(0)
    
    # Got col cardinality    
    Xtrain[list_cat] = Xtrain[list_cat].astype('category')
    '''Ebbedding cardinality is a list of two-tuples. First is no of unique values in a cat,
        the second is the number steps used to embedd'''
    embedding_cardinality = {n: len(c.cat.categories) for n,c in Xtrain[list_cat].items()}
    emb_sizes = [(size, min(50, (size+1) // 2 )) for item, size in embedding_cardinality.items()]
    Xtrain[list_cat] = Xtrain[list_cat].astype(np.int32)
    Xtest[list_cat] = Xtest[list_cat].astype(np.int32)

    # # Split X in num, cat
    # Xtrain_num = Xtrain.loc[:,list_float32]
    # Xtrain_cat = Xtrain.loc[:,list_cat]
    # Xtest_num = Xtest.loc[:,list_float32]
    # Xtest_cat = Xtest.loc[:,list_cat]
    # Calc class imbalance
    class_no = ytrain.value_counts()/len(ytrain)

    print('Scaler')
    scaler = StandardScaler()
    Xtrain.loc[:,list_float32] = scaler.fit_transform(Xtrain.loc[:,list_float32])
    Xtest.loc[:,list_float32] = scaler.transform(Xtest.loc[:,list_float32])

    
    train_ds = EmbDataset(Xtrain, ytrain, list_float32, list_cat)      
    test_ds = EmbDataset(Xtest, ytest, list_float32, list_cat)
    
    # trainloader = DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=4, \
    #     pin_memory=False, shuffle=False)
    # testloader = DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=4, \
    #     pin_memory=False, shuffle=False)

    # Maket Loader
    trainloader = FastDataLoader(train_ds, batch_size=BATCH_SIZE)
    testloader = FastDataLoader(test_ds, batch_size=BATCH_SIZE)



    # #Make tensors for fasttensordataloader, Fast loader for full dataset
    # train_x = torch.tensor(Xtrain_norm.values, dtype=torch.float32) 
    # test_x = torch.tensor(Xtest_norm.values, dtype=torch.float32) 
    # train_y = torch.tensor(ytrain.values, dtype=torch.long)
    # test_y = torch.tensor(ytest.values, dtype=torch.long)
    #trainloader = FastTensorDataLoader(train_x, train_y, batch_size=BATCH_SIZE, shuffle=False)  
    #testloader = FastTensorDataLoader(test_x, test_y, batch_size=BATCH_SIZE, shuffle=False)

    return trainloader, testloader, class_no, emb_sizes, \
            len_cat_cols, len_num_cols #, Xtest_norm, Xtrain, Xtest, ytrain, ytest
# Test the dataloader
# dataiter = iter(trainloader)
# feature,label = next(dataiter)

# print('Features', feature.shape) #, '\n', feature)
# print('labels', label.shape) #, '\n', label)

# %%

@profile
def train(epochs, emb_sizes, trainloader, testloader, class_no, lr, len_cat_cols, len_num_cols):
    
    # Device selection
    device = torch.device('cuda:0') #('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Def model
    model = Model(emb_sizes, 0.1, [256, 128, 10], [0.05, 0.05, 0.05], len_cat_cols, len_num_cols).to(device)
    print(model.parameters)
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
        for i, data in enumerate(trainloader):
            in1, in2, labels = data
            # Send data to gpu if used
            in1 = in1.to(device)
            in2 = in2.to(device)
            labels = labels.to(device)

            # Reset gradient 
            optimizer.zero_grad()

            # Forward, backward, optimize
            outputs = model.forward(in1, in2) 
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
            in1, in2, labels = data
            in1 = in1.to(device)
            in2 = in2.to(device)
            labels = labels.to(device)
            
            y_pred = model.forward(in1, in2)
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

if __name__ == "__main__":
    EPOCHS = 50
    
    start_t = time.time()
    print('Loading Data')
    trainloader, testloader, class_no, emb_sizes, \
        len_cat_cols, len_num_cols = load_data()
#        Xtest_norm, Xtrain, Xtest, ytrain, ytest 
    mid_t = time.time()
    print(f'Loading time: {mid_t - start_t}')
    print ('Predicting')
    model, train_loss_per_epoch, test_loss_per_epoch = \
        train(EPOCHS, emb_sizes, trainloader, testloader, class_no, 0.001, \
                len_cat_cols, len_num_cols)
    end_t = time.time()
    print(f'Prediction time: {end_t - start_t}')
    plot_training(EPOCHS, train_loss_per_epoch, test_loss_per_epoch)

# %% Save model

# %who DataFrame
#Save
torch.save(model.state_dict(), 'nn_emb_fulldata_v2.pth')

#print(emb_sizes)
#[(3, 2), (290, 50), (152, 50), (53, 27), (2, 1), (2, 1), (2, 1), (2, 1)]

# %%
# Load data
# predict data
# save data

# Data prep
list_int32 = ['Vehicle_Age', 'Vintage','Policy_Sales_Channel','Region_Code']
list_bool = ['Driving_License', 'Previously_Insured', 'Vehicle_Damage', 'Gender']
list_float32 = ['Annual_Premium','Age']
list_target = ['Response']
list_cat = list_int32 + list_bool
len_cat_cols = len(list_cat)
len_num_cols = len(list_float32)

df_test = pd.read_parquet('data/test_norm.parquet')
y_dummy = pd.DataFrame(data=np.zeros((df_test.shape[0],1)), columns=['Response'])

# Make dataset
pred_ds = EmbDataset(df_test, y_dummy, list_float32, list_cat)
# Maket Loader
predloader = FastDataLoader(pred_ds, batch_size=24000)

#%%

y_pred_list = []
device = 'cuda:0'
model.eval()
for i, data in enumerate(predloader):
    in1, in2, labels = data
    in1 = in1.to(device)
    in2 = in2.to(device)
    #labels = labels.to(device)
    
    y_pred = model.forward(in1, in2).detach().cpu().tolist()
    y_pred_list += y_pred

#%%

pred_data_tensor = torch.FloatTensor(y_pred_list)
print(pred_data_tensor.shape)

#%%

soft_pred = F.softmax(pred_data_tensor, dim=1) 
print(soft_pred[:5,:])

#%%

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
df_sub.to_parquet('data/sub_nn_emb_fulldata_v2.parquet')

# %%
