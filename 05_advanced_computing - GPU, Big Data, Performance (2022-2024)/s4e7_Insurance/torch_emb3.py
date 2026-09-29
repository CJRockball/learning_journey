#%%
import time
import sys
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from line_profiler import profile
import logging
import time
import joblib 

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder, TargetEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn import set_config
set_config(transform_output = "pandas")

from captum.attr import IntegratedGradients, LayerConductance, NeuronConductance
from IPython.display import display

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset, IterableDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.classification import AUROC, BinaryAUROC

from torch.utils.tensorboard import SummaryWriter
# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('runs/kags4e7')

import gc
torch.cuda.empty_cache()
gc.collect()


def seed_everything(seed=27):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
#seed_everything()
os.chdir('/home/patrick/Python/timeseries/weather/kaggle/Insurance_s4e7')


logger = logging.getLogger('logging/nn_log.log')
if not logger.hasHandlers():
    logger.setLevel(logging.INFO)
    f_handler = logging.FileHandler('logging/nn_log.log')
    f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    f_handler.setFormatter(f_format)
    f_handler.name = 'logging/nn_log.log'
    logger.addHandler(f_handler)        

#%%

class EmbDataset(Dataset):
    def __init__(self, dfX, dfy, num_cols, cat_cols):
        self.dfy = dfy.values
        #self.num_features = dfX.loc[:,num_cols].values
        self.cat_features = dfX.loc[:,cat_cols].values
         
    def __len__(self):
        return len(self.dfy)
    
    @profile
    def __getitem__(self,idx, batch_size):
        #num_val = torch.tensor(self.num_features[idx:idx+batch_size,:], dtype=torch.float32)
        cat_val = torch.tensor(self.cat_features[idx:idx+batch_size,:], dtype=torch.long)
        y       = torch.tensor(self.dfy[idx:idx+batch_size]           , dtype=torch.long)
        return [cat_val, y]

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


class LRScheduler():
    """
    Learning rate scheduler. If the validation loss does not decrease for the 
    given number of `patience` epochs, then the learning rate will decrease by
    by given `factor`.
    """
    def __init__(
        self, optimizer, patience=3, min_lr=1e-6, factor=0.6):
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


class Model(nn.Module):
    def __init__(self, meta_data, emb_dropout, fc_in_out, dropout_perc, d_out=2):
        super().__init__()
        n_num_cols = meta_data['num_num_cols']
        emb_sizes = meta_data['emb_sizes']
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
    def forward(self, cat_fields): #, num_fields):
        # Initialize embedding for respective cat fields
        x1 = [e(cat_fields[:,i]) for i,e in enumerate(self.embeddings)]
        # Concatenate all embeddings on axis 1
        x1 = torch.cat(x1,1)
        # Dropout for embeddings
        x1 = self.emb_dropout(x1)
        
        # Input normalization for cont fields
        #x2 = self.batchnorm_num(num_fields)
        # Concat inputs
        #x1 = torch.cat([x1, x2], 1)
        
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

def read_in_file(n=int(1e6),BATCH_SIZE=24000, full=False):
    if full:
        print(f'load full dataset. Batch size: {BATCH_SIZE}')
        X = pd.read_parquet('data/artifacts/train.parquet')
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
        y = y.to_frame()
        print('X shape: ', X.shape)
    return X, y

def transform_pipeline(Xtrain, Xtest, list_cat, list_num):
    # Assume 'numerical_features' and 'categorical_features' are lists of feature names
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[
        ('oe', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan))
        ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, list_num),
            ('cat', categorical_transformer, list_cat),
        ]
    ) #.set_output(transform='pandas')
    
    # Assuming 'model' is your machine learning model (e.g., RandomForestClassifier)
    pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
    # Now you can use the pipeline for training and prediction
    Xtrain = pipeline.fit_transform(Xtrain)
    # Pipeline changes the col names, change back
    Xtrain.columns = list_num + list_cat
    Xtest = pipeline.transform(Xtest)
    Xtest.columns = list_num + list_cat
    # Put all unseed classes in class 0
    Xtest[list_cat] = Xtest[list_cat].fillna(0)
    
    return Xtrain, Xtest, pipeline


def get_presplit_meta(y, meta_data):
    label_counts = y.value_counts()/len(y)
    meta_data['label_imbalance'] = label_counts
    return meta_data


def get_postsplit_meta(Xtrain, list_cat, meta_data):
    '''Ebbedding cardinality is a list of two-tuples. First is no of unique values in a cat,
        the second is the number steps used to embedd'''
    embedding_cardinality = {n: len(c.unique()) for n,c in Xtrain[list_cat].items()}
    emb_sizes = [(size, min(50, (size+1) // 2 )) for item, size in embedding_cardinality.items()]
    meta_data['emb_sizes'] = emb_sizes
    return meta_data


def add_cross(df):

    # num fatures
    value_counts_mapping = df['Annual_Premium'].value_counts().to_dict()
    annual_premium_counts = df['Annual_Premium'].map(value_counts_mapping).astype('int32')
    df['Annual_Premium'] = df['Annual_Premium'].where(annual_premium_counts >= 50, -1).astype('int32')
    df['Annual_Premium_weights'] = annual_premium_counts

    value_counts_mapping = df['Vintage'].value_counts().to_dict()
    vintage_counts = df['Vintage'].map(value_counts_mapping).astype('int32')
    df['Vintage'] = df['Vintage'].where(vintage_counts >= 50, -1).astype('int32')
    df['Vintage_weights'] = vintage_counts

    value_counts_mapping = df['Age'].value_counts().to_dict()
    age_counts = df['Aage'].map(value_counts_mapping).astype('int32')
    df['Age'] = df['Age'].where(age_counts >= 50, -1).astype('int32')
        
    df['PI_AP'] = (pd.factorize((df['Previously_Insured'].astype(str) + df['Annual_Premium'].astype(str)).to_numpy())[0]).astype(np.float32)
    # Cat features
    df['PI_VA'] = pd.factorize((df['Previously_Insured'].astype(str) + df['Vehicle_Age'].astype(str)).to_numpy())[0]
    #df['PI_VD'] = pd.factorize((df['Previously_Insured'].astype(str) + df['Vehicle_Damage'].astype(str)).to_numpy())[0]
    #df['PI_V'] = pd.factorize((df['Previously_Insured'].astype(str) + df['Vintage'].astype(str)).to_numpy())[0]
    add_cat_features = ['PI_VA', 'PI_AP', 'Annual_Premium_weights', 'Vintage_weights'] #, 'PI_VD', 'PI_V']
    add_num_features = []
    
    return df, add_cat_features, add_num_features

def choose_int_type(df, list_change_cat):
    for col in list_change_cat:
        col_min = df[col].min()
        col_max = df[col].max()
        if col_min >= -128 and col_max <= 127:
            df[col] = df[col].astype(np.int8)
        elif col_min >= -32768 and col_max <= 32767:
            df[col] = df[col].astype(np.int16)
        elif col_min >= -2147483648 and col_max <= 2147483647:
            df[col] = df[col].astype(np.int32)
    return df

def make_balanced(X, y):
    X['Response'] = y
    X = X.sample(frac=1)
    X_1 = X.loc[X.Response == 1,:]
    X_0 = X.loc[X.Response == 0,:][:len(X_1)]
    del X, y
    gc.collect()
    X = pd.concat([X_1, X_0], axis=0).sample(frac=1).reset_index(drop=True)
    y = X.pop('Response')
    
    return X, y
 
@profile
def load_data(n=int(1e6), BATCH_SIZE =24000, full=False, pred=False,
              save_tmp=False, load_tmp=False):
    
    assert (pred == False or load_tmp == False), "'pred' and 'load_tmp' should be true at the same time"
    
    print('Read in')
    if not load_tmp:
        # Define
        meta_data = {}
        # Load, shuffle get sample
        X, y = read_in_file(n=n, BATCH_SIZE=BATCH_SIZE, full=full)
        
        ## Make balanced dataset
        X, y = make_balanced(X, y)
        
        # Data prep
        list_int = ['Annual_Premium','Vehicle_Age','Policy_Sales_Channel',
                    'Region_Code','Age', 'Vintage']
        list_bool = ['Driving_License', 'Previously_Insured', 'Vehicle_Damage', 'Gender']
        list_float = []
        list_target = ['Response']

        list_cat = list_int + list_bool
        list_num = list_float

        ## Pre-split Feature Engieering
        # X, add_cat_features, add_num_features = add_cross(X)
        # add_cat_features = []
        # add_num_features = []
        # list_cat = list_cat + add_cat_features
        # list_num = list_num + add_num_features

        ## Pre split transforms
        

        # Get pre-split meta data
        meta_data = get_presplit_meta(y, meta_data)
        
        # Split data in train, test
        print('Split data')
        if full:
            Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.01) #, stratify=y)
        else: 
            Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.1, stratify=y)
        
        del X, y
        gc.collect()
        
        # Post-spit Feature Engineering
        #Xtrain, add_cat_features, add_num_features = add_cross(Xtrain)
        #Xtest,_,_ = add_cross(Xtest)
        # add_cat_features = []
        # add_num_features = []
        # list_cat = list_cat + add_cat_features
        # list_num = list_num + add_num_features

        # Get postsplit meta
        # Use category for embedding
        meta_data = get_postsplit_meta(Xtrain, list_cat, meta_data)
        
        
        # Transforms with data leakage
        #Xtrain, Xtest, transform_dict = leakage_transforms(Xtrain, Xtest, list_cat, list_num, transform_dict)
        Xtrain, Xtest, pipeline = transform_pipeline(Xtrain, Xtest, list_cat, list_num)
        Xtrain = choose_int_type(Xtrain, list_cat)
        Xtest = choose_int_type(Xtest, list_cat)
        
        # Get len of cat, num features after feature engineering
        meta_data['list_cat'] = list_cat
        meta_data['list_num'] = list_num
        meta_data['num_cat_cols'] = len(list_cat)
        meta_data['num_num_cols'] = len(list_num)
        
        if save_tmp:
            temp_data = {'Xtrain':Xtrain, 'Xtest':Xtest, 'ytrain':ytrain, 'ytest':ytest}
            for fname,fdata in temp_data.items():
                fdata.to_parquet(f'data/artifacts/{fname}_tmp.parquet')
            joblib.dump(pipeline, 'data/artifacts/pipeline.pkl')
            joblib.dump(meta_data, 'data/artifacts/meta.pkl')
        
    if load_tmp:
        Xtrain = pd.read_parquet(f'data/artifacts/Xtrain_tmp.parquet')
        Xtest = pd.read_parquet(f'data/artifacts/Xtest_tmp.parquet')
        ytrain = pd.read_parquet(f'data/artifacts/ytrain_tmp.parquet')
        ytest = pd.read_parquet(f'data/artifacts/ytest_tmp.parquet')
        meta_data = joblib.load('data/artifacts/meta.pkl')
        
    print('Process combined')
    print(f'Dataset size: {Xtrain.shape}')
    # Make Dataset
    train_ds = EmbDataset(Xtrain, ytrain, meta_data['list_num'], meta_data['list_cat'])      
    test_ds = EmbDataset(Xtest, ytest,  meta_data['list_num'], meta_data['list_cat'])
    # Maket Loader
    trainloader = FastDataLoader(train_ds, batch_size=BATCH_SIZE)
    testloader = FastDataLoader(test_ds, batch_size=BATCH_SIZE)
    
    del Xtrain, ytrain, Xtest, ytest
    gc.collect()
    
    
    # Prep pred data
    predloader = None
    if pred and not load_tmp:
        print('Processing prediction data')
        # Add test to see if basic features are in the df
        # Post-split Feature Engineering
        df_test = pd.read_parquet('data/artifacts/test.parquet')

        # Split data on PI, same as train
        
        #df_test, _,_ = add_cross(df_test)
        # Features are already added to list_num and list_cat
        # Add test to see if all features are in df
        
        ## Transform with leakage data
        ## Use individual transformers
        # df_test[list_cat] = transform_dict['oe'].transform(df_test[list_cat])
        # df_test[list_cat] = df_test[list_cat].fillna(0)
        # df_test[list_num] = transform_dict['scaler'].transform(df_test[list_num])
        ## Set up pipeline
        df_test = pipeline.transform(df_test)
        df_test.columns = list_num + list_cat
        # Put all unseed classes in class 0
        df_test[list_cat] = df_test[list_cat].fillna(0)
        df_test = choose_int_type(df_test, list_cat)

        # Add dummy y for dataset/dataloader
        y_dummy = pd.DataFrame(data=np.zeros((df_test.shape[0],1)), columns=['Response'])

        if save_tmp:
            df_test.to_parquet('data/artifacts/df_test.parquet')

        # Make dataset
        pred_ds = EmbDataset(df_test, y_dummy, meta_data['list_num'], meta_data['list_cat'])
        # Maket Loader
        predloader = FastDataLoader(pred_ds, batch_size=BATCH_SIZE)
        return trainloader, testloader, meta_data, predloader


    if load_tmp:
        df_test = pd.read_parquet('data/artifacts/df_test.parquet')
        pipeline = joblib.load('data/artifacts/pipeline.pkl')
        
        # Make dataset
        y_dummy = pd.DataFrame(data=np.zeros((df_test.shape[0],1)), columns=['Response'])
        pred_ds = EmbDataset(df_test, y_dummy, meta_data['list_num'], meta_data['list_cat'])
        # Maket Loader
        predloader = FastDataLoader(pred_ds, batch_size=BATCH_SIZE)
    
    return trainloader, testloader, meta_data, predloader


# %%

@profile
def make_model(meta_data):
    # Device selection
    device = torch.device('cuda:0') #('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Def model
    #nn_model = Model(meta_data, 0.1, [20, 5], [0.1, 0.1]).to(device) 
    nn_model = Model(meta_data, 0.1, [40, 20, 5], [0.1, 0.1, 0.1]).to(device) 
    
    return nn_model


@profile
def train(epochs, model, trainloader, testloader, \
            class_no, lr, lr_scheduler_=False):
    
    # Device selection
    device = torch.device('cuda:0') #('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Set metric (criterion) and choose optimizer
    class_weights = torch.FloatTensor(1 - class_no.values).to(device)
    #class_weights = torch.tensor([0.2, 0.8], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss() #weight = class_weights).to(device)
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=2e-3)
    if lr_scheduler_:
        lr_scheduler = LRScheduler(optimizer)
    # Training Loop
    #rocauc = BinaryAUROC().to(device)
    train_loss_per_epoch = []
    test_loss_per_epoch = []   
    for epoch in range(epochs):
        
        model.train() 
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            in1, labels = data
            # Send data to gpu if used
            in1 = in1.to(device)
            #in2 = in2.to(device)
            labels = labels.to(device)

            # Reset gradient 
            optimizer.zero_grad()

            # Forward, backward, optimize
            outputs = model.forward(in1) 
            loss = criterion(outputs, torch.flatten(labels))
            loss.backward()
            optimizer.step()
            
            # Metrics
            running_loss += loss.item()
            # # Compute roc auc
            # pred_layer = nn.Softmax(dim=1)
            # auc = rocauc(pred_layer(outputs)[:, 1], labels).item()

        
        train_loss_per_epoch.append(running_loss / (i+1))    
        running_loss = 0.0
        
        running_test_loss = 0.0
        model.eval()
        for i, data in enumerate(testloader):
            in1, labels = data
            in1 = in1.to(device)
            #in2 = in2.to(device)
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

        if lr_scheduler_:
            lr_scheduler(loss.cpu().numpy())        
        
        # Print
        if epoch % 1 == 0:
            print(f"Epoch: {epoch} Loss Training: {train_loss_per_epoch[-1]:.4f} Test: {test_loss_per_epoch[-1]:.4f}, Lr: {lr_scheduler.lr_scheduler.get_last_lr()[0]}")
                
    return model, train_loss_per_epoch, test_loss_per_epoch   

def plot_training(epochs, train_loss_per_epoch, test_loss_per_epoch):
    plt.figure()
    plt.plot(np.arange(epochs), train_loss_per_epoch, label="Train")
    plt.plot(np.arange(epochs), test_loss_per_epoch, label='Test')
    plt.legend()
    plt.show()
    return

def metrics(model, testloader):
    device = torch.device('cuda:0') #('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    y_true = []
    y_pred = []
    y_pred_proba = []
    model.eval()
    for data in testloader:
        in1, label = data
        in1 = in1.to(device)
        #in2 = in2.to(device)
        y_true.append(label.detach().cpu().tolist())

        output = model.forward(in1)
        y_pred.append(torch.argmax(output, 1).detach().cpu().tolist())

        proba_layer = nn.Softmax(dim=1)
        y_proba = proba_layer(output)
        y_pred_proba.append(y_proba.detach().cpu().tolist())
    
    y_true1 = [v for lst in y_true for v in lst]
    y_pred1 = [v for lst in y_pred for v in lst]
    y_pred_proba1 = [v[1] for lst in y_pred_proba for v in lst]

    print(confusion_matrix(y_true1, y_pred1))
    print(classification_report(y_true1, y_pred1))
    print("Accuracy: ", accuracy_score(y_true1, y_pred1))
    print("ROC AUC:", roc_auc_score(y_true1, y_pred_proba1)) #[:,1]))
    return

def save_model(model, emb_size, len_cat, len_num, fname=None):
    # add meta data 
    if fname == None:
        fname = f'model_{time.time()}'
    torch.save(model.state_dict(), f'data/models/{fname}.pth')
    data_dict = {'emb_size':emb_size, 'len_cat':len_cat, 'len_num':len_num}
    joblib.dump(data_dict, f'data/models/{fname}.pkl')
    logger.info(f"Save model {fname} and data file")
    return

#%% Predict on sub data

def predict_sub(model, predloader, fname, save_=True):
    # Do predictions on the data
    y_pred_list = []
    device = 'cuda:0'
    model.eval()
    print("Starting")
    for i, data in enumerate(predloader):
        if i == 0:
            print('first data move')
        in1, labels = data
        in1 = in1.to(device)
        #in2 = in2.to(device)
        
        if i == 0:
            print('prediction')
        y_pred = model.forward(in1).detach().cpu().tolist()
        y_pred_list += y_pred
        if i % 25 == 0:
            print(len(y_pred_list))
        

    # Convert the prediction list (row x 2 cols) to a tensor
    pred_data_tensor = torch.FloatTensor(y_pred_list)
    # Run the prediction tensor through a softmax layer to get probabilities
    soft_pred = F.softmax(pred_data_tensor, dim=1) 
    # Check where the tensor is cpu:-1, gpu:0,1....
    #print(soft_pred.get_device())
    # Put tensor in dataframe
    df_pred = pd.DataFrame(soft_pred.detach(), columns=['pred0', 'Response'])

    if save_:
        # Prep to save prediction data
        # Load csv with index numbers
        df_org_pred = pd.read_csv('data/sub_df_bare.csv')
        # Combine dataframe for submission
        df_sub = pd.concat([df_org_pred, df_pred.Response], axis=1)

        # Save to parquet
        df_sub.to_parquet(f'data/submissions/{fname}.parquet')
        logger.info(f'Predicted on sub data with model {fname} and saved to file.')

    return df_pred

#%%
## TODO
# Get checkpoints
# Get tensorboard
# Add more logging data
# Save description and metrics

if __name__ == "__main__":
    n = int(1e6)
    BATCH_SIZE = 25000
    EPOCHS = 40
    lr = 1e-4
    bags = 10
    
    df_org_pred = pd.read_csv('data/sub_df_bare.csv')
    sub_test = np.zeros(len(df_org_pred))
    
    for i in range(bags):
        logger.info(f"Starting Run")
        logger.info(f'Data: n={n}, BATCH_SIZE={BATCH_SIZE}, Model: EPOOCHS={EPOCHS}, lr={lr}')
        start_t = time.time()
        print('Loading Data')
        trainloader, testloader, meta_data, predloader = \
                load_data(n=n, BATCH_SIZE=BATCH_SIZE, full=True, pred=True,
                        save_tmp=False, load_tmp=False)

        mid_t = time.time()
        print(f'Loading time: {mid_t - start_t}')
        logger.info(f'Loading Data time: {mid_t - start_t} ')
        
        # -----------------------------------------------------------
        print ('Training')
        nn_model = make_model(meta_data)
        nn_model_trained, train_loss_per_epoch, test_loss_per_epoch = \
            train(EPOCHS, nn_model, trainloader, testloader, \
                meta_data['label_imbalance'], lr, lr_scheduler_=True)
        # metrics = calc_test_metrics(trainloader)
            
        end_t = time.time()
        print(f'Prediction time: {end_t - start_t}')
        plot_training(EPOCHS, train_loss_per_epoch, test_loss_per_epoch)
        metrics(nn_model_trained, testloader)
        
        # -----------------------------------------------------------
        save_model(nn_model_trained, meta_data['emb_sizes'], meta_data['num_cat_cols'], meta_data['num_num_cols'],'nn_emb_fd_base')
        df_pred = predict_sub(nn_model_trained, predloader, 'sub_nn_emb_fd_all_emb_extvar', save_=False)
        sub_test += df_pred.Response.to_numpy() / bags

    df_org_pred['Response'] = sub_test   

# %%

df_org_pred.to_parquet(f'data/submissions/sub_ens_eq_10.parquet', index=False)


#%%

df_4r = pd.read_parquet('data/submissions/sub_ens_eq_10.parquet')
display(df_4r)

#%%

df1 = pd.read_parquet('data/submissions/sub_nn_emb_fd_VD1.parquet')
df0 = pd.read_parquet('data/submissions/sub_nn_emb_fd_VD0.parquet')
df_test = pd.read_csv('data/raw/test.csv')
df_test['Vehicle_Damage']  = df_test.Vehicle_Damage.replace({'No':0, 'Yes':1}).astype(np.int8)

df1 = df1.dropna().reset_index(drop=True)
df0 = df0.dropna().reset_index(drop=True)

s1 = df1.shape[0]
s2 = df0.shape[0]
print(s1, s2, s1+s2)
print(df_test.shape[0])
display(df_test.head(10))


#%%
df_test = df_test.loc[df_test.Vehicle_Damage == 1].reset_index(drop=True)
print(df_test.shape[0])

# %% make new prediction df with correct id

df_p1 = pd.DataFrame()
df_p1['id'] = df_test.id
print(df_p1.shape[0])

#del df_test
# %%

df_p1 = pd.concat([df_p1, df1.Response],axis=1)
print(df_p1.shape[0])
display(df_p1.head(5))
#del df

# %%
del df_test
df_test = pd.read_csv('data/raw/test.csv')
df_test['Vehicle_Damage']  = df_test.Vehicle_Damage.replace({'No':0, 'Yes':1}).astype(np.int8)

df_test = df_test.loc[df_test.Vehicle_Damage == 0].reset_index(drop=True)
display(df_test.head())
print(df_test.shape[0])

# %% make new prediction df with correct id

df_p0 = pd.DataFrame()
df_p0['id'] = df_test.id
df_p0 = pd.concat([df_p0, df0.Response],axis=1)

display(df_p0.tail())
print(df_p0.info())

# %%

df_sub = pd.concat([df_p1, df_p0], axis=0)
df_sub = df_sub.sort_values(by='id')
display(df_sub.head(10))
display(df_sub.tail(10))
print(df_sub.info())
print(df_sub.isnull().sum())

# %%
df_sub = df_sub.reset_index(drop=True)

df_sub.to_parquet('data/submissions/nn_emb_fd_VDsplit.parquet')


# %%


# %%



