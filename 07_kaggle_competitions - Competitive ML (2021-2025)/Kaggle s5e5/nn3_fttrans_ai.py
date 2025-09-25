#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, time, random
from tqdm import tqdm

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_log_error, root_mean_squared_error
from sklearn.metrics import root_mean_squared_error
from itertools import combinations, combinations_with_replacement

from IPython.display import display

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

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

#%% Load data

df_train = pd.read_csv('data/ra/train.csv').drop(columns=['id'])
df_train_org = pd.read_csv('data/ra/calories.csv').drop(columns=['User_ID'])
df_train_org['Sex'] = df_train_org['Gender']
df_train_org = df_train_org.drop(columns=['Gender'])
df_train = pd.concat([df_train, df_train_org], axis=0).reset_index(drop=True)
df_train = df_train.drop_duplicates().reset_index(drop=True)
df_test =  pd.read_csv('data/ra/test.csv').drop(columns=['id'])

sex_transform = {'male':0, 'female':1}
df_train['Sex'] = df_train['Sex'].replace(sex_transform) #.astype('category')
df_test['Sex'] = df_test['Sex'].replace(sex_transform) #.astype('category')

display(df_train)
print(df_train.shape)
# display(df_train_org)

target = ['Calories']
nums = [name for name in df_train.columns if name not in [target[0]]]
cats = []
features = nums + cats

# %% add 2x cross

# ncols = range(len(nums))
# comb_list2 = list(combinations_with_replacement(ncols, 2))
# print('adding: ', len(comb_list2))
# cross_cols = []
# for nr1, nr2 in comb_list2:
#     name1, name2 = nums[nr1], nums[nr2]
#     df_train[f'{name1}-m-{name2}'] = df_train[name1] * df_train[name2]
#     df_test[f'{name1}-m-{name2}'] = df_test[name1] * df_test[name2]
#     cross_cols += [f'{name1}-m-{name2}']

# nums = nums + cross_cols


#%%Torch classes and model
# Fast Loader is for batches.


class FastDataset(Dataset):
    def __init__(self, dfX, dfy, num_cols, cat_cols):
        self.cat_features = torch.tensor(dfX.loc[:,cat_cols].values, dtype=torch.long)
        self.num_features = torch.tensor(dfX.loc[:,num_cols].values, dtype=torch.float32)
        self.dfy = torch.tensor(dfy.values, dtype=torch.float32)
         
    def __len__(self):
        return len(self.dfy)
    
    def __getitem__(self,idx, batch_size):
        cat_val = self.cat_features[idx:idx+batch_size,:]
        num_val = self.num_features[idx:idx+batch_size,:]
        #X_out   = torch.concat([num_val, cat_val], axis=1)
        y       = self.dfy[idx:idx+batch_size]
        return [num_val, cat_val, y]

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

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = self.ds.__getitem__(self.i, self.batch_size)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches

class StdDataset(Dataset):
    def __init__(self, dfX, dfy, num_cols, cat_cols):
        self.cat_features = torch.tensor(dfX.loc[:,cat_cols].values, dtype=torch.long)
        self.num_features = torch.tensor(dfX.loc[:,num_cols].values, dtype=torch.float32)
        self.dfy = torch.tensor(dfy.values, dtype=torch.long)
        
    def __len__(self):
        return len(self.dfy)
    
    def __getitem__(self, idx):
        cat = self.cat_features[idx]
        num = self.num_features[idx]
        y = self.dfy[idx]
        return [num, cat, y]


#%% 


class EarlyStopping:
    def __init__(self, patience=1):
        self.patience = patience
        #print(self.patience)
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None
        
    def __call__(self, val_loss, model):
        score = val_loss
        if self.best_score is None:
            self.best_score = score
            #self.best_model_state = model.state_dict()
            torch.save(model.state_dict(), 'models/best.pt')
            #print('first best score')
        elif score >= self.best_score:
            self.counter += 1
            #print('counter', self.counter)
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            #self.best_model_state = model.state_dict()
            torch.save(model.state_dict(), 'models/best.pt')
            self.counter = 0
            #print('counter reset')
            
    def load_best_model(self, model):
#        model.load_state_dict(self.best_model_state)           
        model_data = torch.load('models/best.pt', weights_only=False)
        model.load_state_dict(model_data)       

#%%

class FeatureTokenizer(nn.Module):
    def __init__(self, n_num_features, cat_cardinalities, d_token=8):
        """ Feature Tokenizer for FT-Transform
    
        Args:
            n_num_features: Number of numerial features
            cat_cardinalities: List of cardinalities for categorical features
            d_token: Embedding dimension for each token
        """
    
        super().__init__()
        self.n_num_features = n_num_features
        self.cat_cardinalities = cat_cardinalities or []
        self.d_token = d_token
        
        # Numerival feature embedding - separate linear layer per feature
        if n_num_features > 0:
            self.num_embeddings = nn.ModuleList([
                nn.Linear(1, d_token) for _ in range(n_num_features)
            ])
        
        # Categorical feature embeddings - Lookup tables
        if cat_cardinalities:
            self.cat_embeddings = nn.ModuleList([
                nn.Embedding(cardinality, d_token) 
                for cardinality in cat_cardinalities
            ])
            
    def forward(self, x_num=None, x_cat=None):
        """ Transform numerical and categorical features to tokens
        
        Args:
            x_num: Numerical feature [batch_size, n_num_features]
            x_cat: Categorical feature [batch_size, n_cat_features]
        
        Returns:
            tokens: tensor [batch_size, n_features, d_tokens]
        """
        
        tokens = []
        
        # Process numerical featues
        if x_num is not None and self.n_num_features > 0:
            for i in range(self.n_num_features):
                # Each numerical featue gets its own embedding layer
                feature_val = x_num[:, i:i+1] # [batch_size, 1]
                token = self.num_embeddings[i](feature_val) # [batch_size, d_token]
                tokens.append(token)        

        # Process categorical features
        if x_cat is not None and self.cat_cardinalities:
            for i, embedding_layer in enumerate(self.cat_embeddings):
                token = embedding_layer(x_cat[:,i]) # [batch_Size, d_token]
                tokens.append(token)
        if tokens:
            tokens = torch.stack(tokens, dim=1) # [batch_size, n_features, d_token]
        else:
            # Handle edge case where no features are provided
            batch_size = x_num.size(0) if x_num is not None else x_cat.size(0)
            tokens = torch.empty(batch_size, 0, self.d_token)    
        
        return tokens
                
                
class FTTransform(nn.Module):
    def __init__(self, n_num_features, cat_cardinalities,
                 d_token=8, n_layers=3, n_heads=8, dropout=0.1):
        """ Complete FT-Transform implementation
        
        Args:
            n_num_features: Number of numerical features
            cat_cardinalities: List of cardinalities for categorical features
            d_token: Token embedding dimensions
            n_layers: Number of transformer layers
            n_heads: Number of attention heads
            dropout: Dropout probabilities
        """
        super().__init__()
        
        self.feature_tokenizer = FeatureTokenizer(
            n_num_features, cat_cardinalities, d_token
        )
        
        # CLS token for classification
        self.cls_token = nn.Parameter(torch.randn(1,1,d_token))
        
        # Transform encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=d_token * 4,
            dropout=dropout,
            activation='gelu',  # Smoother activation
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        
        # Classification head
        self.ln = nn.LayerNorm(d_token)  
        self.head = nn.Linear(d_token, 1) # Single output for regression
        
    def forward(self, x_num=None, x_cat=None):
        """ Forward pass through FT-Transformer
        
        Args:
            x_num: Numerical features [batches_size, n_num_features]
            x_cat: Categorical features [batch_size, n_cat_features]
             
        Returns:
            out: Regression value [batch_size, value]
        """
                
        batch_size = x_num.size(0) if x_num is not None else x_cat.size(0)
        
        # Tokenize features
        tokens = self.feature_tokenizer(x_num, x_cat) # [batch_size, n_features, d_token]
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1) # [batch_size, 1, d_token]
        tokens = torch.cat([cls_tokens, tokens], dim=1)        
        
        # Apply transformer layer
        tokens = self.transformer(tokens) # [batch_size, n_features+1, d_token]      
                
        # Use CLS token for predition
        cls_output = tokens[:,0] # [batch_size, d_token]
        cls_output = self.ln(cls_output)
        predictions = self.head(cls_output).squeeze(-1) # Remove last dimension
        
        return predictions
                

class MLPTab(nn.Module):
    def __init__(self, n_num_features, cat_cardinalities, d_token=8, 
                 hidden_dims=[256, 128, 32], dropout=0.1, output_dim=1):
        """ MLP for tabular data with tokenized input, both num and cat.
        
        Args:
            n_num_features: Number of numerical features
            cat_cardinalities: List of categorical feature cardinalities
            d_toke: Number of embedding dimensions
            hidden_dims: List of hidden layer sizes
            dropout: Dropout rate
            output_dim: Output dimension (1 for regression 2 or higher for classification)
        """
        super().__init__()
        self.feature_tokenizer = FeatureTokenizer(
            n_num_features, cat_cardinalities, d_token
        )

        # Total number of feature after tokinzation
        n_features = n_num_features + (len(cat_cardinalities) if cat_cardinalities else 0)
        input_dim = n_features * d_token
        
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_num=None, x_cat=None):
        """ Forward pass
        Args:
            x_num: tensor [batch_size, n_num_features]
            x_cat: tensor [batch_size, n_cat_features]
        Returns:
            out: tensor [batch_size, output_dim]        
        """

        # Tokenize features
        tokens = self.feature_tokenizer(x_num, x_cat) # [batch_size, n_features, d_token]
        # Flatten all tokens to a single col for linear layer to work
        x1 = tokens.flatten(1) # [batch size, n_features * d_token]
        out = self.mlp(x1)
        return out





#%%
def get_postsplit_meta(Xtrain, meta_data):
    '''Embedding cardinality is a list of two-tuples. First is no of unique values in a cat,
        the second is the number dimensions used to embedd'''
    embedding_cardinality = {n: len(c.unique()) for n,c in Xtrain[meta_data['CATS']].items()}
    emb_sizes = [(size, min(50, (size+1) // 2 )) for item, size in embedding_cardinality.items()]
    meta_data['emb_sizes'] = emb_sizes
    return meta_data

 
def train(model, loader, optimizer, criterion, DEVICE):
    running_loss = 0.0
    model.train()
    for data in loader: #tqdm(loader):
        in1, in2, label = data[0].to(DEVICE), data[1].to(DEVICE), data[2].to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        
        output = model.forward(in1, in2)
        mse_ = criterion(output, label.float()) #torch.flatten(output)
        loss = torch.sqrt(mse_)
        
        loss.backward()
        optimizer.step()
        running_loss += mse_.item()
        
    training_loss = np.sqrt(running_loss/len(loader))
    return training_loss


def valid(model, loader, criterion, DEVICE):
    y_prediction = []
    y_true = []
    running_loss = 0.0
    model.eval()
    for data in loader: #tqdm(loader):
        in1, in2, label = data[0].to(DEVICE), data[1].to(DEVICE), data[2].to(DEVICE)
        
        output = model.forward(in1, in2)
        loss = criterion(output, label.float())
        running_loss += loss.item()
        
        y_prediction.append(output.detach().cpu().tolist())
        y_true.append(label.detach().cpu().tolist())
    
    
    # Flatten prediction and labels    
    y_true1 = np.array([v for lst in y_true for v in lst])
    y_prediction1 = np.array([v for lst in y_prediction for v in lst])
    
    validation_loss = np.sqrt(running_loss/len(loader))
    validation_rmsle = root_mean_squared_error( y_true1, y_prediction1 )
    return validation_loss, validation_rmsle, y_prediction1 


def test_predictions(model, loader, DEVICE):
    y_prediction = []
    model.eval()
    for data in tqdm(loader):
        in1, in2, label = data[0].to(DEVICE), data[1].to(DEVICE), data[2].to(DEVICE)
        output = model.forward(in1, in2)
        
        y_prediction.append(output.detach().cpu().tolist())
        
    y_prediction1 = np.array([v for lst in y_prediction for v in lst])

    return y_prediction1

#%% Set up data
def plot_data(train_d, valid_d):
    xx = np.arange(len(train_d))
    plt.figure()
    plt.plot(xx, train_d, label='Train', color='navy')
    plt.plot(xx, valid_d, label='Validation', color='darkgreen')
    plt.legend()
    plt.show()
    return


# Split data
df_y = np.log1p(df_train[target].copy())
df_X = df_train.drop(columns=target)
Xtrain, Xvalid, ytrain, yvalid = train_test_split(df_X, df_y, test_size=0.2, random_state=42)

# Make meta-data
CATS = cats
NUM = nums
meta_data = {}
meta_data['NUM'] = NUM
meta_data['CATS'] = CATS
meta_data['num_cats'] = len(CATS)
meta_data['num_nums'] = len(NUM)
# Use category for embedding
# Made sure there are no "new" features in Xtest
meta_data = get_postsplit_meta(df_X, meta_data)
meta_data['n_cat_feat'] = [a for a,b in meta_data['emb_sizes']]


#%%
import warnings
warnings.filterwarnings("ignore")

EPOCHS = 250
LR = 1e-4
BATCH_SIZE = 1024
PATIENCE = 20
DEVICE = torch.device('cuda:0') #'cpu') #


KFOLD = 5
kf = KFold(n_splits=KFOLD, shuffle=True, random_state=42)

start_time = time.time()
oof = np.zeros(len(df_X))
fold_metric = []
for i, (train_idx, valid_idx) in enumerate(kf.split(df_X)):
    print(f'#### FOLD {i} ####')
    Xtrain = df_X.loc[train_idx].copy()
    ytrain = df_y.loc[train_idx].copy()
    Xvalid = df_X.loc[valid_idx].copy()
    yvalid = df_y.loc[valid_idx].copy()
    Xtest = df_test.copy()
     
    m = Xtrain[nums].mean()
    s = Xtrain[nums].std()
    Xtrain[nums] = (Xtrain[nums] - m) / s
    Xvalid[nums] = (Xvalid[nums] - m) / s
    Xtest[nums]  = (Xtest[nums]  - m) / s
 
    # SET UP DATA standard dataset, dataloader functions
    traindataset = FastDataset(Xtrain, ytrain, meta_data['NUM'], meta_data['CATS'])
    validdataset = FastDataset(Xvalid, yvalid, meta_data['NUM'], meta_data['CATS'])
    trainloader = FastDataLoader(traindataset, batch_size=BATCH_SIZE)
    validloader = FastDataLoader(validdataset, batch_size=BATCH_SIZE)


    # DEF MODEL
    model = FTTransform(len(nums), meta_data['n_cat_feat'], d_token=8, n_layers=3, n_heads=8, dropout=0.1).to(DEVICE) 
    # MLPTab(len(nums), meta_data['n_cat_feat'], d_token=8, 
    #              hidden_dims=[256, 128, 32], dropout=0.1, output_dim=1).to(DEVICE)
    
    #Model(meta_data, [256, 128, 32], 3*[0.1], 10).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR) #, weight_decay=1e-5)
    early_stopping = EarlyStopping(patience=PATIENCE)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)


    train_epoch_list = []
    valid_epoch_list = []
    for epoch in range(EPOCHS):
        train_data = train(model, trainloader, optimizer, criterion, DEVICE)
        validation_loss, validation_rmsle, _ = valid(model, validloader, criterion, DEVICE) #, oof, val_idx)

        scheduler.step(validation_loss)
        lrs = optimizer.param_groups[0]['lr']
        
        
        print(f'Epoch: {epoch}/{EPOCHS}, Train loss: {train_data:.6f}, Validation loss: {validation_loss:.6f}, LR: {lrs:.6f}')
        #train_epoch_list.append(train_data)
        #valid_epoch_list.append(validation_loss)


        early_stopping(validation_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    early_stopping.load_best_model(model)

        
#plot_data(train_epoch_list, valid_epoch_list)

    validation_loss, validation_rmsle, oof_pred = valid(model, validloader, criterion, DEVICE)
    print(f'RMSLE: {validation_rmsle}')
    fold_metric.append(validation_rmsle)
    oof[valid_idx] = oof_pred.flatten()
    
    ydummy = pd.DataFrame(data=np.zeros((Xtest.shape[0],1)), columns=target) 
    testdataset = FastDataset(Xtest, ydummy, meta_data['NUM'], meta_data['CATS'])
    testloader = FastDataLoader(testdataset, batch_size=BATCH_SIZE)

    y_pred = test_predictions(model, testloader, DEVICE)
    if i > 0:
        preds += y_pred
    else:
        preds = y_pred

    
preds /= KFOLD
end_time = time.time()
print(f'Total time: {end_time - start_time}')
print(fold_metric)
print(np.mean(fold_metric))

#%%
fname = 'nn5_cv0.0616'

y_preds_nn = np.expm1(preds)
print('predict mean :',y_preds_nn.mean())
print('predict median :',np.median(y_preds_nn))

y_preds_nn = np.clip(y_preds_nn,1,314)
print('predict mean after clip:',y_preds_nn.mean())
print('predict median after clip:',np.median(y_preds_nn))


#%%
df_sub = pd.read_csv('data/ra/sample_submission.csv')
df_sub[target] = y_preds_nn

df_sub.to_csv(f'submissions/{fname}.csv', index=False)

df_check = pd.read_csv(f'submissions/{fname}.csv')
display(df_check)


lgb_oof_conv = np.expm1(oof)
df_lgb_oof = pd.DataFrame(data=lgb_oof_conv, index=df_train.index, columns=[fname])
df_lgb_oof.to_csv(f'oof/{fname}.csv', index=False)

df_check = pd.read_csv(f'oof/{fname}.csv')
display(df_check)

# %%

print(model)


+# %%
               
                
                
