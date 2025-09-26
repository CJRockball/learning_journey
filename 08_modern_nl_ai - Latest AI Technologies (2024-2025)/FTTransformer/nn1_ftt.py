"""
"""
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, time, random
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import math
import gc

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything()

#%% import data

df_train = pd.read_csv('data/ra/train.csv')
df_org = pd.read_csv('data/ra/Fertilizer Prediction.csv')
df_test = pd.read_csv('data/ra/test.csv')

df_train = df_train.drop(columns=['id'])
df_test = df_test.drop(columns=['id'])

df_train = pd.concat([df_train, df_org], axis=0)

#df_train = df_train.iloc[:100000,:]
#df_test = df_test.iloc[:10000,:]

target = ['Fertilizer Name']
cats = ['Soil Type', 'Crop Type', 'Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
nums = []


#%% change classes for target

# Transform target
df_train[target[0]], list_cat_names = df_train[target[0]].factorize()
target_dict = {i:name for i, name in enumerate(list_cat_names)}


# Tranform features
def feature_trans(df, df1, features):
    train_len = len(df)
    df_total = pd.concat([df, df1], axis=0)

    for name in features:
        df_total[name], _ = df_total[name].factorize()
        
    df = df_total.iloc[:train_len,:]
    df1 = df_total.iloc[train_len:,:-1]
    print('df_train ', df.shape)
    print('df_test ', df1.shape)
    return df, df1

df_train, df_test = feature_trans(df_train, df_test, cats)
card_list = [df_train[name].nunique() for name in cats]


# Normalize, standardize numerical features
m = df_train[nums].mean()
s = df_train[nums].std()
df_train[nums] = (df_train[nums] - m) / s
df_test[nums]  = (df_test[nums]  - m) / s

#%% Dataset Classes
class FertilizerDataset(Dataset):
    """Dataset for fertilizer data with numerical and categorical features."""
    def __init__(self, df_X_num, df_X_cat, df_y):
        self.num_features = torch.tensor(df_X_num.values, dtype=torch.float32)
        self.cat_features = torch.tensor(df_X_cat.values, dtype=torch.long)
        self.targets = torch.tensor(df_y.values, dtype=torch.long)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        num_val = self.num_features[idx]
        cat_val = self.cat_features[idx]
        y = self.targets[idx]
        return [num_val, cat_val, y]


#%% Early Stopping
class EarlyStopping:
    """Early stopping with model checkpointing to prevent overfitting."""
    def __init__(self, patience=10, min_delta=1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, val_loss, model):
        score = val_loss

        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict().copy()
        elif score >= self.best_score - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = model.state_dict().copy()
            self.counter = 0

    def load_best_model(self, model):
        model.load_state_dict(self.best_model_state)
        
#%% Feature Tokenizer
class FeatureTokenizer(nn.Module):
    """Converts numerical and categorical features into embeddings (tokens)."""
    def __init__(self, n_num_features, cat_cardinalities, d_token=10):
        super().__init__()
        self.n_num_features = n_num_features
        self.cat_cardinalities = cat_cardinalities or []
        self.d_token = d_token

        # Numerical: Each feature gets its own linear embedding
        if n_num_features > 0:
            self.num_embeddings = nn.ModuleList([
                nn.Linear(1, d_token) for _ in range(n_num_features)
            ])

        # Categorical: Each feature gets its own embedding table
        if self.cat_cardinalities:
            self.cat_embeddings = nn.ModuleList([
                nn.Embedding(card, d_token) for card in cat_cardinalities
            ])

    def forward(self, x_num=None, x_cat=None):
        tokens = []

        # Process numerical features
        if x_num is not None and self.n_num_features > 0:
            for i in range(self.n_num_features):
                feature_val = x_num[:, i:i+1]  # [batch_size, 1]
                token = self.num_embeddings[i](feature_val)  # [batch_size, d_token]
                tokens.append(token)

        # Process categorical features
        if x_cat is not None and self.cat_cardinalities:
            for i, emb in enumerate(self.cat_embeddings):
                token = emb(x_cat[:, i])  # [batch_size, d_token]
                tokens.append(token)

        # Stack tokens: [batch, n_features, d_token]
        if tokens:
            tokens = torch.stack(tokens, dim=1)
        else:
            batch_size = x_num.size(0) if x_num is not None else x_cat.size(0)
            tokens = torch.empty(batch_size, 0, self.d_token)

        return tokens
    

# Single-Head Self-Attention Layer
class SingleHeadSelfAttention(nn.Module):
    """Single-head self-attention for FT-Transformer."""
    def __init__(self, d_token, dropout=0.1):
        super().__init__()
        self.d_token = d_token
        self.scale = 1.0 / math.sqrt(d_token)

        # Combined QKV projection for efficiency
        self.qkv_proj = nn.Linear(d_token, d_token * 3, bias=False)
        self.out_proj = nn.Linear(d_token, d_token)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [batch, n_features, d_token]
        qkv = self.qkv_proj(x)  # [batch, n_features, 3*d_token]
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        # Compute attention scores with scaling
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values and project output
        attn_output = torch.matmul(attn_weights, v)
        out = self.out_proj(attn_output)
        return out


# FT-Transformer Block
class FTTransformerBlock(nn.Module):
    """FT-Transformer block: PreNorm, single-head attention, and MLP with residuals."""
    def __init__(self, d_token, mlp_hidden_mult=2, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_token)
        self.attn = SingleHeadSelfAttention(d_token, dropout)
        self.norm2 = nn.LayerNorm(d_token)

        # Feed-forward network
        d_ffn = int(d_token * mlp_hidden_mult)
        self.mlp = nn.Sequential(
            nn.Linear(d_token, d_ffn),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ffn, d_token),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Attention block with residual connection (PreNorm)
        x = x + self.attn(self.norm1(x))
        # Feed-forward block with residual connection (PreNorm)
        x = x + self.mlp(self.norm2(x))
        return x
    

# FT-Transformer Model
class FTTransformer(nn.Module):
    """FT-Transformer for fertilizer multi-class classification."""
    def __init__(self, n_num_features, cat_cardinalities, n_classes=7, d_token=10, dropout=0.1):
        super().__init__()
        self.n_classes = n_classes
        self.d_token = d_token

        # Feature tokenizer
        self.tokenizer = FeatureTokenizer(n_num_features, cat_cardinalities, d_token)

        # Add a [CLS] token for global pooling
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))
        nn.init.normal_(self.cls_token, std=0.02)

        # Single transformer block (as specified)
        self.transformer = FTTransformerBlock(d_token, dropout=dropout)

        # Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_token, n_classes)
        )

    def forward(self, x_num=None, x_cat=None):
        # Tokenize features: [batch, n_features, d_token]
        tokens = self.tokenizer(x_num, x_cat)
        batch_size = tokens.size(0)

        # Prepend [CLS] token: [batch, n_features+1, d_token]
        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)

        # Pass through single transformer block
        x = self.transformer(tokens)

        # Use [CLS] token for prediction
        cls_out = x[:, 0, :]  # [batch, d_token]
        logits = self.head(cls_out)  # [batch, n_classes]

        return logits


#%% Training and Validation Functions
def train(model, loader, optimizer, criterion, device):
    """Training function for one epoch."""
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    model.train()
    for data in loader:
        num_features, cat_features, labels = data[0].to(device), data[1].to(device), data[2].to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(num_features, cat_features)
        loss = criterion(outputs,labels.flatten())

        # Backward pass
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()
        running_loss += loss.item()

    training_loss = running_loss / len(loader)
    training_accuracy = correct_predictions / total_samples
    return training_loss, training_accuracy

def validate(model, loader, criterion, device):
    """Validation function for one epoch."""
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    all_predictions = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for data in loader:
            num_features, cat_features, labels = data[0].to(device), data[1].to(device), data[2].to(device)

            # Forward pass
            outputs = model(num_features, cat_features)
            loss = criterion(outputs, labels.flatten())

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            running_loss += loss.item()

            # Store predictions and labels for detailed metrics
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    validation_loss = running_loss / len(loader)
    validation_accuracy = correct_predictions / total_samples

    return validation_loss, validation_accuracy, all_predictions, all_labels


#%% Training/Validating
def plot_data(train_d, valid_d):
    xx = np.arange(len(train_d))
    plt.figure()
    plt.plot(xx, train_d, label='Train', color='navy')
    plt.plot(xx, valid_d, label='Validation', color='darkgreen')
    plt.legend()
    plt.show()
    return

df_y = df_train[target]
df_X = df_train.drop(columns=target).copy()
Xtrain, Xvalid, ytrain, yvalid = train_test_split(df_X, df_y, test_size=0.2, random_state=42)

EPOCHS = 40
LR = 3e-5
BATCH_SIZE = 256
PATIENCE = 5
DEVICE = torch.device('cuda') #'cpu') #

# SET UP DATA standard dataset, dataloader functions
traindataset = FertilizerDataset(Xtrain[nums], Xtrain[cats], ytrain)
validdataset = FertilizerDataset(Xvalid[nums], Xvalid[cats], yvalid)
trainloader = DataLoader(traindataset, batch_size=BATCH_SIZE)
validloader = DataLoader(validdataset, batch_size=BATCH_SIZE)

#%%

# DEF MODEL
model = FTTransformer(len(nums), card_list).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
early_stopping = EarlyStopping(patience=PATIENCE)
#scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

train_epoch_list = []
valid_epoch_list = []
start_time = time.time()
for epoch in range(EPOCHS):
    training_loss, training_accuracy = train(model, trainloader, optimizer, criterion, DEVICE)
    validation_loss, validation_accuracy, all_predictions, all_labels = validate(model, validloader, criterion, DEVICE)
    print(f'Epoch: {epoch}/{EPOCHS}, Train loss: {training_loss:.6f}, Validation loss: {validation_loss:.6f}') #, Validation rmsle: {validation_rmsle:.6f}')
    train_epoch_list.append(training_loss)
    valid_epoch_list.append(validation_loss)

    early_stopping(validation_loss, model)
    
    if early_stopping.early_stop:
        print("Early stopping")
        break
early_stopping.load_best_model(model)

        
plot_data(train_epoch_list, valid_epoch_list)

end_time = time.time()
print(f'Total time: {end_time - start_time}')





# %%
