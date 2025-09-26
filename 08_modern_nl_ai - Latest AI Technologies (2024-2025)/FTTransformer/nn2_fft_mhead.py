"""
"""
#%% =======================================================================
#  Imports + deterministic seed
#==========================================================================
import os, random, time, math
import numpy  as np
import pandas as pd
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm

def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
seed_everything()
DEVICE = torch.device("cuda") #torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%% =======================================================================
#  Data ingest + feature engineering
#==========================================================================
df_train = pd.read_csv('data/ra/train.csv')
df_org = pd.read_csv('data/ra/Fertilizer Prediction.csv')
df_test = pd.read_csv('data/ra/test.csv')

df_train = df_train.drop(columns=['id'])
df_test = df_test.drop(columns=['id'])

df = pd.concat([df_train, df_org], axis=0)
#df = df.iloc[:100000,:]

# column groups -----------------------------------------------------------
TARGET = ["Fertilizer Name"]
CATS   = ["Soil Type", "Crop Type","Temparature", "Humidity", "Moisture",
          "Nitrogen", "Potassium", "Phosphorous"]
NUMS   = []

# advances feature -------------------------------------------------------
def soil_crop_compatibility_flag(df: pd.DataFrame,
                                 soil_col: str = "Soil Type",
                                 crop_col: str = "Crop Type",
                                 flag_name: str = "soil_crop_compatibility"
) -> pd.DataFrame:
    """
    Add an agronomy-based compatibility score (0–1) for every row.

    Parameters
    ----------
    df        : DataFrame with two categorical columns,
                e.g. “Soil Type” and “Crop Type”.
    soil_col  : name of the soil column  (default “Soil Type”)
    crop_col  : name of the crop column  (default “Crop Type”)
    flag_name : name of the new compatibility column.

    Returns
    -------
    DataFrame  —  original df plus one extra float column.
    """

    # -------------------------------- knowledge table ------------------------------
    compat = {
        # Clayey soil
        ("Clayey", "Paddy"): 1.0,  ("Clayey", "Sugarcane"): 0.9,
        ("Clayey", "Cotton"): 0.7, ("Clayey", "Wheat"): 0.6,
        ("Clayey", "Maize"): 0.6,  ("Clayey", "Millets"): 0.4,
        ("Clayey", "Barley"): 0.5, ("Clayey", "Tobacco"): 0.5,
        ("Clayey", "Ground Nuts"): 0.3, ("Clayey", "Pulses"): 0.6,
        ("Clayey", "Oil seeds"): 0.5,

        # Sandy soil
        ("Sandy", "Ground Nuts"): 1.0, ("Sandy", "Millets"): 0.9,
        ("Sandy", "Cotton"): 0.8,      ("Sandy", "Oil seeds"): 0.8,
        ("Sandy", "Tobacco"): 0.7,     ("Sandy", "Barley"): 0.7,
        ("Sandy", "Pulses"): 0.7,      ("Sandy", "Wheat"): 0.6,
        ("Sandy", "Maize"): 0.6,       ("Sandy", "Sugarcane"): 0.4,
        ("Sandy", "Paddy"): 0.2,

        # Loamy soil
        ("Loamy", "Wheat"): 1.0,   ("Loamy", "Maize"): 1.0,
        ("Loamy", "Cotton"): 0.9,  ("Loamy", "Tobacco"): 0.9,
        ("Loamy", "Barley"): 0.9,  ("Loamy", "Pulses"): 0.8,
        ("Loamy", "Ground Nuts"): 0.8, ("Loamy", "Oil seeds"): 0.8,
        ("Loamy", "Sugarcane"): 0.8,   ("Loamy", "Millets"): 0.7,
        ("Loamy", "Paddy"): 0.7,

        # Red soil
        ("Red", "Millets"): 0.9,  ("Red", "Cotton"): 0.8,
        ("Red", "Ground Nuts"): 0.8, ("Red", "Oil seeds"): 0.7,
        ("Red", "Tobacco"): 0.7,  ("Red", "Maize"): 0.7,
        ("Red", "Pulses"): 0.7,   ("Red", "Wheat"): 0.6,
        ("Red", "Barley"): 0.6,   ("Red", "Sugarcane"): 0.6,
        ("Red", "Paddy"): 0.5,

        # Black soil
        ("Black", "Cotton"): 1.0, ("Black", "Sugarcane"): 0.8,
        ("Black", "Pulses"): 0.8, ("Black", "Wheat"): 0.7,
        ("Black", "Maize"): 0.7,  ("Black", "Oil seeds"): 0.7,
        ("Black", "Tobacco"): 0.6, ("Black", "Barley"): 0.6,
        ("Black", "Ground Nuts"): 0.6, ("Black", "Paddy"): 0.6,
        ("Black", "Millets"): 0.5,
    }

    # ----------------------------- compute score -----------------------------------
    def score(row):
        return compat.get((row[soil_col], row[crop_col]), 0.5)

    out = df.copy()
    out[flag_name] = out.apply(score, axis=1)
    return out

df = soil_crop_compatibility_flag(df)
df_test = soil_crop_compatibility_flag(df_test)

NUMS = NUMS + ['soil_crop_compatibility']

#%%

# label-encode target ------------------------------------------------------
# Transform target
df[TARGET[0]], list_cat_names = df[TARGET[0]].factorize()
target_dict = {i:name for i, name in enumerate(list_cat_names)}


# Tranform features
def feature_trans(df, df1, features):
    train_len = len(df)
    df_total = pd.concat([df, df1], axis=0)

    for name in features:
        df_total[name], _ = df_total[name].factorize()
        
    df = df_total.iloc[:train_len,:]
    df1 = df_total.iloc[train_len:,:]
    print('df ', df.shape)
    print('df_test ', df1.shape)
    return df, df1

df, df_test = feature_trans(df, df_test, CATS)
df_test = df_test.drop(columns=TARGET)
card_list = [df[name].nunique() for name in CATS]

# Normalize, standardize numerical features
m = df[NUMS].mean()
s = df[NUMS].std()
df[NUMS] = (df[NUMS] - m) / s
df_test[NUMS]  = (df_test[NUMS]  - m) / s

# train/valid split --------------------------------------------------------
train_df, valid_df = train_test_split(
    df, test_size=0.2, stratify=df[TARGET[0]], random_state=42
)

#%% =======================================================================
#  Torch dataset wrapper
#==========================================================================
class FertilizerDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame):
        self.X_num = torch.tensor(dataframe[NUMS].values, dtype=torch.float32)
        self.X_cat = torch.tensor(dataframe[CATS].values, dtype=torch.long)
        self.y     = torch.tensor(dataframe[TARGET].values, dtype=torch.long)

    def __len__(self): return len(self.y)

    def __getitem__(self, idx):
        return self.X_num[idx], self.X_cat[idx], self.y[idx]

BATCH = 256
train_loader = DataLoader(FertilizerDataset(train_df), batch_size=BATCH, shuffle=True)
valid_loader = DataLoader(FertilizerDataset(valid_df), batch_size=BATCH)

#%% =======================================================================
#  Feature-to-token layer
#==========================================================================
class FeatureTokenizer(nn.Module):
    """
    Numerical → linear projection | Categorical → embedding lookup
    All tokens share a common dimension (d_token = 128).
    """
    def __init__(self, n_num: int, cat_cardinalities: list[int], d_token: int = 128):
        super().__init__()
        self.num_proj  = nn.ModuleList([nn.Linear(1, d_token) for _ in range(n_num)])
        self.cat_embed = nn.ModuleList(
            [nn.Embedding(card, d_token) for card in cat_cardinalities]
        )

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        tokens = []
        for i, proj in enumerate(self.num_proj):
            tokens.append(proj(x_num[:, i : i + 1]))          # shape → [B, d_token]
        for i, emb in enumerate(self.cat_embed):
            tokens.append(emb(x_cat[:, i]))                   # shape → [B, d_token]
        return torch.stack(tokens, dim=1)                     # → [B, F, d_token]

#%% =======================================================================
#  Transformer building blocks
#==========================================================================
class TransformerBlock(nn.Module):
    """
    PreNorm → Multi-Head Self-Attention → MLP (with residual connections)
    """
    def __init__(self, d_token: int = 128, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_token)
        self.attn  = nn.MultiheadAttention(
            embed_dim=d_token, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_token)
        self.mlp   = nn.Sequential(
            nn.Linear(d_token, d_token * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_token * 2, d_token),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

#%% =======================================================================
#  FT-Transformer with 4 × 8-head blocks & 128-d tokens
#==========================================================================
class FTTransformer(nn.Module):
    def __init__(
        self,
        n_num: int,
        cat_cardinalities: list[int],
        d_token: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        n_classes: int = 7,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.tokenizer = FeatureTokenizer(n_num, cat_cardinalities, d_token)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))
        self.blocks    = nn.ModuleList(
            [TransformerBlock(d_token, n_heads, dropout) for _ in range(n_layers)]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_token, n_classes),
        )

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        tok   = self.tokenizer(x_num, x_cat)                 # [B, F, d]
        cls   = self.cls_token.expand(tok.size(0), -1, -1)   # [B, 1, d]
        x     = torch.cat([cls, tok], dim=1)                 # prepend [CLS]
        for blk in self.blocks: x = blk(x)
        return self.head(x[:, 0])                            # use [CLS] token

#%% =======================================================================
#  Train / validate loops + AdamW & cosine restarts
#==========================================================================
def step(loader, train: bool = False):
    model.train() if train else model.eval()
    total, correct, running_loss = 0, 0, 0.0
    with torch.set_grad_enabled(train):
        for x_num, x_cat, y in tqdm(loader, desc="Validation"):
            x_num, x_cat, y = x_num.to(DEVICE), x_cat.to(DEVICE), y.to(DEVICE).squeeze()
            if train: optimizer.zero_grad()
            logits = model(x_num, x_cat)
            loss   = criterion(logits, y)
            if train:
                loss.backward()
                optimizer.step()
            running_loss += loss.item() * y.size(0)
            preds  = logits.argmax(1)
            total += y.size(0)
            correct += (preds == y).sum().item()
    return running_loss / total, correct / total

# hyper-params -------------------------------------------------------------
EPOCHS     = 20
LR         = 3e-4
T0         = 5              # initial restart period
criterion  = nn.CrossEntropyLoss()
model      = FTTransformer(
    n_num=len(NUMS),
    cat_cardinalities=[df[c].nunique() for c in CATS],
    n_classes=len(target_dict),
    n_layers=2,
    dropout=0.2,
).to(DEVICE)

optimizer  = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler  = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T0)

# training loop ------------------------------------------------------------
history = {"train_loss": [], "valid_loss": []}
start_time = time.time()
for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_acc = step(train_loader, train=True)
    val_loss, val_acc = step(valid_loader, train=False)
    scheduler.step(epoch)                       # update LR each epoch
    history["train_loss"].append(tr_loss)
    history["valid_loss"].append(val_loss)
    print(f"Epoch {epoch:2d} | "
          f"train loss: {tr_loss:.4f}  acc: {tr_acc:.4f} || "
          f"valid loss: {val_loss:.4f}  acc: {val_acc:.4f}")

end_time = time.time()
print(f'Total time: {end_time - start_time}')




# %%
import matplotlib.pyplot as plt
print(history)

xx = range(len(history['train_loss']))
plt.figure()
plt.plot(xx, history['train_loss'], label='Train_loss')
plt.plot(xx, history['valid_loss'], label='Valid_loss')
plt.legend()
plt.grid()
plt.show()


#%%
# Modified Dataset class for data without targets
class FertilizerDatasetNoTarget(Dataset):
    """Dataset for fertilizer data with numerical and categorical features (no targets)."""
    def __init__(self, df_X_num, df_X_cat):
        self.num_features = torch.tensor(df_X_num.values, dtype=torch.float32)
        self.cat_features = torch.tensor(df_X_cat.values, dtype=torch.long)

    def __len__(self):
        return len(self.num_features)

    def __getitem__(self, idx):
        num_val = self.num_features[idx]
        cat_val = self.cat_features[idx]
        return [num_val, cat_val]


def predict_new_dataset_top3(model, dataloader, target_dict, device, save_to_file=True, filename='new_dataset_predictions.npz'):
    """
    Make top-3 predictions for new dataset and save to numpy array.
    
    Args:
        model: Trained FT-Transformer model
        dataloader: DataLoader for new dataset
        target_dict: Dictionary mapping class indices to fertilizer names
        device: Device to run predictions on
        save_to_file: Whether to save results to file
        filename: Filename for saved results
    
    Returns:
        numpy array with top-3 predictions (shape: [n_samples, 3])
    """
    model.eval()
    all_top3_predictions = []
    all_top3_probabilities = []
    sample_count = 0
    
    print("Making predictions on new dataset...")
    
    with torch.no_grad():
        for batch_idx, (num_x, cat_x) in enumerate(dataloader):
            num_x, cat_x = num_x.to(device), cat_x.to(device)
            
            # Get model predictions
            logits = model(num_x, cat_x)
            probabilities = F.softmax(logits, dim=1)
            
            # Get top-3 predictions for each sample in the batch
            top3_probs, top3_indices = torch.topk(probabilities, 3, dim=1)
            
            # Convert to fertilizer names
            batch_predictions = []
            batch_probabilities = []
            
            for i in range(num_x.size(0)):
                sample_predictions = []
                sample_probs = []
                
                for j in range(3):
                    fertilizer_idx = top3_indices[i][j].item()
                    fertilizer_name = target_dict[fertilizer_idx]
                    probability = top3_probs[i][j].item()
                    
                    sample_predictions.append(fertilizer_name)
                    sample_probs.append(probability)
                
                batch_predictions.append(sample_predictions)
                batch_probabilities.append(sample_probs)
                sample_count += 1
            
            all_top3_predictions.extend(batch_predictions)
            all_top3_probabilities.extend(batch_probabilities)
    
    # Convert to numpy arrays
    predictions_array = np.array(all_top3_predictions)  # Shape: [n_samples, 3]
    probabilities_array = np.array(all_top3_probabilities)  # Shape: [n_samples, 3]
    
    # Create space-separated string array for each sample
    predictions_spaced = np.array([' '.join(row) for row in predictions_array])
    
    print(f"Completed predictions for {sample_count} samples")
    print(f"Predictions array shape: {predictions_array.shape}")
    
    # Save to file if requested
    if save_to_file:
        np.savez(filename, 
                 top3_predictions=predictions_array,
                 top3_probabilities=probabilities_array,
                 predictions_spaced=predictions_spaced)
        print(f"Results saved to {filename}")
    
    return predictions_array, probabilities_array, predictions_spaced

def display_prediction_sample(predictions_array, probabilities_array, predictions_spaced, n_samples=10):
    """Display sample predictions in a readable format."""
    print(f"\n=== SAMPLE PREDICTIONS (First {n_samples} samples) ===")
    print("-" * 80)
    print(f"{'Sample':<8} {'Top 1':<12} {'Top 2':<12} {'Top 3':<12} {'Space-separated':<30}")
    print("-" * 80)
    
    for i in range(min(n_samples, len(predictions_array))):
        top1, top2, top3 = predictions_array[i]
        prob1, prob2, prob3 = probabilities_array[i]
        spaced = predictions_spaced[i]
        
        print(f"{i+1:<8} {top1:<12} {top2:<12} {top3:<12} {spaced:<30}")
        print(f"{'Conf:':<8} {prob1:.3f}        {prob2:.3f}        {prob3:.3f}")
        print()

def analyze_predictions_distribution(predictions_array, target_dict):
    """Analyze the distribution of predicted fertilizers."""
    print("\n=== PREDICTION DISTRIBUTION ANALYSIS ===")
    
    # Count frequency of each fertilizer in top-1 predictions
    top1_predictions = predictions_array[:, 0]
    unique, counts = np.unique(top1_predictions, return_counts=True)
    
    print("Top-1 Prediction Distribution:")
    for fertilizer, count in zip(unique, counts):
        percentage = (count / len(predictions_array)) * 100
        print(f"  {fertilizer:<12}: {count:>4} samples ({percentage:>5.1f}%)")
    
    # Count frequency across all top-3 predictions
    all_predictions = predictions_array.flatten()
    unique_all, counts_all = np.unique(all_predictions, return_counts=True)
    
    print("\nOverall Top-3 Distribution:")
    for fertilizer, count in zip(unique_all, counts_all):
        percentage = (count / (len(predictions_array) * 3)) * 100
        print(f"  {fertilizer:<12}: {count:>4} mentions ({percentage:>5.1f}%)")

#%%
# Make new dataloader for test data
dataset_test = FertilizerDatasetNoTarget(df_test[NUMS], df_test[CATS])
dataloader_test = DataLoader(dataset_test, batch_size=BATCH, shuffle=False)

# Do predictions
predictions_array, probabilities_array, predictions_spaced = predict_new_dataset_top3(
    model=model,
    dataloader=dataloader_test,
    target_dict=target_dict,
    device=DEVICE,
    save_to_file=True,
    filename='new_dataset_top3_predictions.npz'
)

# display
display_prediction_sample(predictions_array, probabilities_array, predictions_spaced, n_samples=15)

# Analyze prediction distribution
analyze_predictions_distribution(predictions_array, target_dict)

# %%
fname = 'nn5'

df_sub = pd.read_csv('data/ra/sample_submission.csv')
df_sub['Fertilizer Name'] = predictions_spaced
df_sub.to_csv(f'submissions/{fname}.csv', index=False)

df_check = pd.read_csv(f'submissions/{fname}.csv')
display(df_check)


#%%
display(df_test[CATS])

# %%
