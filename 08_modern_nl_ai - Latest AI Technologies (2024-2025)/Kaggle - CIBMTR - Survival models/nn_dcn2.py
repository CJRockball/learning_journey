#%%
import pprint

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

#%% Create data

def get_mixer_data(data_size=100_000, random_seed=1337):
  # We need to fix the random seed
  # to make colab runs repeatable.
  rng = np.random.RandomState(random_seed)
  country = rng.randint(200, size=[data_size, 1]) #/ 200.
  bananas = rng.randint(24, size=[data_size, 1]) #/ 24.
  cookbooks = rng.randint(6, size=[data_size, 1]) #/ 6.

  x = np.concatenate([country, bananas, cookbooks], axis=1)
  # # Create 1st-order terms.
  #y = 0.1 * country + 0.4 * bananas + 0.7 * cookbooks
  # Create 2nd-order cross terms.
  y = 0.5 * country * bananas + 3.1 * bananas * cookbooks + (
        0.5 * cookbooks * cookbooks)

  return x, y

x, y = get_mixer_data()

#%% Normalize

num_train = 90000
Xtrain = pd.DataFrame(x[:num_train], columns=['country', 'bananas', 'cookbooks']) 
ytrain = pd.DataFrame(y[:num_train], columns=['target'])
Xvalid = pd.DataFrame(x[num_train:], columns=['country', 'bananas', 'cookbooks'])
yvalid = pd.DataFrame(y[num_train:], columns=['target'])

m = Xtrain.mean()
s = Xtrain.std()
Xtrain = (Xtrain - m) / s
Xtest = (Xvalid - m) / s


Xtraint = torch.tensor(Xtrain.values, dtype=torch.float32)
ytraint = torch.tensor(ytrain.values, dtype=torch.float32).reshape(-1, 1)
Xvalidt = torch.tensor(Xvalid.values, dtype=torch.float32)
yvalidt = torch.tensor(yvalid.values, dtype=torch.float32).reshape(-1, 1)

#%%

Xtrain_red = Xtraint[:10000,:]

print(Xtrain_red.shape)
Xt_sum =  torch.linalg.norm(Xtrain_red, dim = 0)
print(Xt_sum)
Xtrain_norm = Xtrain_red / Xt_sum
print(Xtrain_norm.shape)

Xt = Xtrain_norm.transpose(0,1)
print(Xt.shape)
dcn = Xt @ Xtrain_norm

print(dcn.detach())


# %%

class CrossNet(nn.Module):

    def __init__(self, in_features, layer_num=2, seed=1024, device='cpu'):
        super(CrossNet, self).__init__()
        self.layer_num = layer_num
        
        # weight matrix in DCN-M.  (in_features, in_features)
        self.kernels = nn.Parameter(torch.Tensor(self.layer_num, in_features, in_features))        
        self.bias = nn.Parameter(torch.Tensor(self.layer_num, in_features, 1))

        for i in range(self.kernels.shape[0]):
            nn.init.xavier_normal_(self.kernels[i])
        for i in range(self.bias.shape[0]):
            nn.init.zeros_(self.bias[i])

        self.to(device)

    def forward(self, inputs):
        x_0 = inputs.unsqueeze(2)
        x_l = x_0
        for i in range(self.layer_num):
            xl_w = torch.matmul(self.kernels[i], x_l)  # W * xi  (bs, in_features, 1)
            dot_ = xl_w + self.bias[i]  # W * xi + b
            x_l = x_0 * dot_ + x_l  # x0 · (W * xi + b) + xl  Hadamard-product
        x_l = torch.squeeze(x_l, dim=2)
        return x_l


#%%

class DCN(nn.Module):
    def __init__(self, inp):
      super().__init__()
      self.dcn_model = nn.Sequential(
                                      CrossNet(inp, layer_num=1),
                                      nn.Linear(3, 1, bias=False)
                                    )

    def forward(self, x):
      out = self.dcn_model(x)
      return out

cross_model = DCN(3)
print(cross_model)

# %%

n_epochs = 50
batch_size = 500

loss_fn = nn.MSELoss()
optimizer = optim.Adam(cross_model.parameters(), lr=0.001)

train_loss_num = 0.0
train_loss = []
for epoch in range(n_epochs):
    for i in range(0, len(Xtraint), batch_size):
        
        Xbatch = Xtraint[i:i+batch_size]
        y_pred = cross_model(Xbatch)
        ybatch = ytraint[i:i+batch_size]
        loss = loss_fn(y_pred, ybatch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss_num += loss.item()
        
    train_loss_batch = train_loss_num/batch_size
    train_loss.append(train_loss_batch)
    #print(f'Finished epoch {epoch}, latest loss {train_loss_batch}')
    train_loss_num = 0.0
    
plt.figure()
plt.plot(list(range(n_epochs)), train_loss)
plt.show()

# %%

# compute accuracy (no_grad is optional)
with torch.no_grad():
    y_pred = cross_model(Xvalidt)
 
mse_ = loss_fn(y_pred, yvalidt)
print(f"Accuracy {mse_}")

#%%
mat =  cross_model.dcn_model[0].kernels.detach().numpy().reshape(3,3)
print(np.round(mat,2))

features = ["country", "bananas", "cookbooks"]

plt.figure(figsize=(9,9))
im = plt.matshow(np.abs(mat), cmap=plt.cm.Blues)
ax = plt.gca()
divider = make_axes_locatable(plt.gca())
cax = divider.append_axes("right", size="5%", pad=0.1)
plt.colorbar(im, cax=cax)
cax.tick_params(labelsize=10) 
_ = ax.set_xticklabels([''] + features, rotation=45, fontsize=10)
_ = ax.set_yticklabels([''] + features, fontsize=10)


# %% Implement linear model
from sklearn.linear_model import Lasso
from sklearn.metrics import root_mean_squared_error

# Add data features
import itertools

feature_list = list(range(len(features)))
all_col_comb = list(itertools.combinations_with_replacement(feature_list,2))

print(all_col_comb)

#%%

# Make df for linear
Xtrain_l = Xtrain.copy()
Xvalid_l = Xvalid.copy()
ytrain_l = ytrain.copy()
yvalid_l = yvalid.copy()

# Add cross features for linear
for (i,j) in all_col_comb:
    name1, name2 = features[i], features[j]
    new_name = f'{name1}x{name2}'
    Xtrain_l[new_name] = Xtrain_l[name1] * Xtrain_l[name2]
    Xvalid_l[new_name] = Xvalid_l[name1] * Xvalid_l[name2]


# FIT LASSO MODEL
lin_model = Lasso(alpha=1e1) 
lin_model.fit(Xtrain_l, ytrain) 

lin_pred = lin_model.predict(Xvalid_l)
rmse_ = root_mean_squared_error(yvalid, lin_pred)

print('rmse: ', rmse_)
# print(lin_model.coef_)

df_coef = pd.DataFrame(data=lin_model.coef_, index=Xtrain_l.columns, columns=['coef'])
display(df_coef)



# %%
