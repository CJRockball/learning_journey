#%%
import pprint

%matplotlib inline
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

#%%

def get_mixer_data(data_size=100_000, random_seed=42):
  # We need to fix the random seed
  # to make colab runs repeatable.
  rng = np.random.RandomState(random_seed)
  country = rng.randint(200, size=[data_size, 1]) / 200.
  bananas = rng.randint(24, size=[data_size, 1]) / 24.
  cookbooks = rng.randint(6, size=[data_size, 1]) / 6.

  x = np.concatenate([country, bananas, cookbooks], axis=1)
  # # Create 1st-order terms.
  y = 0.1 * country + 0.1 * bananas + 0.1 * cookbooks
  # Create 2nd-order cross terms.
  y += 0.1 * country * bananas + 3.1 * bananas * cookbooks + (
        0.1 * cookbooks * cookbooks)

  return x, y

x, y = get_mixer_data()

#%%
num_train = 90000
Xtrain = pd.DataFrame(x[:num_train], columns=['country', 'bananas', 'cookbooks']) 
ytrain = pd.DataFrame(y[:num_train], columns=['target'])
Xvalid = pd.DataFrame(x[num_train:], columns=['country', 'bananas', 'cookbooks'])
yvalid = pd.DataFrame(y[num_train:], columns=['target'])


# %%
# Model
# Dataloader
# training
# eval

class DNN(nn.Module):
    def __init__(self, feature_in):
        super().__init__()
        self.dummy_model = nn.Sequential(
        nn.Linear(feature_in, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 1)
    )   
    
    def forward(self, x):
        out = self.dummy_model(x)
        return out

model = DNN(3)
print(model)


# %%

loss_fn = nn.MSELoss()  # binary cross entropy
optimizer = optim.Adam(model.parameters(), lr=0.001)

Xtraint = torch.tensor(Xtrain.values, dtype=torch.float32)
ytraint = torch.tensor(ytrain.values, dtype=torch.float32).reshape(-1, 1)
Xvalidt = torch.tensor(Xvalid.values, dtype=torch.float32)
yvalidt = torch.tensor(yvalid.values, dtype=torch.float32).reshape(-1, 1)


# %%

n_epochs = 10
batch_size = 500
 
for epoch in range(n_epochs):
    for i in range(0, len(Xtraint), batch_size):
        
        Xbatch = Xtraint[i:i+batch_size]
        y_pred = model(Xbatch)
        ybatch = ytraint[i:i+batch_size]
        loss = loss_fn(y_pred, ybatch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Finished epoch {epoch}, latest loss {loss}')
    
#%%

#print(model.dummy_model[0].weight)    
#print(dir(model))   

print(model.dummy_model[2].bias)


# %%

# compute accuracy (no_grad is optional)
with torch.no_grad():
    y_pred = model(Xvalidt)
 
mse_ = loss_fn(y_pred, yvalidt)
print(f"Accuracy {mse_}")

# %%

class CrossNet(nn.Module):
    """The Cross Network part of Deep&Cross Network model,
    which learns both low and high degree cross feature.
      Input shape
        - 2D tensor with shape: ``(batch_size, units)``.
      Output shape
        - 2D tensor with shape: ``(batch_size, units)``.
      Arguments
        - **in_features** : Positive integer, dimensionality of input features.
        - **input_feature_num**: Positive integer, shape(Input tensor)[-1]
        - **layer_num**: Positive integer, the cross layer number
        - **parameterization**: string, ``"vector"``  or ``"matrix"`` ,  way to parameterize the cross network.
        - **l2_reg**: float between 0 and 1. L2 regularizer strength applied to the kernel weights matrix
        - **seed**: A Python integer to use as random seed.
      References
        - [Wang R, Fu B, Fu G, et al. Deep & cross network for ad click predictions[C]//Proceedings of the ADKDD'17. ACM, 2017: 12.](https://arxiv.org/abs/1708.05123)
        - [Wang R, Shivanna R, Cheng D Z, et al. DCN-M: Improved Deep & Cross Network for Feature Cross Learning in Web-scale Learning to Rank Systems[J]. 2020.](https://arxiv.org/abs/2008.13535)
    """

    def __init__(self, in_features, layer_num=2, parameterization='vector', seed=1024, device='cpu'):
        super(CrossNet, self).__init__()
        self.layer_num = layer_num
        self.parameterization = parameterization
        if self.parameterization == 'vector':
            # weight in DCN.  (in_features, 1)
            self.kernels = nn.Parameter(torch.Tensor(self.layer_num, in_features, 1))
        elif self.parameterization == 'matrix':
            # weight matrix in DCN-M.  (in_features, in_features)
            self.kernels = nn.Parameter(torch.Tensor(self.layer_num, in_features, in_features))
        else:  # error
            raise ValueError("parameterization should be 'vector' or 'matrix'")

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
            if self.parameterization == 'vector':
                xl_w = torch.tensordot(x_l, self.kernels[i], dims=([1], [0]))
                dot_ = torch.matmul(x_0, xl_w)
                x_l = dot_ + self.bias[i] + x_l
            elif self.parameterization == 'matrix':
                xl_w = torch.matmul(self.kernels[i], x_l)  # W * xi  (bs, in_features, 1)
                dot_ = xl_w + self.bias[i]  # W * xi + b
                x_l = x_0 * dot_+ x_l  # x0 · (W * xi + b) +xl  Hadamard-product
            else:  # error
                raise ValueError("parameterization should be 'vector' or 'matrix'")
        x_l = torch.squeeze(x_l, dim=2)
        return x_l


#%%
#cross_out = self.crossnet(dnn_input)
#logit += self.dnn_linear(cross_out)

cross_model = CrossNet(3, layer_num=1, parameterization='matrix')
print(cross_model)

#%%

class DCN(nn.Module):
    def __init__(self, inp):
      super().__init__()
      self.dcn_model = nn.Sequential(
                                      CrossNet(inp, layer_num=1, parameterization='matrix'),
                                      nn.Linear(3, 1, bias=False)
                                    )

    def forward(self, x):
      out = self.dcn_model(x)
      return out

cross_model = DCN(3)
print(cross_model)

# %%

n_epochs = 10
batch_size = 500

loss_fn = nn.MSELoss()  # binary cross entropy
optimizer = optim.Adam(cross_model.parameters(), lr=0.001)

train_batch = 0.0
running_loss = []
for epoch in range(n_epochs):
    for i in range(0, len(Xtraint), batch_size):
        
        Xbatch = Xtraint[i:i+batch_size]
        y_pred = cross_model(Xbatch)
        ybatch = ytraint[i:i+batch_size]
        loss = loss_fn(y_pred, ybatch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_batch += loss.item()
        
    train_epoch_loss = train_batch/batch_size    
    running_loss.append(train_epoch_loss) 
        
    print(f'Finished epoch {epoch}, latest loss {loss}')

# %%

# compute accuracy (no_grad is optional)
with torch.no_grad():
    y_pred = model(Xvalidt)
 
mse_ = loss_fn(y_pred, yvalidt)
print(f"Accuracy {mse_}")

#%%
mat =  cross_model.dcn_model[0].kernels.detach().numpy().reshape(3,3)
print(mat)

features = ["country", "bananas", "cookbooks"]

plt.figure(figsize=(9,9))
im = plt.matshow(np.abs(mat), cmap=plt.cm.Blues)
ax = plt.gca()
divider = make_axes_locatable(plt.gca())
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax=cax)
cax.tick_params(labelsize=10) 
_ = ax.set_xticklabels([''] + features, rotation=45, fontsize=10)
_ = ax.set_yticklabels([''] + features, fontsize=10)



# %%

a = ()
print(len(a))
print(type(a))

# %%
