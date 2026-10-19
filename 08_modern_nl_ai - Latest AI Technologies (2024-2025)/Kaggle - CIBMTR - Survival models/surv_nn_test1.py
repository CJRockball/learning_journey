#%%
import pandas as pd
pd.options.mode.copy_on_write = True
import numpy as np
import matplotlib.pyplot as plt


#%%

df_raw = pd.read_csv('data/other/support2.csv')

df = df_raw[['age', 'sex', 'race', 'num.co', 'diabetes',
             'dementia', 'ca', 'meanbp', 'hrt', 'resp',
             'temp', 'wblc', 'sod', 'crea',
             'd.time', 'death']] #.astype({'race': 'category'})

ord = ['sex', 'race', 'ca']
df['sex'] = df['sex'].map({'female':1, 'male':0})
df['ca'] = df['ca'].map({'metastatic':2, 'no':0, 'yes':1})
#df['race'] = df['race'].map({'other':0, 'white':1, 'black':2, 'hispanic':3,'asian':4})

std_features = ['age','meanbp','hrt','resp','temp','wblc','sod','crea']
ohe_features = ['race']

for name in std_features:
    m = df[name].mean()
    s = df[name].std()
    df[name] = (df[name] - m) / s

df['num.co'] = df['num.co'] / 9
df['ca'] = df['ca'] / 2
df = pd.get_dummies(df, columns=['race'])

# display(df)
df = df.dropna()
df = df.reset_index(drop=True)

display(df)
print(df.shape)
print(df.columns)

#%%
from sklearn.model_selection import train_test_split

Y = df['d.time'].to_numpy().astype('float32')
D = df['death'].to_numpy().astype('int32')
X = df.drop(columns=['d.time','death']).to_numpy().astype('float32')


Xtrain, Xtest, Ytrain, Ytest, Dtrain, Dtest = train_test_split(X,Y,D, test_size=0.3,
                                                            random_state=0)

Xtrain, Xval, Ytrain, Yval, Dtrain, Dval = train_test_split(Xtrain,Ytrain,Dtrain, test_size=0.3,
                                                            random_state=0)

# %%
from pycox.preprocessing.label_transforms import LabTransDiscreteTime

num_time_steps = 128  # set this to 0 to use all unique times of death

if num_time_steps == 0:
    mask = (Dtrain == 1)  # boolean mask specifying which training patients experienced death
    label_transform = LabTransDiscreteTime(np.unique(Ytrain[mask]))
else:
    # use a quantile based discretization, which could possibly end up using fewer than the
    # number of time steps requested (if it turns out that in the dataset, there are many
    # duplicate observed times)
    label_transform = LabTransDiscreteTime(num_time_steps, scheme='quantiles')

Y_train_discrete_np, D_train_discrete_np = label_transform.fit_transform(Ytrain, Dtrain)
Y_val_discrete_np, D_val_discrete_np = label_transform.transform(Yval, Dval)
# note: PyCox's discretization code allows for the possibility that the event indicator changes
# (if a patient's survival time is known so that the event indicator is 1, but the survival time
# is after the maximum discrete time grid time, then the patient's event indicator is changed to
# being censored)

time_grid_train_np = label_transform.cuts
output_num_time_steps = len(time_grid_train_np)
print(f'Number of time steps to be used with DeepHit: {output_num_time_steps}')
print('Time grid:', time_grid_train_np)

# %%
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

X_train = torch.tensor(Xtrain, dtype=torch.float32, device=device)
Y_train = torch.tensor(Y_train_discrete_np, dtype=torch.int64, device=device)
D_train = torch.tensor(D_train_discrete_np, dtype=torch.int32, device=device)
train_data = list(zip(X_train, Y_train, D_train))

X_val = torch.tensor(Xval, dtype=torch.float32, device=device)
Y_val = torch.tensor(Y_val_discrete_np, dtype=torch.int64, device=device)
D_val = torch.tensor(D_val_discrete_np, dtype=torch.int32, device=device)
val_data = list(zip(X_val, Y_val, D_val))

#%%
import torch.nn as nn
from pycox.models import DeepHitSingle

torch.manual_seed(0)

num_input_features = X_train.size(1)
base_neural_net = nn.Sequential(nn.Linear(num_input_features, 8),
                                nn.ReLU(),
                                nn.Linear(8, output_num_time_steps)).to(device) # no softmax needed at the end

# setting alpha=1.0 means that we only use the negative log likelihood loss and not also the ranking loss
# (we do this just as an illustrative example as it corresponds to how DeepHit is first presented in
# Example 2.3.1 of the monograph)

deephit_model = DeepHitSingle(base_neural_net, alpha=1.0, device=device,
                              duration_index=time_grid_train_np)
deephit_loss = deephit_model.loss # we'll see how to use this loss function in the next code cell

#%%
from copy import deepcopy
from torch.utils.data import DataLoader
from pycox.models.data import pair_rank_mat

# minibatch gradient descent
num_epochs = 100
batch_size = 128
learning_rate = 1e-3

train_loader = DataLoader(train_data, batch_size, shuffle=True)  # shuffling for minibatch gradient descent
val_loader = DataLoader(val_data, batch_size, shuffle=False)  # there is no need to shuffle the validation data

optimizer = torch.optim.Adam(base_neural_net.parameters(), lr=learning_rate)
train_epoch_losses = []
val_epoch_losses = []
best_val_loss = float('inf')
best_params = None
best_epoch_index = None
for epoch_index in range(num_epochs):
    base_neural_net.train()
    for X_batch, Y_batch, D_batch in train_loader:
        neural_net_output = base_neural_net(X_batch)

        # note that the loss function requires a ranking matrix to be specified;
        # this is specific to the ranking loss that we are actually not using in this
        # particular demo (but we leave this here since one could modify the demo
        # easily to also be using the ranking loss, in which case this ranking matrix
        # will be needed)
        rank_mat = pair_rank_mat(Y_batch.cpu().numpy(), D_batch.cpu().numpy())
        rank_mat = torch.tensor(rank_mat, dtype=torch.int, device=device)

        loss_batch = deephit_loss(neural_net_output, Y_batch, D_batch, rank_mat)

        optimizer.zero_grad()
        loss_batch.backward()
        optimizer.step()

    # evaluate training and validation set losses
    # (note that in practice, instead of evaluating the negative log likelihood loss,
    # we could instead evaluate other metrics such as time-dependent concordance index,
    # integrated Brier score, etc)
    base_neural_net.eval()
    with torch.no_grad():
        train_loss = torch.tensor(0.0, dtype=torch.float, device=device)
        num_points = 0
        for X_batch, Y_batch, D_batch in train_loader:
            batch_num_points = X_batch.size(0)
            neural_net_output = base_neural_net(X_batch)
            rank_mat = pair_rank_mat(Y_batch.cpu().numpy(), D_batch.cpu().numpy())
            rank_mat = torch.tensor(rank_mat, dtype=torch.int, device=device)
            train_loss += deephit_loss(neural_net_output, Y_batch, D_batch, rank_mat) * batch_num_points
            num_points += batch_num_points
        train_loss = float(train_loss / num_points)
        train_epoch_losses.append(train_loss)
        print(f'Epoch {epoch_index + 1} - train loss {train_loss}', end=' ', flush=True)

        val_loss = torch.tensor(0.0, dtype=torch.float, device=device)
        num_points = 0
        for X_batch, Y_batch, D_batch in val_loader:
            batch_num_points = X_batch.size(0)
            neural_net_output = base_neural_net(X_batch)
            rank_mat = pair_rank_mat(Y_batch.cpu().numpy(), D_batch.cpu().numpy())
            rank_mat = torch.tensor(rank_mat, dtype=torch.int, device=device)
            val_loss += deephit_loss(neural_net_output, Y_batch, D_batch, rank_mat) * batch_num_points
            num_points += batch_num_points
        val_loss = float(val_loss / num_points)
        val_epoch_losses.append(val_loss)
        print(f'- val loss {val_loss}', flush=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch_index = epoch_index
            best_params = deepcopy(base_neural_net.state_dict())
print(f'Best validation loss ({best_val_loss}) achieved at epoch {best_epoch_index + 1}')
base_neural_net.load_state_dict(best_params)
# %%

plt.plot(range(1, num_epochs + 1), train_epoch_losses, label='Training')
plt.plot(range(1, num_epochs + 1), val_epoch_losses, '--', label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# %% Make Prediction np

surv_test_np = deephit_model.predict_surv(Xtest, batch_size=batch_size, to_cpu=True, numpy=True)
surv_test_np.shape

# Predict df
surv_test_df = deephit_model.predict_surv_df(Xtest, batch_size=batch_size)
print(surv_test_df.shape)

# %% here we interpolate by a factor of 10 (so that if we initially had m time steps, we would now have (m-1)*10 + 1 time steps)
interpolation_factor = 10
surv_test_interp_df = deephit_model.interpolate(interpolation_factor).predict_surv_df(Xtest, batch_size=batch_size)
print(surv_test_interp_df.shape)

time_grid_train_interp_np = surv_test_interp_df.index.to_numpy()  # the interpolated time grid is an interpolated version of the *training*  time grid
surv_test_interp_np = surv_test_interp_df.to_numpy().T

# %% Plot some survival curves

for i in range(5):
    plt.plot(time_grid_train_interp_np, surv_test_interp_np[-i])

plt.xlabel('Time')
plt.ylabel('Survival probability')

#%%

c_index = concordance_index(Ytest, surv_test_np, Dtest)
print(f"Concordance Index on Validation Set: {c_index:.4f}")


# %% ------------------ Metrics ---------------------------------
# Make time grid for predictions
time_grid_test_np = np.unique(Ytest)

#%%
from pycox.evaluation import EvalSurv

eval_pycox = EvalSurv(surv_test_interp_df, Ytest, Dtest)
C_td = eval_pycox.concordance_td('antolini')
print(f'Time-dependent concordance index: {C_td}')

# %%

from sksurv.util import Surv
from sksurv.metrics import concordance_index_ipcw
from lifelines.utils import concordance_index

# convert training and test labels into the structured array format used by scikit-survival
labels_train_sksurv = Surv.from_arrays(Dtrain, Ytrain)
labels_test_sksurv = Surv.from_arrays(Dtest, Ytest)

# as an illustrative example, we evaluate the truncated time-dependent concordance index at
# 25, 50, and 75 percentile values of the time grid we specified earlier
eval_time_indices = [int(.25 * len(time_grid_test_np)),
                     int(.5 * len(time_grid_test_np)),
                     int(.75 * len(time_grid_test_np))]

for eval_time_index in eval_time_indices:
    eval_time = time_grid_test_np[eval_time_index]
    #print(len(eval_time))
    print(eval_time)
    # find the interpolated time grid's time point closest to the evaluation time
    interp_time_index = np.argmin(np.abs(eval_time - time_grid_train_interp_np))
    surv_values_at_eval_time_np = surv_test_interp_np[:, interp_time_index]

    estimated_risks_np = 1 - surv_values_at_eval_time_np
    concordance = concordance_index_ipcw(labels_train_sksurv, labels_test_sksurv, estimated_risks_np, tau=eval_time)[0]
    print(f'Eval time {eval_time} - truncated time-dependent concordance: {concordance}')
    
    # Calculate the concordance index
    c_index = concordance_index(Ytest, -estimated_risks_np, Dtest)
    print(f"Concordance Index on Validation Set: {c_index:.4f}")


#%% Nelson Aalen model
from lifelines import KaplanMeierFitter, CoxPHFitter, NelsonAalenFitter


naf = NelsonAalenFitter()
naf.fit(durations=Ytrain, event_observed=Dtrain)
na_hazard_train = naf.cumulative_hazard_at_times(Ytrain).values
na_hazard_train = na_hazard_train * -1

na_hazard_valid = naf.cumulative_hazard_at_times(Yval).values

c_index = concordance_index(Yval, na_hazard_valid, Dval)
print(f"Concordance Index on Validation Set: {c_index:.4f}")

#%%

#Xtrain, Xtest, Ytrain, Ytest, Dtrain, Dtest

print(Dtrain)


# %%


naf = NelsonAalenFitter()
naf.fit(durations=df_train['efs_time'], event_observed=df_train['efs'])
df_target['na_hazard'] = naf.cumulative_hazard_at_times(df_train['efs_time']).values
df_target['na_hazard'] = df_target['na_hazard'] * -1
