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

Y = df['d.time'].astype('float32')
D = df['death'].astype('int32')
X = df.drop(columns=['d.time','death']).astype('float32')


Xtrain, Xtest, Ytrain, Ytest, Dtrain, Dtest = train_test_split(X,Y,D, test_size=0.3,
                                                            random_state=0)

Xtrain, Xval, Ytrain, Yval, Dtrain, Dval = train_test_split(Xtrain,Ytrain,Dtrain, test_size=0.3,
                                                            random_state=0)

#%% Check the time steps. Can use all steps from the training data, or make some specific spacing
from pycox.preprocessing.label_transforms import LabTransDiscreteTime

# Get all time steps
mask = Dtrain.values == 1
# All ros ith events 2912
all_timesteps = Ytrain.loc[mask]
assert (mask.sum() == all_timesteps.shape[0]), 'Dims off'
# Get all unique timesteps in sorted array
time_grid_atrain_np = all_timesteps.sort_values().unique()
# Number of timesteps 756
number_all_timesteps = len(time_grid_atrain_np)
print(f'Number of event ros {len(all_timesteps)}, Number of unique timesteps {number_all_timesteps}')

# Get Quantile timesteps. Reduces the number of timesteps
# Move values of both Ytrain and Yvalid
# Choose number of timesteps
num_time_steps = 128
# Set up transformer
label_transform = LabTransDiscreteTime(num_time_steps, scheme='quantiles')
# Fit and transform train
Ytrain_quant, Dtrain_quant = label_transform.fit_transform(Ytrain.values, Dtrain.values)
# Transform val data
Yval_quant, Dval_quant = label_transform.transform(Yval.values, Dval.values)

# Get time steps in ordered array
time_grid_qtrain_np = label_transform.cuts
# Number of timesteps
output_num_time_steps = len(time_grid_qtrain_np)
print(f'Number of quantile timesteps {output_num_time_steps}')

#%% plot timeteps
yy_a = np.ones(number_all_timesteps).cumsum()
yy_q = np.ones(output_num_time_steps).cumsum()

plt.figure()
plt.plot(time_grid_qtrain_np, yy_q, label='Quantile')
plt.plot(time_grid_atrain_np, yy_a, label='All')
plt.legend()
plt.show()



# %%

print(time_grid_atrain_np)


# %%
