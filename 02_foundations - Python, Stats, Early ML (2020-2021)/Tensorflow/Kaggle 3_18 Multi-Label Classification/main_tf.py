#%%
""" 
Use main2 to set up general structure for solving tabular data

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import lightgbm as lgb
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
import category_encoders as ce

import warnings
warnings.filterwarnings('ignore')

from sklearn.compose import ColumnTransformer    
from sklearn.pipeline import Pipeline
from sklearn.metrics import ConfusionMatrixDisplay, f1_score,\
    roc_auc_score, accuracy_score, confusion_matrix, RocCurveDisplay
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from catboost import CatBoostClassifier, Pool
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras import regularizers
import time 
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer
from joblib import dump, load

from itertools import combinations_with_replacement, combinations

from colorama import Style, Fore
blk = Style.BRIGHT + Fore.BLACK
red = Style.BRIGHT + Fore.RED
blu = Style.BRIGHT + Fore.BLUE
gren = Style.BRIGHT + Fore.GREEN
res = Style.RESET_ALL

palette_color = sns.color_palette('dark')
palette1 = ['dimgrey','crimson']
palette2 = ['crimson', 'dimgrey']
palette3 = ['darkgreen', 'orange']
palette4= ['salmon','mediumseagreen']

# display.max_columns
# max_colwidth
# max_rows and min_rows
# max_seq_items  
# pd.set_option("max_rows", None)
#pd.reset_option("max_columns")
    
df = pd.read_csv('train.csv')
dft = pd.read_csv('test.csv')

y = df[['EC1', 'EC2', 'EC3', 'EC4', 'EC5', 'EC6']]
X = df.drop(columns=['id', 'EC1', 'EC2', 'EC3', 'EC4', 'EC5', 'EC6'])

print(X.shape, y.shape)

# List cols of different types
print(X.select_dtypes(include=np.int64).columns)
print(X.select_dtypes(include=np.float64).columns)

num_cols = ['BertzCT', 'Chi1', 'Chi1n', 'Chi1v', 'Chi2n', 'Chi2v', 'Chi3v', 'Chi4n',
       'EState_VSA1', 'EState_VSA2', 'ExactMolWt', 'FpDensityMorgan1',
       'FpDensityMorgan2', 'FpDensityMorgan3', 'HallKierAlpha',
       'HeavyAtomMolWt', 'Kappa3', 'MaxAbsEStateIndex', 'MinEStateIndex',
        'PEOE_VSA10', 'PEOE_VSA14', 'PEOE_VSA6', 'PEOE_VSA7',
       'PEOE_VSA8', 'SMR_VSA10', 'SMR_VSA5', 'SlogP_VSA3', 'VSA_EState9',
       ]

bin_cols = ['NumHeteroatoms','fr_COO', 'fr_COO2']
labels = ['EC1', 'EC2', 'EC3', 'EC4', 'EC5', 'EC6']
labels12 = ['EC1', 'EC2']

#%% X2 remove correlated data

def remove_corr(df):
    # Create correlation matrix
    corr_matrix = df.corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Find index of feature columns with correlation greater than 0.95
    to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]
    df2 = df.drop(columns=to_drop)
    return df2

X2 = remove_corr(X)
bin_cols = X2.select_dtypes(include=np.int64).columns.to_list()
num_cols = X2.select_dtypes(include=np.float64).columns.to_list()


#%% Set up and train model

import tensorflow as tf
from keras import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras import regularizers



input_nodes = X.shape[1]
hidden_nodes = input_nodes/3
EPOCHS = 50



model = Sequential ([
    Dense(hidden_nodes, activation='relu', input_shape=(input_nodes,),
          kernel_regularizer=regularizers.L2(1)),
    Dropout(0.2),
    BatchNormalization(),
    # Dense(hidden_nodes, activation='relu',
    #       kernel_regularizer=regularizers.L2(0.1)),
    # Dropout(0.2),
    # BatchNormalization(),
    Dense(1, activation='sigmoid')
    ])

for _ in range(1):
    model.compile(optimizer='adam', loss=tf.keras.losses.binary_crossentropy, metrics=[tf.keras.metrics.AUC()])

    x_train, x_test, y_train, y_test = train_test_split(X, y['EC2'], train_size=0.7)

    mmc = MinMaxScaler()
    x_train_temp = mmc.fit_transform(x_train[num_cols])
    x_train_norm = pd.DataFrame(data=x_train_temp, columns=num_cols)
    x_train_norm = pd.concat([x_train_norm, x_train[bin_cols].reset_index(drop=True)], axis=1)
    x_test_temp = mmc.transform(x_test[num_cols])
    x_test_norm = pd.DataFrame(data=x_test_temp, columns=num_cols)
    x_test_norm = pd.concat([x_test_norm, x_test[bin_cols].reset_index(drop=True)], axis=1)

    start = time.time()
    weight = {0:4, 1:1}
    history = model.fit(x_train_norm, y_train, class_weight=weight,
                        validation_data=(x_test_norm, y_test),
                        epochs=EPOCHS,  batch_size=64, verbose=0)

    end = time.time() - start
    print(f'Training tim {round(end,2)}')
    print(f"Train loss: {round(history.history['loss'][-1],4)}, Test loss: {round(history.history['val_loss'][-1],4)}")

    
    
#%% Print metrics

labs=['EC2']

y_pred_prob = model.predict(x_test_norm)
df_pred_prob = pd.DataFrame(data=y_pred_prob, columns=labs)
display(df_pred_prob.head())

y_pred = y_pred_prob.copy()
y_pred[y_pred < 0.5] = 0
y_pred[y_pred >= 0.5] = 1
df_pred = pd.DataFrame(data=y_pred, columns=labs)

for label in labs:
    acc = accuracy_score(y_test, df_pred[label])
    roc = roc_auc_score(y_test, df_pred_prob[label])

    print(f'{label} acc:{round(acc, 4)}, roc:{round(roc, 4)}')
    print(confusion_matrix(y_test, df_pred[label]))    


#%% Print prediction probabilities

plt.figure()
plt.hist(df_pred_prob['EC2'])
plt.show()


    
# %% Plot train test training

plt.figure()
plt.plot(range(EPOCHS), history.history['loss'], label='Train')
plt.plot(range(EPOCHS), history.history['val_loss'], label='Test')
plt.legend()
plt.show()



# %% Print predicted display

data = model.predict(X.iloc[104:135,:])
print(data[0,:])


#%% Train model on full dataset

input_nodes = X.shape[1]
hidden_nodes = input_nodes
EPOCHS = 100



model = Sequential ([
    Dense(hidden_nodes, activation='relu', input_shape=(input_nodes,),
          kernel_regularizer=regularizers.L2(0.05)),
    BatchNormalization(),
    Dense(2, activation='sigmoid')
    ])


model.compile(optimizer='adam', loss=tf.keras.losses.binary_crossentropy, metrics=[tf.keras.metrics.AUC()])

start = time.time()
history = model.fit(x_train_norm, y_train,
                    validation_data=(x_test_norm, y_test),
                    epochs=EPOCHS,  batch_size=64, verbose=0)

end = time.time() - start

#%%

fname = "models"
model.save(fname)


#%% load test data

df_test = pd.read_csv('test.csv')
df_test_run = df_test.drop(columns=['id'])
df_solution = pd.DataFrame()
df_solution['id'] = df_test.id

#%% predict

tf_clf = tf.keras.models.load_model('models')
y0_pred = tf_clf.predict(df_test_run)

df_solution['EC1'] = y0_pred[:,0]
df_solution['EC2'] = y0_pred[:,1]

df_solution.to_csv('solution_tf.csv',index=False)




# %%
