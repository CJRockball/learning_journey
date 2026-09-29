#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
import category_encoders as ce

import warnings
warnings.filterwarnings('ignore')

from sklearn.compose import ColumnTransformer    
from sklearn.pipeline import Pipeline
from sklearn.metrics import ConfusionMatrixDisplay, f1_score,\
    roc_auc_score, accuracy_score, confusion_matrix, RocCurveDisplay
from sklearn.model_selection import train_test_split, cross_validate
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from catboost import CatBoostClassifier, Pool
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras import regularizers
import time 
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from joblib import dump, load

import optuna
from utils import pred_metrics

palette_color = sns.color_palette('dark')
palette1 = ['dimgrey','crimson']
palette2 = ['crimson', 'dimgrey']
palette3 = ['darkgreen', 'orange']
palette4= ['salmon','mediumseagreen']

df = pd.read_csv('train.csv')
dft = pd.read_csv('test.csv')

y = df[['EC1', 'EC2', 'EC3', 'EC4', 'EC5', 'EC6']]
X = df.drop(columns=['id', 'EC1', 'EC2', 'EC3', 'EC4', 'EC5', 'EC6'])

print(X.shape, y.shape)
# %%
y0 = y.iloc[:,1]
xtrain, xtest, ytrain, ytest = train_test_split(X,y0, test_size=0.2)

def lgbm_objective(trial):

    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 900),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.1),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 0.001, 100),
        'reg_lambda': trial.suggest_loguniform('reg_lamdba', 0.001, 100),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'num_leaves': trial.suggest_int('num_leaves', 10, 1000),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
    }

    classifier = lgb.LGBMClassifier(**params, metric = 'auc' )
    classifier.fit(xtrain, ytrain)
    y_pred_proba = classifier.predict_proba(xtest)[:, 1]

    # Calculate ROC AUC score for validation predictions
    roc_auc = roc_auc_score(ytest, y_pred_proba)

    return roc_auc

study = optuna.create_study(direction='maximize')
study.optimize(lgbm_objective, n_trials=100)

# Print the best hyperparameters and corresponding ROC AUC score
lgbm_best_params = study.best_params
lgbm_best_score= study.best_value
print("Best Hyperparameters: ",lgbm_best_params)
print("Best ROC AUC Score: ", lgbm_best_score)
# %%
