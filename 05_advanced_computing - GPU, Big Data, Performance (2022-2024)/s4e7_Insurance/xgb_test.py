#%%
# Import Libraries
import numpy as np 
import pandas as pd 
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import trange
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, train_test_split
from sklearn.base import clone
from sklearn.ensemble import VotingClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score,roc_curve
import gc

#%%

train_data = pd.read_csv('data/raw/train.csv', index_col='id')
#test_data = pd.read_csv('data/raw/test.csv', index_col='id')

#%%



def preprocess(data):
    df = data.copy()
    df["Vehicle_Age"] = df["Vehicle_Age"].astype('category').cat.rename_categories({
        "1-2 Year": 1, "< 1 Year": 0, "> 2 Years": 2}).astype('int8')
    df['Gender'] = (df['Gender']=='Male').astype(np.uint8)
    df['Vehicle_Damage'] = (df['Vehicle_Damage']=='Yes').astype(np.uint8)
    df['Age'] = df['Age'].astype('int8')
    df['Driving_License'] = df['Driving_License'].astype('int8')
    df['Region_Code'] = df['Region_Code'].astype('int8')
    df['Previously_Insured'] = df['Previously_Insured'].astype('int8')
    df['Annual_Premium'] = df['Annual_Premium'].astype('int32')
    value_counts_mapping = df['Annual_Premium'].value_counts().to_dict()
    annual_premium_counts = df['Annual_Premium'].map(value_counts_mapping).astype('int32')
    df['Annual_Premium'] = df['Annual_Premium'].where(annual_premium_counts >= 50, -1).astype('int32')
    df['Annual_Premium_weights'] = annual_premium_counts
    df['Policy_Sales_Channel'] = df['Policy_Sales_Channel'].astype('int16')
    df['Vintage'] = df['Vintage'].astype('int16')
    value_counts_mapping = df['Vintage'].value_counts().to_dict()
    vintage_counts = df['Vintage'].map(value_counts_mapping).astype('int32')
    df['Vintage'] = df['Vintage'].where(vintage_counts >= 50, -1).astype('int32')
    df['Vintage_weights'] = vintage_counts
    df['Annual_Premium_Insurance'] = df['Previously_Insured'].astype(str) + df['Annual_Premium'].astype(str)
    df['Annual_Premium_Insurance'] = pd.factorize(df['Annual_Premium_Insurance'])[0] + 1
    df['Vehicle_Age_Insurance'] = df['Previously_Insured'].astype(str) + df['Vehicle_Age'].astype(str)
    df['Vehicle_Age_Insurance'] = pd.factorize(df['Vehicle_Age_Insurance'])[0] + 1

    return df

# df = pd.concat([train_data, test_data])
# train_size = train_data.shape[0]
# test_size = test_data.shape[0]
# del train_data, test_data

# df = preprocess(df)
# train_data = df[:train_size]
# test_data = df[test_size:].drop(columns = 'Response')
train_data = preprocess(train_data)



# %%

X, X_test = train_test_split(train_data, test_size=100000, random_state=31, stratify=train_data['Response'])
X_train, X_valid = train_test_split(X, test_size=100000, random_state=31, stratify=X['Response'])

y_train = X_train.pop('Response')
y_valid = X_valid.pop('Response')
y_test = X_test.pop('Response')

#%%

display(X.head())

#%%

print(len(X.Annual_Premium.unique()))

#%%

def down_sampling(X, y, i):
    majority_class = X[y == 0].copy()
    #duplicates were generated during feature engineering of annual premium
    majority_class['Response'] = y[y == 0]
    majority_class = majority_class.drop_duplicates()
    y_major = majority_class.pop('Response')
    minority_class = X[y == 1]
    sample_size = len(minority_class)
    majority_sample, X_rest, y_sample, y_rest = train_test_split(majority_class, y_major, 
                                                                  train_size=sample_size, random_state=i, 
                                                stratify=majority_class['Annual_Premium'])
    X_minimal = pd.concat([majority_sample, minority_class], axis=0)
    y_minimal = pd.concat([y_sample, y[y == 1]])


    return X_minimal, y_minimal, X_rest, y_rest

#%%

X_minimal, y_minimal, X_rest, y_rest = down_sampling(X_train, y_train, 63)

#%%

model = XGBClassifier(n_estimators=2000, early_stopping_rounds=100, eval_metric=['auc'], max_bin = 262143,
                   n_jobs=4, random_state=0, colsample_bytree=0.7, max_delta_step = 0.5, gamma = 0.001, 
                      max_depth = 6, device="cuda")

#%%

print(len(X_rest))
print(len(y_minimal))

#%%

test_scores, test_scores2 = [], []


clf = clone(model).fit(X_minimal, y_minimal, eval_set=[(X_valid, y_valid)],verbose=0)
test_scores.append(roc_auc_score(y_test, clf.predict_proba(X_test)[:,1]))

X_t2, _, y_t2, _ = train_test_split(X_train, y_train, train_size=2780918, 
                                  random_state=31)
clf2 = clone(model).fit(X_t2, y_t2, eval_set=[(X_valid, y_valid)],verbose=0)
test_scores2.append(roc_auc_score(y_test, clf2.predict_proba(X_test)[:,1]))
print(f"#ROC-AUC downsampling: {roc_auc_score(y_test, clf.predict_proba(X_test)[:,1])})"
              f"#  ROC-AUC random subsample:   {roc_auc_score(y_test, clf2.predict_proba(X_test)[:,1])}")

#%%

for n in trange(1, 8+1):
    X_t, _, y_t, _ = train_test_split(X_rest, y_rest, train_size=n*1000000, 
                                  random_state=31)
    X_t = pd.concat([X_minimal, X_t], axis=0)
    y_t = pd.concat([y_minimal, y_t], axis=0)
    X_t2, _, y_t2, _ = train_test_split(X_train, y_train, train_size=n*1000000 + 2780918, 
                                  random_state=31)
    clf = clone(model).fit(X_t, y_t, eval_set=[(X_valid, y_valid)],verbose=0)
    clf2 = clone(model).fit(X_t2, y_t2, eval_set=[(X_valid, y_valid)],verbose=0)
    test_scores.append(roc_auc_score(y_test, clf.predict_proba(X_test)[:,1]))
    test_scores2.append(roc_auc_score(y_test, clf2.predict_proba(X_test)[:,1]))
    del clf, clf2
    gc.collect()
    
#%%

n = [0,1,2,3,4,5,6,7,8]
plt.plot(n, test_scores2, marker='o', linestyle='-', color='r', label='random subsample')
plt.plot(n, test_scores, marker='o', linestyle='-', color='b',  label='undersampling')
plt.xlabel('Number of samples (x10^6)')
plt.ylabel('Test Auc Scores')
plt.title('Performance Gain of Adding data rows with Response = 0')
plt.grid(True)
plt.legend()
plt.show()


