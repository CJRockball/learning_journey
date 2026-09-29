#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('train.csv')
dft = pd.read_csv('test.csv')
df.info()
print(df.isnull().sum().sum())
print(df.shape, dft.shape)

# %%
display(df.head())
print(df.columns)
display(dft.head())

# %%
y = df[['EC1', 'EC2', 'EC3', 'EC4', 'EC5', 'EC6']]
X = df.drop(columns=['id', 'EC1', 'EC2', 'EC3', 'EC4', 'EC5', 'EC6'])


# %% Correlation

cor_mat = X.corr().abs()
print(cor_mat.shape)

plt.figure()
sns.heatmap(cor_mat)
plt.show()

#%% SO code
# Create correlation matrix
corr_matrix =X.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]
X2 = X.drop(columns=to_drop)

display(X2.head())
print(X2.shape)

#%% Own try

df_corr = pd.DataFrame(data=cor_mat, columns=X.columns.to_list(), index=X.columns.to_list())
display(df_corr.head())
print(df_corr.shape)

df_list = pd.DataFrame(columns=['var1', 'var2', 'corr_'])

for i in range(31):
    for j in range(i):
        df_list = df_list.append({'var1':df_corr.iloc[:,i].name, 'var2':df_corr.iloc[j,:].name, 'corr_':df_corr.iloc[j,i]},
                       ignore_index=True)

display(df_list.head(7))
print(df_list.shape)

#%%

print(df_corr.iloc[1,:].name)
print(df_corr.iloc[:,2].name)

# %% Mutual information
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif

def make_mi_scores(data, label, discrete_features):
    """Calculate mutual information score"""
    mi_scores = mutual_info_regression(data, label) #, discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=data.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

def make_mi_disc_label(data, label):
    mi_scores = mutual_info_classif(data,label)
    mi_scores = pd.Series(mi_scores, name='MI scores', index=data.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

mi_score_EC1 = make_mi_disc_label(X2, y.EC1)
mi_score_EC2 = make_mi_disc_label(X2, y.EC2)


# %%

display(mi_score_EC1)
display(mi_score_EC2)



# %%

X3 = X2.copy(deep=True)
X3_cols = X3.columns

list_cross_tuples = []
for i in range(len(X2.columns)):
    for j in range(i):
        list_cross_tuples.append((X3_cols[i], X3_cols[j]))

for i in range(len(list_cross_tuples)):
    name1 = list_cross_tuples[i][0]
    name2 = list_cross_tuples[i][1]
    X3[name1 + '_' + name2] = X3[name1] * X3[name2]
    
print(X3.shape)

#%%

mi_score_EC2_all_cross = make_mi_disc_label(X3, y.EC2)
mi_score_EC1_all_cross = make_mi_disc_label(X3, y.EC1)

#%%
display(mi_score_EC2_all_cross.iloc[:10]) 
display(mi_score_EC1_all_cross.iloc[:10]) 
    
# %%
