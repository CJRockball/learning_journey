
#%%
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import seaborn as sns
import numpy as np

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

target_cols = ['EC1', 'EC2']
target_cols_full = ['EC1', 'EC2', 'EC3', 'EC4', 'EC5', 'EC6']

num_cols = ['BertzCT', 'Chi1', 'Chi1n', 'Chi1v', 'Chi2n', 'Chi2v', 'Chi3v',
       'Chi4n', 'EState_VSA1', 'EState_VSA2', 'ExactMolWt', 'FpDensityMorgan1',
       'FpDensityMorgan2', 'FpDensityMorgan3', 'HallKierAlpha',
       'HeavyAtomMolWt', 'Kappa3', 'MaxAbsEStateIndex', 'MinEStateIndex',
        'PEOE_VSA10', 'PEOE_VSA14', 'PEOE_VSA6', 'PEOE_VSA7',
       'PEOE_VSA8', 'SMR_VSA10', 'SMR_VSA5', 'SlogP_VSA3', 'VSA_EState9']
cat_cols = ['NumHeteroatoms']
bin_cols = ['fr_COO', 'fr_COO2']

y = df_train[target_cols_full]
X = df_train.drop(columns=(['id'] + target_cols_full))

### TODO
# Reduce data
    # Correlation
    # PCA
# Viz; hist, pairplot
# Remove outliers
# Transform distributions

# %% Normalize
X_mmc = X.copy(deep=True)

mmc = MinMaxScaler()
X_mmc[num_cols] = mmc.fit_transform(X[num_cols])

# %% Correlation

def plot_heat_corr(df, title):
    # Create mask for upper tri
    mask = np.zeros_like(df.astype(float).corr())
    mask[np.triu_indices_from(mask)] = True
    
    # Set the colormap and figure state
    colormap = plt.cm.RdBu_r
    plt.figure(figsize=(15,15))
    
    # Set title and front properties
    plt.title(f'{title} Correlation of Features', fontweight='bold', y=1.02, size=20)
    
    # Plot heatmap with the masked diagonal elements
    sns.heatmap(df.astype(float).corr(), linewidths=0.1, vmax=1.0, vmin=-1.0,
                square=True, cmap=colormap, linecolor='white',  mask=mask) #,
#                annot=True, annot_kws={'size':14, 'weight':'bold'})

plot_heat_corr(X,'Train Feature Correlation')
plot_heat_corr(X_mmc,'Train Feature Correlation')

#%% Remove correlated features
# Create correlation matrix
corr_matrix = X.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]
X2 = X.drop(columns=to_drop)

display(X2.head())
print(X.shape, X2.shape)
# Plot corr matrix
plot_heat_corr(X2,'Reduced Train Feature Correlation')

#%% Plot pairsplot ***SLOW*******************************

feature_cols_red = X2.columns.to_list()

def plot_scatter_matrix(df,feature_cols1, target_col, size=26):
    sns.set_style('whitegrid')
    
    sns.pairplot(data=df[feature_cols1+[target_col]], diag_kind='kde', hue=target_col, 
                 plot_kws={'alpha': 0.6}, palette='bright')
    plt.suptitle('Scatterplot Matrix - {target_col}')
    plt.tight_layout()
    plt.show()

data = pd.concat([X2,y['EC1']], axis=1)
plot_scatter_matrix(data,feature_cols_red, 'EC1')

#%% Check data distribution

def plot_histogram(df, features, num_target_cols=6, n_cols=3):
    n_rows = (len(df.columns) - num_target_cols ) // n_cols + 1

    fig,axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(18, 4*n_rows))
    axes = axes.flatten()
    
    for i, name in enumerate(features):
        ax = axes[i]
        sns.histplot(df[name], kde=True, ax=ax)
    ax.set_title(f'{name} Distribution')
    plt.tight_layout()
    plt.show()

features = X2.columns
plot_histogram(X2, features, 0)

# %% Remove outliers by replacing them with median

def outliers(df, feature_cols):
    df2 = df.copy(deep=True)
    limits_dict = {}
    for col in feature_cols:
        Q1 = df2[col].quantile(0.25) # 1st quantile
        Q3 = df2[col].quantile(0.75) # 3rd quantile
        IQR = Q3 - Q1

        LTV_col = Q1 - 1.5*IQR # lower bounds
        UTV_col = Q3 + 1.5*IQR # upper bounds

        col_median = df2[col].median()

        limits_dict[col] = [LTV_col, UTV_col, col_median]


    for col in feature_cols:
        col_values = limits_dict[col]
        
        df2.loc[df2[col] < col_values[0],col] = col_values[2]
        df2.loc[df2[col] > col_values[1],col] = col_values[2]
    
    return df2

feature_cols_red = X2.columns.to_list()
cont_cols = [x for x in feature_cols_red if x not in ['fr_COO', 'NumHeteroatoms']]
X3 = X2.copy(deep=True)
X3[cont_cols] = outliers(X3[cont_cols], cont_cols)

plot_histogram(X3,0)
# %% Box plot to see if outlier removal works

def plot_boxplot(df, label, feat_cols, n_cols=3, title=''):
    sns.set_style('whitegrid')

    cols = df[feat_cols].columns
    n_rows = (len(cols) - 1) // n_cols + 1

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(14, 4*n_rows))

    for i, var_name in enumerate(cols):
        row = i // n_cols
        col = i % n_cols

        ax = axes[row, col]
        sns.boxplot(data=df, x=label, y=var_name, ax=ax, showmeans=True, 
                    meanprops={"marker":"s","markerfacecolor":"white", "markeredgecolor":"blue", "markersize":"5"})
        ax.set_title(f'{var_name} by {label}')
        ax.set_xlabel('')

    fig.suptitle(f'{title} Boxplot by {label}', fontweight='bold', fontsize=16)
    plt.tight_layout()
    plt.show()

data_x2 = pd.concat([X2, y['EC1']], axis=1)
plot_boxplot(data_x2, 'EC1', feature_cols_red, title='Train Raw')
data_x3 = pd.concat([X3, y['EC1']], axis=1)
plot_boxplot(data_x3, 'EC1', feature_cols_red, title='Train Reduced')


# %% Try quantile transformation
from sklearn.preprocessing import QuantileTransformer

def quant_transform(df, features):
    
    X_tmp = df.copy(deep=True)
    
    for name in features:
        qt = QuantileTransformer(n_quantiles=1000, random_state=0)
        X_tmp[name] = qt.fit_transform(X_tmp[name].to_numpy().reshape(-1, 1))

    return X_tmp

X4 = quant_transform(X3, cont_cols)

features = X4.columns
plot_histogram(X4, features, 0)


# %% Quantile transform data

X_test = X3.copy(deep=True)

for name in ['BertzCT']:
    qt = QuantileTransformer(n_quantiles=1000, random_state=0)
    X_test[name] = qt.fit_transform(X_test[name].to_numpy().reshape(-1, 1))

#%%

from sklearn.feature_selection import mutual_info_regression, mutual_info_classif

def plot_label_corr(df, feature_cols, label):
    df_ = df[feature_cols+[label]]
    corr = df_.corr()
    label_corr = corr[label].sort_values().to_frame()
    single_col_plot(label_corr, f'Correlation with {label}')    
    
def single_col_plot(df, title):
    # Create a heatmap of the correlations with EC1
    plt.figure(dpi=190)
    sns.set(font_scale=0.8)
    sns.set_style("white")
    sns.set_palette("PuBuGn_d")
    sns.heatmap(df, cmap="coolwarm", annot=True, fmt='.2f')
    plt.title(title)
    plt.show()

def plot_label_mi(df, feature_cols, label, discrete_features=[]):
    y_data = df[label]
    data = df[feature_cols]
    """Calculate mutual information score"""
    mi_scores = mutual_info_classif(data, y_data)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=data.columns)
    mi_scores = mi_scores.sort_values(ascending=False).to_frame()
    single_col_plot(mi_scores, f'MI score with {label}')
    
features = X4.columns.to_list()  
X4_graph = pd.concat([X4, y[['EC1', 'EC2', 'EC3', 'EC4', 'EC5', 'EC6']]], axis=1) 
plot_label_corr(X4_graph, features+['EC2', 'EC3', 'EC4', 'EC5', 'EC6'], 'EC1')
plot_label_corr(X4_graph, features+['EC1', 'EC3', 'EC4', 'EC5', 'EC6'], 'EC2')

plot_label_mi(X4_graph, features+['EC2', 'EC3', 'EC4', 'EC5', 'EC6'], 'EC1')
plot_label_mi(X4_graph, features+['EC1', 'EC3', 'EC4', 'EC5', 'EC6'], 'EC2')


# %%
