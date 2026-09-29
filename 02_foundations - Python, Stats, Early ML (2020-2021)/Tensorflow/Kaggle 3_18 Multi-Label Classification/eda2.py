#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import os
from copy import deepcopy
from functools import partial
from itertools import combinations
import random
import gc
import time
import math

# Import sklearn classes for model selection, cross validation, and performance evaluation
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import seaborn as sns
from category_encoders import OrdinalEncoder, CountEncoder, CatBoostEncoder, OneHotEncoder
from sklearn.preprocessing import FunctionTransformer, LabelEncoder # OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.under_sampling import RandomUnderSampler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.decomposition import PCA, NMF
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.compose import make_column_selector

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from colorama import Style, Fore
blk = Style.BRIGHT + Fore.BLACK
red = Style.BRIGHT + Fore.RED
blu = Style.BRIGHT + Fore.BLUE
res = Style.RESET_ALL

df_train = pd.read_csv('train.csv')
df_train = df_train.drop(columns=['id'])
df_test = pd.read_csv('test.csv')
df_test = df_test.drop(columns=['id'])

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
feature_cols = num_cols + cat_cols + bin_cols

print(f'train data size: {df_train.shape}, test data size: {df_test.shape}')
# %%

def set_frame_style(df, caption=""):
    """Helper function to set dataframe presentation style.
    """
    return df.style.background_gradient(cmap='Blues').set_caption(caption).set_table_styles([{
    'selector': 'caption',
    'props': [
        ('color', 'Blue'),
        ('font-size', '18px'),
        ('font-weight','bold')
    ]}])

def check_data(data, title):
    cols = data.columns.to_list()
    display(set_frame_style(data[cols].head(),f'{title}: First 5 Rows Of Data'))
    display(set_frame_style(data[cols].describe(),f'{title}: Summary Statistics'))
    display(set_frame_style(data[cols].nunique().to_frame().rename({0:'Unique Value Count'}, axis=1).transpose(), f'{title}: Unique Value Counts In Each Column'))
    display(set_frame_style(data[cols].isna().sum().to_frame().transpose(), f'{title}:Columns With Nan'))
    
check_data(df_train, 'Train data')
print('-'*100)
check_data(df_test, 'Test data')
print('-'*100)

# %%    EDA
# Contents:
#     Train, Test and Original data histograms
#     Correlation of Features
#     Scatter plots of features by Machine failure (random undersampling)
#     Hierarchical Clustering
#     Pie and bar charts for categorical column features
#     Distribution Plot by Type
#     Boxplot by Machine failure
#     Violinplot by Machine failure
#     Scatter plots after dimensionality reduction with PCA by Machine failure

#%% Train has 6 target cols, test has 0 target cols *** SLOW ***

def plot_histogram(df, num_target_cols=6, n_cols=3):
    n_rows = (len(df.columns) - num_target_cols ) // n_cols + 1

    fig,axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(18, 4*n_rows))
    axes = axes.flatten()
    
    for i, name in enumerate(num_cols+cat_cols+bin_cols):
        ax = axes[i]
        sns.histplot(df_train[name], kde=True, ax=ax)
    ax.set_title(f'{name} Distribution')
    plt.tight_layout()
    plt.show()

plot_histogram(df_train)
plot_histogram(df_test,0)
# %% Plot Correlation 

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
    sns.heatmap(df.astype(float).corr(), linewidths=0.1, vmax=1., vmin=-1.,
                square=True, cmap=colormap, linecolor='white',  mask=mask)
#                annot=True, annot_kws={'size':14, 'weight':'bold'})

plot_heat_corr(df_train,'Train Feature Correlation')
plot_heat_corr(df_test, 'Test Feature Correlation')
plot_heat_corr(df_train[target_cols_full], 'Label Correlation')

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
    
    
    
plot_label_corr(df_train, feature_cols+['EC1', 'EC2', 'EC3', 'EC4', 'EC5'], 'EC6')
#plot_label_corr(df_train, feature_cols, 'EC2')

plot_label_mi(df_train, feature_cols+['EC1', 'EC2', 'EC3', 'EC4', 'EC5'], 'EC6')
#plot_label_mi(df_train, feature_cols+['EC1'], 'EC2')
 
# %% 

def plot_scatter_matrix(df,feature_cols1, target_col, size=26):
    sns.set_style('whitegrid')
    cols = df[num_cols]

    sns.pairplot(data=df[feature_cols1+[target_col]], diag_kind='kde', hue=target_col, 
                 plot_kws={'alpha': 0.6}, palette='bright')
    plt.suptitle('Scatterplot Matrix - EC1')
    plt.tight_layout()
    plt.show()
    
plot_scatter_matrix(df_train,feature_cols, 'EC1')

# %%
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

def h_cluster(data, title):
    fig, ax = plt.subplots(1,1, figsize=(14,8), dpi=120)
    correlations = data.corr()
    converted_corr = 1 - np.abs(correlations)
    Z = linkage(squareform(converted_corr), 'complete')

    dn = dendrogram(Z, labels=data.columns, ax=ax, 
                    above_threshold_color='#ff0000', orientation='right')
    hierarchy.set_link_color_palette(None)
    plt.grid()
    plt.title(f'{title} Hierarchical clustering, Dendrogram',
                fontsize=18, fontweight='bold')
    plt.show()
    
    
h_cluster(df_train[feature_cols], 'Train data')
h_cluster(df_test[feature_cols], 'Test data')

# %% Visualize categorical cats

def plot_target_feature(df_train, target_col, figsize=(16,5), palette='colorblind', name='Train'):

    fig, ax = plt.subplots(1, 2, figsize=figsize)
    ax = ax.flatten()

    # Pie chart
    pie_colors = sns.color_palette(palette, len(df_train[target_col].unique()))
    ax[0].pie(
        df_train[target_col].value_counts(),
        shadow=True,
        explode=[0.05] * len(df_train[target_col].unique()),
        autopct='%1.f%%',
        textprops={'size': 15, 'color': 'white'},
        colors=pie_colors
    )
    ax[0].set_aspect('equal')  # Fix the aspect ratio to make the pie chart circular

    # Bar plot
    bar_colors = sns.color_palette(palette)
    sns.countplot(
        data=df_train,
        y=target_col,
        ax=ax[1],
        palette=bar_colors
    )
    ax[1].set_xlabel('Count', fontsize=14)
    ax[1].set_ylabel('')
    ax[1].tick_params(labelsize=12)
    ax[1].yaxis.set_tick_params(width=0)  # Remove tick lines for y-axis

    fig.suptitle(f'{target_col} in {name} Dataset', fontsize=16, fontweight='bold')
    plt.tight_layout()

    # Show the plot
    plt.show()

   
plot_target_feature(df_train, 'fr_COO', figsize=(16,5), palette='colorblind', name='Train data')    
plot_target_feature(df_train, 'fr_COO2', figsize=(16,5), palette='colorblind', name='Train data')    
plot_target_feature(df_train, 'NumHeteroatoms', figsize=(16,5), palette='colorblind', name='Train data')    

plot_target_feature(df_train, 'EC1', figsize=(16,5), palette='colorblind', name='Train data')    
plot_target_feature(df_train, 'EC2', figsize=(16,5), palette='colorblind', name='Train data')    


# %% Check distribution by label

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
    
plot_boxplot(df_train, 'EC1', feature_cols, title='Train Data')
plot_boxplot(df_train, 'EC2', feature_cols, title='Train Data')

# %%

class Decomp:
    def __init__(self, n_components, method="pca", scaler_method='standard'):
        self.n_components = n_components
        self.method = method
        self.scaler_method = scaler_method
        
    def dimension_reduction(self, df):
        X_reduced = self.dimension_method(df)
        df_comp = pd.DataFrame(X_reduced, columns=
                [f'{self.method.upper()}_{i}' for i in range(self.n_components)],
                index=df.index)
        return df_comp
    
    def dimension_method(self, df):
        X = self.scaler(df)
        if self.method == 'pca':
            pca = PCA(n_components=self.n_components, random_state=0)
            X_reduced = pca.fit_transform(X)
            self.comp = pca
        elif self.method == 'nmf':
            nmf = NMF(n_components=self.n_components, random_state=0)
            X_reduced = nmf.fit_transform(X)
        else: 
            raise ValueError(f"Invalid method name: {self.method}")

        return X_reduced
    
    def scaler(self, df):
        _df = df.copy()
        
        if self.scaler_method == 'standard':
            return StandardScaler().fit_transform(_df)
        elif self.scaler_method == 'minmax':
            return MinMaxScaler().fit_transform(_df)
        elif self.scaler_method == None:
            return _df.values
        else:
            raise ValueError(f"Invalid svaler_method name")

    def get_columns(self):
        return [f'{self.method.upper()}_{i}' for i in range(self.n_components)]

    def get_explained_variance_ratio(self):
        return np.sum(self.comp.explained_variance_ratio_)

    def transform(self,df):
        X = self.scaler(df)
        X_reduced = self.comp.transform(X)
        df_comp = pd.DataFrame(X_reduced, columns=[
            f'{self.method.upper()}_{i}' for i in range(self.n_components)],
                               index=df.inedx)
        
        return df_comp
    
    def decomp_plot(self, tmp, label, hue='genre'):
        plt.figure(figsize=(16,9))
        sns.scatterplot(x=f"{label}_0", y=f"{label}_1", data=tmp, hue=hue,
                        alpha=0.7, s=100, palette='muted')
        plt.title(f'{label} on {hue}', fontsize=20)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=10);
        plt.xlabel(f'{label} Component 1', fontsize=15)
        plt.ylabel(f'{label} Component 2', fontsize=15)

data = df_train[num_cols].copy()
y_train = df_train[target_cols].copy()
for method in ['pca', 'nmf']:
    decomp = Decomp(n_components=2, method=method, scaler_method='minmax')
    decomp_feature = decomp.dimension_reduction(data)
    decomp_feature = pd.concat([y_train, decomp_feature], axis=1)
    decomp.decomp_plot(decomp_feature, method.upper(), 'EC1')

del y_train, data

# %%
