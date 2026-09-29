# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 08:48:34 2021

@author: PatCa
"""

import numpy as np
import pandas as pd
import pathlib
import joblib
from pickle import dump
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.compose import ColumnTransformer

from sklearn.decomposition import PCA


def PCA_Data():
    
    CURRENTDIR = pathlib.Path(__file__).resolve().parent
    fname_feature = CURRENTDIR / 'data/features.csv'
    fname_label = CURRENTDIR / 'data/labels.csv'
    #Import raw data
    source_feature = pd.read_csv(fname_feature)
    source_label = pd.read_csv(fname_label)
    
    #Combine features and labels and copy
    source_data = pd.merge(source_feature, source_label, on="trackID")
    clean_data = source_data.copy()
    
    #Remove na and duplicates
    clean_data = clean_data.dropna()
    clean_data = clean_data.drop_duplicates()
    
    #Check type
    clean_data = clean_data.astype({'time_signature':int,'key':int,'mode':int})
    
    #Rename categorical values
    mode_dict = {0:'minor', 1:'major'}
    key_dict = {0:'C', 1:'D', 2:'E',3:'F', 4:'G', 5:'H', 6:'I', 7:'J', 8:'K', 9:'L',
                10:'M', 11:'N'}
    label_dict = {'soul and reggae':0, 'pop':1, 'punk':2, 'jazz and blues':3, 
                  'dance and electronica':4,'folk':5, 'classic pop and rock':6, 'metal':7}
    
    clean_data['mode'] = clean_data['mode'].replace(mode_dict)
    clean_data['key'] = clean_data['key'].replace(key_dict)
    clean_data['genre'] = clean_data['genre'].replace(label_dict)
    
    #Remove small categories
    clean_data = clean_data[clean_data.time_signature != 0]
    
    #Separate out text feature "tags" and remove from clean_data dataframe
    word_df = pd.DataFrame(data=clean_data[['tags','genre']].to_numpy(), columns=['tags','genre'])
    clean_data = clean_data.drop(columns=['title','trackID'])
    
    #%%Split data for training and testing
    
    train_data = clean_data
    y = train_data[['genre']] #Make Dataframe
    training_data = train_data.loc[:,train_data.columns != 'genre']
    
    (X_train, X_test, Y_train, Y_test) = train_test_split(training_data, y,
                                                          test_size=0.2,
                                                          random_state=42,stratify=y)
    
    #Separate out text data
    word_df_train = pd.concat((X_train['tags'],Y_train), axis=1) 
    word_df_test = pd.concat((X_test['tags'],Y_test), axis=1)
    X_train = X_train.drop(columns='tags')
    X_test = X_test.drop(columns='tags')
    
    #%%Check feature correlation
    
    nc_cols = ['loudness','tempo','time_signature','key','mode','duration']
    cat_feat = ['time_signature','key','mode']
    
    cont_data = X_train.drop(columns=nc_cols)
    
    #%% PCA on cont_data2
    
    pca_scaler = StandardScaler()
    pca_scaler.fit(cont_data)
    
    fname_scaler = CURRENTDIR / 'model_artifacts/pca_scaler.pkl'
    dump(pca_scaler, open(fname_scaler, 'wb'))
    cont_data_norm = pca_scaler.transform(cont_data)
    
    fname_pca = CURRENTDIR / 'model_artifacts/pca.pkl'
    pca = PCA(0.95).fit(cont_data_norm)
    dump(pca, open(fname_pca, 'wb'))
    num_pca_cols = pca.n_components_
    
    data_pca_array = pca.transform(cont_data_norm)
    cont_data_pca = pd.DataFrame(data=data_pca_array)
    col_names = ['PCA_'+str(i) for i in range(num_pca_cols)]
    cont_data_pca.columns = col_names
    
    X_train2 = X_train[nc_cols]
    X_train3 = pd.concat([X_train2.reset_index(drop=True), cont_data_pca.reset_index(drop=True)], axis=1)
    
    #%% Transform test data
    
    cont_test_data = X_test.drop(columns=nc_cols)
    cont_test_data_norm = pca_scaler.transform(cont_test_data)
    #cont_test_data_norm = (cont_test_data-cont_test_data.mean())/(cont_test_data.std())
    
    test_data_pca_array = pca.transform(cont_test_data_norm)
    cont_test_data_pca = pd.DataFrame(data=test_data_pca_array)
    test_col_names = ['PCA_'+str(i) for i in range(num_pca_cols)]
    cont_test_data_pca.columns = test_col_names
    
    X_test2 = X_test[nc_cols]
    X_test3 = pd.concat((X_test2.reset_index(drop=True), cont_test_data_pca.reset_index(drop=True)), axis=1)
    
    #%% Make pipeline for all data except text data
    
    cat_cols = ['time_signature','key','mode']
    num_cols = ['loudness','tempo','duration']
    pca_cols = col_names
    label_cols = ['genre']
    #Combined
    norm_cols = num_cols + pca_cols
    feature_ar = np.array(num_cols + cat_cols + pca_cols)
    
    # Apply preprocess
    preprocessor = ColumnTransformer(
        transformers=[('nums', RobustScaler(), num_cols), 
                      ('cats', OneHotEncoder(sparse=False), cat_cols),
                      ('pca', 'passthrough',pca_cols)])
    
    # ('pass','passthrough', num_cols)
    
    #Transform data
    pipe = Pipeline(steps=[('preprocessor',preprocessor)])
    X_train_pipe = pipe.fit_transform(X_train3)
    X_test_pipe = pipe.transform(X_test3)
    #print(pipe.steps)
    
    #Make list of feature names after one-hot
    feature_list = []
    for name, estimator, features in preprocessor.transformers_:
        if hasattr(estimator,'get_feature_names'):
            if isinstance(estimator,OneHotEncoder):
                f = estimator.get_feature_names(features)
                feature_list.extend(f)
        else:
            feature_list.extend(features)
    
    fname_pipe = CURRENTDIR / 'model_artifacts/pipe.joblib'
    joblib.dump(pipe, fname_pipe)
    
    return X_train_pipe, X_test_pipe, Y_train, Y_test, word_df, word_df_train, word_df_test, num_pca_cols
