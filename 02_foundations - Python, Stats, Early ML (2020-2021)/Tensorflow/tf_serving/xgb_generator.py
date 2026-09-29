# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 09:09:21 2021

@author: PatCa
"""


import numpy as np
import random
import pandas as pd
import pickle
import requests
import json
import os
import sys
import pathlib
import joblib
import concurrent.futures
import time

import xgboost as xgb
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, balanced_accuracy_score,f1_score, \
    precision_score, recall_score, roc_auc_score, classification_report,confusion_matrix
from sklearn.model_selection import StratifiedKFold
from mlxtend.plotting import plot_confusion_matrix


CURRENTDIR = pathlib.Path(__file__).resolve().parent
sys.path.append(str(CURRENTDIR))
#Import data processing function 
from cleaning_functions import PCA_Data


def my_timer(orig_func):
    import time
    
    def wrapper(*args, **kwargs):
        t1 = time.time()
        result = orig_func(*args, **kwargs)
        t2 = time.time() - t1
        print('{} ran in : {} sec'.format(orig_func.__name__, t2))
        return result
    
    return wrapper


def prep_data():
    """
    Reads in data, cleans and preps it

    Returns
    -------
    X_train_pipe : Array of Float
    X_test_pipe : Array of Float
    text_train : Array of object
    text_test : Array of object
    y_labels : Array of Int
    y_labels_test : Array of Int

    """
    #Ingest dataset, clean data, prep data. Separate text data and numeric
    X_train_pipe, X_test_pipe, Y_train, Y_test, word_df, word_df_train, \
        word_df_test, _ = PCA_Data()
    
    
    # Make labels one hot for training nn
    y_labels = Y_train.to_numpy() #np.array(pd.get_dummies(Y_train['genre'], dtype=int))
    y_labels_test = Y_test.to_numpy()  #np.array(pd.get_dummies(Y_test['genre'], dtype=int))
    
    return X_train_pipe, X_test_pipe, y_labels, y_labels_test


def make_xgb_model(X_train_pipe, X_test_pipe, y_labels, y_labels_test):

    # Define model
    eval_set = [(X_train_pipe,y_labels), (X_test_pipe, y_labels_test)]
    eval_metric = ['mlogloss']
    clf = XGBClassifier()
    
    # Send values to pipeline
    xgb_baseline = clf.fit(X_train_pipe,y_labels,
                           eval_set=eval_set, eval_metric=eval_metric,
                           early_stopping_rounds=10, verbose=True)

    return

    
def get_slice0(X_train_pipe, y_labels, a, b):

    data = [X_train_pipe, y_labels]
    data_out = []
    for arr in data:
        arr = arr[a:b,:]
        data_out.append(arr)
    
    labels = np.unique(data_out[1])
    
    for i in range(8):
        if i not in labels:
            get_index = y_labels.index(i) 
            data_out[0] = np.concatenate((data_out[0], X_train_pipe[get_index]))
            data_out[1] = np.concatenate((data_out[1], y_labels[get_index]))       
    
    
    return data_out
    

def make_gen_model(X_train_pipe, X_test_pipe, y_labels, y_labels_test):
    """Home made generator. Doesn't work if not all labels are in each fold"""
    CURRENTDIR = pathlib.Path(__file__).resolve().parent
    n=25
    k = int(len(y_labels)/n)
    gen_list = [i*k for i in range(n)]

    X_train0, X_test0, y_labels0, y_labels_test0 = get_slice0(X_train_pipe,
                                    X_test_pipe, y_labels, y_labels_test, gen_list[0], gen_list[1])
    
    
    eval_set = [(X_train0,y_labels0), (X_test0, y_labels_test0)]
    eval_metric = ['mlogloss']
    clf = XGBClassifier(n_estimators=20, n_jobs=-1)
    
    # Send values to pipeline
    xgb_batch = clf.fit(X_train0,y_labels0,
                           eval_set=eval_set, eval_metric=eval_metric,
                           early_stopping_rounds=10, verbose=True)  
    
    fname = CURRENTDIR / "model_artifacts/model_1.model"
    xgb_batch.save_model(fname)
    
    for i in range(n-2):
        X_trainx, X_testx, y_labelsx, y_labels_testx = get_slice0(X_train_pipe,
            X_test_pipe, y_labels, y_labels_test, gen_list[i+1], gen_list[i+2])
        
        eval_set = [(X_trainx,y_labelsx), (X_testx, y_labels_testx)]
        eval_metric = ['mlogloss']
        clf = XGBClassifier(n_estimators=20, n_jobs=-1)
        
        # Send values to pipeline
        xgb_batch = clf.fit(X_trainx,y_labelsx, xgb_model = fname,
                               eval_set=eval_set, eval_metric=eval_metric,
                               early_stopping_rounds=20, verbose=True)
        
        fname = CURRENTDIR / "model_artifacts/model_1.model"
        xgb_batch.save_model(fname)
        print(f"round{i}")     
    
    
    file_name = CURRENTDIR / "model_artifacts/xgb_batch.pkl"
    pickle.dump(xgb_batch, open(file_name,'wb'))

    return 


def make_gen_model1(X_train_pipe, X_test_pipe, y_labels, y_labels_test):
    """Home made generator. Doesn't work if not all labels are in each fold"""
    CURRENTDIR = pathlib.Path(__file__).resolve().parent
    n=25
    k = int(len(y_labels)/n)
    gen_list = [i*k for i in range(n)]

    X_train0, y_labels0 = get_slice0(X_train_pipe,
                                    y_labels, gen_list[0], gen_list[1])
    
    
    xg_train_0 = xgb.DMatrix(X_train0, label=y_labels0)
    xg_test_0 = xgb.DMatrix(X_test_pipe, label=y_labels_test)
    watchlist = [(xg_test_0, 'eval'), (xg_train_0, 'train')]
    
    params = {
    'max_depth': 6,  # the maximum depth of each tree
    'eta': 0.3,  # the training step for each iteration
    'silent': 1,  # logging mode - quiet
    'objective': 'multi:softprob',  # error evaluation for multiclass training
    'num_class': 8,
    'eval_metric': 'mlogloss'}  # the number of classes that exist in this datset
    num_round = 20  # the number of training iterations
    model_0 = xgb.train(params, xg_train_0,1000, watchlist, 
                        early_stopping_rounds=10, 
                        verbose_eval=True)
    
    fname = CURRENTDIR / "model_artifacts/xgbmodel.model"
    model_0.save_model(fname)
    num_trees = model_0.best_ntree_limit
    
    for i in range(n-2):
        X_trainx, y_labelsx = get_slice0(X_train_pipe,
            y_labels, gen_list[i+1], gen_list[i+2])
        
        
        xg_train_x = xgb.DMatrix(X_trainx, label=y_labelsx)
        xg_test_x = xgb.DMatrix(X_test_pipe, label=y_labels_test)
        watchlist = [(xg_test_x, 'eval'), (xg_train_x, 'train')]
        
        params = {
            # 'max_depth': 6,  # the maximum depth of each tree
            # 'eta': 0.3,  # the training step for each iteration
            # 'silent': 1,  # logging mode - quiet
            'objective': 'multi:softprob',  # error evaluation for multiclass training
            'num_class': 8, # the number of classes that exist in this datset
            'eval_metric': 'mlogloss',
            'process_type': 'update',
            'updater'     : 'refresh',
            'refresh_leaf': True}
        
        
        mname = CURRENTDIR / "model_artifacts/xgbmodel.model"
        model_1 = xgb.train(params, xg_train_x,num_trees, watchlist, xgb_model=mname,
                            early_stopping_rounds=10, 
                            verbose_eval=True)
        
        num_trees = model_1.best_ntree_limit
        
        fname = CURRENTDIR / "model_artifacts/xgbmodel.model"
        model_1.save_model(fname)
        print(f"round{i}")
    
    file_name = CURRENTDIR / "model_artifacts/xgb_batch.pkl"
    pickle.dump(model_1, open(file_name,'wb'))

    return 


def check_model_dmat(X_test_pipe, y_labels_test):
    
    CURRENTDIR = pathlib.Path(__file__).resolve().parent
    file_name = CURRENTDIR / "model_artifacts/xgb_batch.pkl"
    imported_xgb_batch = pickle.load(open(file_name,'rb'))
    
    
    y_pred = imported_xgb_batch.predict(xgb.DMatrix(X_test_pipe))
    y_pred_arr = np.argmax(y_pred, axis=1)
    
    cm = confusion_matrix(y_labels_test, y_pred_arr)
    plot_confusion_matrix(cm)
    plt.gcf().set_dpi(300)
    plt.show()

    # Model Evaluation
    def mod_metrics(Y_test, y_pred_test):
        ac_sc = accuracy_score(Y_test, y_pred_test)
        rc_sc = recall_score(Y_test, y_pred_test, average="weighted")
        pr_sc = precision_score(Y_test, y_pred_test, average="weighted")
        f1_sc = f1_score(Y_test, y_pred_test, average='micro')
        #auc_sc = roc_auc_score(Y_test, y_pred_test,multi_class='ovo')
        
        print('Accuracy: {:.2f}, Precision: {:.2f}, Recall: {:.2f}, F1: {:.2f}'.format(ac_sc, rc_sc, pr_sc, f1_sc))
        #print(classification_report(Y_test, y_pred_test))

    mod_metrics(y_labels_test, y_pred_arr)
    

def check_model(X_test_pipe, y_labels_test):
    
    CURRENTDIR = pathlib.Path(__file__).resolve().parent
    file_name = CURRENTDIR / "model_artifacts/xgb_batch.pkl"
    imported_xgb_batch = pickle.load(open(file_name,'rb'))
    
    y_pred = imported_xgb_batch.predict(X_test_pipe)
    
    
    cm = confusion_matrix(y_labels_test, y_pred)
    plot_confusion_matrix(cm)
    plt.gcf().set_dpi(300)
    plt.show()

    # Model Evaluation
    def mod_metrics(Y_test, y_pred_test):
        ac_sc = accuracy_score(Y_test, y_pred_test)
        rc_sc = recall_score(Y_test, y_pred_test, average="weighted")
        pr_sc = precision_score(Y_test, y_pred_test, average="weighted")
        f1_sc = f1_score(Y_test, y_pred_test, average='micro')
        #auc_sc = roc_auc_score(Y_test, y_pred_test,multi_class='ovo')
        
        print('Accuracy: {:.2f}, Precision: {:.2f}, Recall: {:.2f}, F1: {:.2f}'.format(ac_sc, rc_sc, pr_sc, f1_sc))
        #print(classification_report(Y_test, y_pred_test))

    mod_metrics(y_labels_test, y_pred)



if __name__ == "__main__":
    ##Get data
    X_train_pipe, X_test_pipe, y_labels, y_labels_test = prep_data()
    
    ##Train model
    #make_xgb_model(X_train_pipe, X_test_pipe, y_labels, y_labels_test)
    #check_model(X_test_pipe, y_labels_test)
    ## Train model with generator
    make_gen_model1(X_train_pipe, X_test_pipe, y_labels, y_labels_test)
    check_model_dmat(X_test_pipe, y_labels_test)



    
 #%%   
    
    
def kfold_gen_train(X_train_pipe, X_test_pipe, y_labels, y_labels_test):
    
    #Concat data 
    X_data = np.concatenate((X_test_pipe, X_train_pipe), axis=0)
    y_data = np.concatenate((y_labels, y_labels_test), axis=0)
    
    n=100
    kfold = StratifiedKFold(n_splits=n, shuffle=True, random_state=7)
    train_index, test_index = next(iter(kfold.split(X_data, y_data)))
    
    print(len(train_index), len(test_index))
    
    X_train0       = X_data[train_index]
    X_test0        = X_data[test_index]
    y_labels0       = y_data[train_index]
    y_labels_test0 = y_data[test_index]   
    
    print(X_train0.shape)
    
    eval_set = [(X_train0,y_labels0), (X_test0, y_labels_test0)]
    eval_metric = ['mlogloss']
    clf = XGBClassifier(n_estimators=100, n_jobs=-1)
    
    # Send values to pipeline
    xgb_baseline = clf.fit(X_train0,y_labels0,
                           eval_set=eval_set, eval_metric=eval_metric,
                           early_stopping_rounds=10, verbose=True)

    
    
    for i in range(n-1):
        train_index, test_index = next(iter(kfold.split(X_data, y_data)))
        
        X_trainx       = X_data[train_index]
        X_testx        = X_data[test_index]
        y_labelsx       = y_data[train_index]
        y_labels_testx = y_data[test_index]   
        
        eval_set = [(X_trainx,y_labelsx), (X_testx, y_labels_testx)]
        eval_metric = ['mlogloss']
        clf = XGBClassifier(n_estimators=100, n_jobs=-1)
        
        # Send values to pipeline
        xgb_baseline = clf.fit(X_trainx,y_labelsx, xgb_model = xgb_baseline.get_booster(),
                               eval_set=eval_set, eval_metric=eval_metric,
                               early_stopping_rounds=10, verbose=True)
        print(f"round{i}")        

    return    
    
    
    
