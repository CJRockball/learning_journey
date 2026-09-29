# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 14:59:07 2021

@author: PatCa
"""

import numpy as np
import random
import pandas as pd
from pickle import load
import requests
import json
import os
import sys
import pathlib
import joblib
import concurrent.futures
import time

import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, balanced_accuracy_score,f1_score, \
    precision_score, recall_score, roc_auc_score, classification_report,confusion_matrix
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



def preprocess_data(x):
    """
    Parameters
    ----------
    x : dataframe column
        takes a dataframe column with space separated words.

    Returns
    -------
    text : np array (1,)
           Format that tensorflow hub embedding layer reads.

    """
    #get dataframe column to list
    text = x.to_list()
    #Get right text structure
    text = [str(t).encode('ascii', 'replace') for t in text]
    #Make text np.array for feeding into NN
    text = np.array(text, dtype=object)[:]   
    return text

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
    
    #Clean text data
    #Remova comma to make a vector of word
    word_df['tags'] = word_df['tags'].str.replace(',', '')
    word_df_train['tags'] = word_df_train['tags'].str.replace(',','')
    word_df_test['tags'] = word_df_test['tags'].str.replace(',','')
    
    text_train = preprocess_data(word_df_train['tags']) 
    text_test = preprocess_data(word_df_test['tags']) 
    
    # Make labels one hot for training nn
    y_labels = np.array(pd.get_dummies(Y_train['genre'], dtype=int))
    y_labels_test = np.array(pd.get_dummies(Y_test['genre'], dtype=int))
    
    return X_train_pipe, X_test_pipe, text_train, text_test, y_labels, y_labels_test


def dnn_design(model, specific:str): 
    """
    Plotting description of neural network

    Parameters
    ----------
    model : NN model
    specific : str
        Model name

    Returns
    -------
    None.

    """
    keras.utils.plot_model(model, to_file=specific + "_model_song_predict.png", show_shapes=True)
    print(model.summary())
    return


# Set up tensorflow model
def make_tf_models(X_train_pipe, X_test_pipe, text_train, text_test, y_labels, y_labels_test):
    """
    Function to make NN model and to train and save. Saves the NN model in
    specified folder

    Parameters
    ----------
    X_train_pipe : Array of Float
    X_test_pipe : Array of Float
    text_train : Array of object
    text_test : Array of object
    y_labels : Array of Int
    y_labels_test : Array of Int

    Returns
    -------
    None.

    """    
    CURRENTDIR = pathlib.Path(__file__).resolve().parent
    #Input shape of numeric data
    n_cols = X_train_pipe.shape[1]
    #Download lyer Use a pretrained layer from Tensorflow hub to do text embedding
    hub_layer = hub.KerasLayer("https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1", output_shape=[128], 
                            input_shape=[], dtype=tf.string, name='hub', trainable=False)
    #Input layer for embedding layer
    text_input = keras.Input(shape=(), name="emb_text_input", dtype=tf.string)
    #Embeddinglayer
    emb_text = hub_layer(text_input)
    #Input 2 numeric matrix
    multi_input = keras.Input(shape=(n_cols,), name="multi_data")
    #Concatenation layer
    common_input =tf.keras.layers.concatenate([emb_text, multi_input])

    initializer = tf.keras.initializers.GlorotNormal(seed=42)

    #Make wide model-----------------------------------------------------------
    w_x = tf.keras.layers.Dense(256, activation='relu', kernel_initializer=initializer)(common_input)
    output = tf.keras.layers.Dense(8, activation='softmax')(w_x)
    
    wide_model = keras.Model(inputs=[text_input, multi_input],
                           outputs=output)
    
    wide_model.compile(optimizer='adam',
                    loss=['categorical_crossentropy'],
                    metrics=['accuracy'])
    
    #Print network properties
    #dnn_design(wide_model, 'wide')
 
    #Train model and save best to folder
    WORKING_DIR = os.getcwd() 
    history = wide_model.fit({"multi_data":X_train_pipe, "emb_text_input":text_train}, 
                   y_labels, 
                   validation_data=({"multi_data":X_test_pipe, "emb_text_input":text_test}, 
                                    y_labels_test),
                   batch_size=64, epochs=100,
                callbacks=[tf.keras.callbacks.EarlyStopping(patience=5),
                           tf.keras.callbacks.ModelCheckpoint(os.path.join(CURRENTDIR,
                                                                 'wide_folder/1'),
                                                    monitor='val_loss', verbose=1,
                                                    save_best_only=True,
                                                    save_weights_only=False,
                                                    mode='auto')])
    #Save history to dataframe for plotting
    #frame = pd.DataFrame(history.history)        
    return 

# Set up tensorflow model
def make_tf_ensemble(X_train_pipe, X_test_pipe, text_train, text_test, y_labels, y_labels_test, model_id):
    """
    Function to make NN model and to train and save. Saves the NN model in
    specified folder

    Parameters
    ----------
    X_train_pipe : Array of Float
    X_test_pipe : Array of Float
    text_train : Array of object
    text_test : Array of object
    y_labels : Array of Int
    y_labels_test : Array of Int

    Returns
    -------
    None.

    """    
    CURRENTDIR = pathlib.Path(__file__).resolve().parent
    #Input shape of numeric data
    n_cols = X_train_pipe.shape[1]
    # Release global state
    tf.keras.backend.clear_session()
    #Download lyer Use a pretrained layer from Tensorflow hub to do text embedding
    hub_layer = hub.KerasLayer("https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1", output_shape=[128], 
                            input_shape=[], dtype=tf.string, name='hub', trainable=False)
    #Input layer for embedding layer
    text_input = keras.Input(shape=(), name="emb_text_input", dtype=tf.string)
    #Embeddinglayer
    emb_text = hub_layer(text_input)
    #Input 2 numeric matrix
    multi_input = keras.Input(shape=(n_cols,), name="multi_data")
    #Concatenation layer
    common_input =tf.keras.layers.concatenate([emb_text, multi_input])

    seed_no = np.random.randint(0,200)
    
    initializer = tf.keras.initializers.GlorotNormal(seed=seed_no)

    #Make wide model-----------------------------------------------------------
    w_x = tf.keras.layers.Dense(256, activation='relu', kernel_initializer=initializer)(common_input)
    output = tf.keras.layers.Dense(8, activation='softmax')(w_x)
    
    wide_model = keras.Model(inputs=[text_input, multi_input],
                           outputs=output)
    
    wide_model.compile(optimizer='adam',
                    loss=['categorical_crossentropy'],
                    metrics=['accuracy'])
    
    #Print network properties
    #dnn_design(wide_model, 'wide')
 
    #Train model and save best to folder
    WORKING_DIR = os.getcwd() 
    history = wide_model.fit({"multi_data":X_train_pipe, "emb_text_input":text_train}, 
                   y_labels, 
                   validation_data=({"multi_data":X_test_pipe, "emb_text_input":text_test}, 
                                    y_labels_test),
                   batch_size=64, epochs=100,
                callbacks=[tf.keras.callbacks.EarlyStopping(patience=5),
                           tf.keras.callbacks.ModelCheckpoint(os.path.join(CURRENTDIR,
                                                                 'wide_folder/' + str(model_id)),
                                                    monitor='val_loss', verbose=1,
                                                    save_best_only=True,
                                                    save_weights_only=False,
                                                    mode='auto')])
    #Save history to dataframe for plotting
    #frame = pd.DataFrame(history.history)        
    return 
 

def get_test_data(sample_row:int):
    """
    Get sample from test file to run on model

    Parameters
    ----------
    sample_row : Int
        Row from testfile to run on model

    Returns
    -------
    test_pipe : Array of Float
        Numeric input data for model
    text_test : Array of object
        Text input data for model

    """
    CURRENTDIR = pathlib.Path(__file__).resolve().parent
    #Import pipeline and prediction model
    fname_pipe = CURRENTDIR / 'model_artifacts/pipe.joblib' 
    pipe = joblib.load(fname_pipe)
    
    #get dataset
    fname_data = CURRENTDIR / 'data/test.csv'
    test_data = pd.read_csv(fname_data)
    test_data = test_data.iloc[sample_row,:].to_frame().T
    
    #preprocess data
    test_data = test_data.astype({'time_signature':int,'key':int,'mode':int})
    #Rename categorical values
    mode_dict = {0:'minor', 1:'major'}
    key_dict = {0:'C', 1:'D', 2:'E',3:'F', 4:'G', 5:'H', 6:'I', 7:'J', 8:'K', 9:'L',
                10:'M', 11:'N'}
    test_data['mode'] = test_data['mode'].replace(mode_dict)
    test_data['key'] = test_data['key'].replace(key_dict)

    #Save text data
    word_df = pd.DataFrame(data=test_data[['tags']].to_numpy(), columns=['tags'])
    test_data = test_data.copy().drop(columns=['title', 'tags','trackID'])

    #Clean data for PCA
    nc_cols = ['loudness','tempo','time_signature','key','mode','duration']
    
    #Get columns for PCA transformation
    pca_test_data = test_data.drop(columns=nc_cols)
    
    #normalize data before pca transformation
    fname_scaler = CURRENTDIR / 'model_artifacts/pca_scaler.pkl'
    pca_scaler = load(open(fname_scaler, 'rb'))
    pca_test_data_norm = pca_scaler.transform(pca_test_data)
    
    #Get pca transformer and run on data
    fname_pca = CURRENTDIR / 'model_artifacts/pca.pkl'
    pca = load(open(fname_pca, 'rb')) 
    pca_test_data_array = pca.transform(pca_test_data_norm)
    
    #Move transformed data to datafroma and name columns "PCA" + number
    cont_test_data_pca = pd.DataFrame(data=pca_test_data_array)
    test_col_names = ['PCA_'+str(i) for i in range(cont_test_data_pca.shape[1])]
    cont_test_data_pca.columns = test_col_names
    
    #Concatenate nc_columns to pca columns
    test_data2 = test_data[nc_cols]
    test_data3 = pd.concat((test_data2.reset_index(drop=True), cont_test_data_pca.reset_index(drop=True)), axis=1)

    # Run test data in pipeline
    test_pipe = pipe.transform(test_data3)
    
    #Preprocess text data
    word_df['tags'] = word_df['tags'].str.replace(',','')   
    text_test = preprocess_data(word_df['tags'])     

    return test_pipe, text_test


@my_timer
def load_models():
    
    CURRENTDIR = pathlib.Path(__file__).resolve().parent
    #models = {}
    models_list = []
    num_list = range(9)
    
    def load_f(i):
#    for i in range(9):
        path = os.path.join(CURRENTDIR,'wide_folder/' + str(i))
        reconstructed_model = keras.models.load_model(path)
        #mname = 'm' + str(i)
        #models[mname] = reconstructed_model
        models_list.append(reconstructed_model)
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(load_f, num_list)             
        
    return models_list


def test_model_serving_orig(test_row:int, models_list):
    """
    Function to use saved tensorflow model to do prediction

    Parameters
    ----------
    test_row : Int
        row number from testfile to get input data

    Returns
    -------
    Prints out prediction

    """
    
    test_pipe, text_test = get_test_data(test_row)
    #Get model from folder, load model
    #CURRENTDIR = pathlib.Path(__file__).resolve().parent
    pred_ens = []
    pred_ens_prob = []
    #num_list = range(9)
    
    #def pred_fcn(i): 
    for modelx in models_list:
        #path = os.path.join(CURRENTDIR,'wide_folder/' + str(i))
        #reconstructed_model = keras.models.load_model(path)
        
        #Predict with loaded data
        pred_model = modelx.predict({"multi_data":test_pipe, "emb_text_input": text_test})
        
        #Change prediction from category probability to text
        pred_genre = np.argmax(pred_model, axis=1)  
        pred_ens.append(int(pred_genre))
        pred_prob = np.max(pred_model, axis=1)
        pred_ens_prob.append(pred_prob)
        
        # rev_label_dict = {0:'soul and reggae', 1:'pop', 2:'punk', 3:'jazz and blues', 
        #                   4:'dance and electronica', 5:'folk', 6:'classic pop and rock', 7:'metal'}
        # suggested_genre = rev_label_dict[pred_genre[0]]
        
        #Print prediction
        #print("suggested genre is:", suggested_genre, " with probability:", round(pred_prob[0],2))
    
    #with concurrent.futures.ThreadPoolExecutor() as executor:
    #    executor.map(pred_fcn, num_list)
    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     executor.map(pred_fcn, num_list)
    y = max(set(pred_ens), key=pred_ens.count)
    print(y)
    return   

def test_model_serving(test_row:int, models_list):
    """
    Function to use saved tensorflow model to do prediction

    Parameters
    ----------
    test_row : Int
        row number from testfile to get input data

    Returns
    -------
    Prints out prediction

    """
    
    test_pipe, text_test = get_test_data(test_row)
    pred_ens = []
    pred_ens_prob = []
    
    for modelx in models_list:
        
        #Predict with loaded data
        pred_model = modelx.predict({"multi_data":test_pipe, "emb_text_input": text_test})
        
        #Change prediction from category probability to text
        pred_genre = np.argmax(pred_model, axis=1)  
        pred_ens.append(int(pred_genre))
        pred_prob = np.max(pred_model, axis=1)
        pred_ens_prob.append(pred_prob)
        
    y = max(set(pred_ens), key=pred_ens.count)

    return


def test_model_serving(test_row:int, models_list):
    """
    Function to use saved tensorflow model to do prediction

    Parameters
    ----------
    test_row : Int
        row number from testfile to get input data

    Returns
    -------
    Prints out prediction

    """
    
    test_pipe, text_test = get_test_data(test_row)
    pred_ens = []
    pred_ens_prob = []
    
    for modelx in models_list:
        
        #Predict with loaded data
        pred_model = modelx.predict({"multi_data":test_pipe, "emb_text_input": text_test})
        
        #Change prediction from category probability to text
        pred_genre = np.argmax(pred_model, axis=1)  
        pred_ens.append(int(pred_genre))
        pred_prob = np.max(pred_model, axis=1)
        pred_ens_prob.append(pred_prob)
        
    y = max(set(pred_ens), key=pred_ens.count)

    return


def get_rest_url(model_name, host='127.0.0.1', port='8501', verb='predict', version=None):
    """ generate the URL path for tensorflow serving"""
    url = "http://{host}:{port}/v1/models/{model_name}".format(host=host, port=port, model_name=model_name)
    if version:
        url += 'versions/{version}'.format(version=version)
    url += ':{verb}'.format(verb=verb)
    return url



def test_model_api(test_row:int):
    """
    Makes API call to tensorflow server and prints out the prediction

    """
    #Get data to predict on
    test_pipe, text_test = get_test_data(test_row)
   
    #url name
    url = get_rest_url(model_name = 'wide_folder')
    #'http://127.0.0.1:8501/v1/models/wide1:predict'
    
    #Prep text data to jsonify
    text_test = text_test.astype(str)
    test_pipe_list = test_pipe.tolist()
    text_test_list = text_test.tolist()
    #Make in data json. Use "inputs" for column format
    json_data = json.dumps({"inputs": {
                                    "emb_text_input": text_test_list,
                                    "multi_data": test_pipe_list
                                        }
                            }
                        )
    
    # API call to model
    response = requests.post(url, data= json_data)
    
    #Change prediction from category probability to text
    rev_label_dict = {0:'soul and reggae', 1:'pop', 2:'punk', 3:'jazz and blues', 
                      4:'dance and electronica', 5:'folk', 6:'classic pop and rock', 7:'metal'}
    prob_pred = response.json()['outputs'][0]
    max_prob = np.argmax(np.array(prob_pred)) 
    suggested_genre = rev_label_dict[max_prob]
    
    #Print out prediction
    print("suggested genre is:", suggested_genre)
      
    return  


def test_model_pred(test_pipe, text_test, y_labels_test, models_list):
    """
    Function to use saved tensorflow model to do prediction

    Parameters
    ----------
    test_row : Int
        row number from testfile to get input data

    Returns
    -------
    Prints out prediction

    """
    
    pred_ens = {}
    pred_ens_prob = []
    data_length = len(y_labels_test)
    
    for i,modelx in enumerate(models_list):
        
        #Predict with loaded data
        pred_model = modelx.predict({"multi_data":test_pipe, "emb_text_input": text_test})
        #print(np.round(pred_model[0,:],2))
        #Get array with best prediction
        pred_genre = np.argmax(pred_model, axis=1)  
        #print(pred_genre[0])
        mname = "m" + str(i)
        pred_ens[mname] = pred_genre
        # pred_ens.append(int(pred_genre))
        # pred_prob = np.max(pred_model, axis=1)
        # pred_ens_prob.append(pred_prob)
        
    
    pred_arr = pred_ens['m0'].reshape(data_length,1)
    for i in range(8):
        mname = 'm' + str(i+1)
        pred_arr = np.concatenate([pred_arr, pred_ens[mname].reshape(data_length,1)],axis=1 )
        
    #print("row preds", pred_arr[0,:]) #pred_arr.shape)
    
    y_pred = []
    y_labels = []
    for i in range(data_length):
        counts = np.bincount(pred_arr[i,:])
        y = np.argmax(counts)
        y_pred.append(y)
    
    y_pred_arr = np.array(y_pred)
    y_labels_arr = np.argmax(y_labels_test, axis=1)
    # # to plot and understand confusion matrix
    cm = confusion_matrix(y_labels_arr, y_pred_arr)
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

    mod_metrics(y_labels_arr, y_pred_arr)
    return
  

def close_models():
    tf.keras.backend.clear_session()
    return


if __name__ == "__main__":
    ##Get data
    X_train_pipe, X_test_pipe, text_train, text_test, y_labels, y_labels_test = prep_data()
    ##Train model
    # make_tf_models(X_train_pipe, X_test_pipe, text_train, text_test, y_labels, y_labels_test)
    ##Train ensemble
    # for i in range(9):
    #     seed_no = make_tf_ensemble(X_train_pipe, X_test_pipe, text_train, text_test, 
    #                               y_labels, y_labels_test, i)
    
    ##----------------------------
    ## Check ensemble predictions on train
    #t1 = time.perf_counter()
    models_list = load_models()
    #t2 = time.perf_counter()
    #print(f'Finished in {round(t2 - t1,2)} seconds')
    ##Test prediction with object
    # t1 = time.perf_counter()
    # test_model_pred(X_test_pipe, text_test, y_labels_test, models_list)
    # t2 = time.perf_counter()
    # print(f'Finished in {round(t2 - t1,2)} seconds')
        
    
    ##----------------------------
    ## Load models
    # t1 = time.perf_counter()
    # models_list = load_models()
    # t2 = time.perf_counter()
    # print(f'Finished in {round(t2 - t1,2)} seconds')
    # ##Test prediction with object
    # t1 = time.perf_counter()
    # test_model_serving(212, models_list)
    # t2 = time.perf_counter()
    # print(f'Finished in {round(t2 - t1,2)} seconds')
    
    ##---------------------------
    ##API test prediction
    #test_model_api(94)
    close_models()






























