# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 14:59:07 2021

@author: PatCa
"""

import numpy as np
import pandas as pd
from pickle import load
import requests
import json
import os
import sys
import pathlib
import joblib

import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub

CURRENTDIR = pathlib.Path(__file__).resolve().parent
sys.path.append(str(CURRENTDIR))
#Import data processing function 
from cleaning_functions import PCA_Data




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
                           tf.keras.callbacks.ModelCheckpoint(os.path.join(WORKING_DIR,
                                                                 'wide_folder/1'),
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
    #Import pipeline and prediction model
    pipe = joblib.load('model_artifacts/pipe.joblib')
    
    #get dataset
    test_data = pd.read_csv('data/test.csv')
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
    pca_scaler = load(open('model_artifacts/pca_scaler.pkl', 'rb'))
    pca_test_data_norm = pca_scaler.transform(pca_test_data)
    
    #Get pca transformer and run on data
    pca = load(open('model_artifacts/pca.pkl', 'rb')) 
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



def test_model_serving(test_row:int):
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
    #Get test data
    test_pipe, text_test = get_test_data(test_row)
    
    #Get model from folder, load model
    WORKING_DIR = os.getcwd() 
    path = os.path.join(WORKING_DIR,'wide_folder/1')
    reconstructed_model = keras.models.load_model(path)
    
    #Predict with loaded data
    pred_model = reconstructed_model.predict({"multi_data":test_pipe, "emb_text_input": text_test})
    
    #Change prediction from category probability to text
    pred_genre = np.argmax(pred_model, axis=1)    
    rev_label_dict = {0:'soul and reggae', 1:'pop', 2:'punk', 3:'jazz and blues', 
                      4:'dance and electronica', 5:'folk', 6:'classic pop and rock', 7:'metal'}
    suggested_genre = rev_label_dict[pred_genre[0]]
    
    #Print prediction
    print("suggested genre is:", suggested_genre)
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



if __name__ == "__main__":
    ##Get data
    X_train_pipe, X_test_pipe, text_train, text_test, y_labels, y_labels_test = prep_data()
    ##Train model
    make_tf_models(X_train_pipe, X_test_pipe, text_train, text_test, y_labels, y_labels_test)
    ##Test prediction with object
    #test_model_serving(94)
    ##API test prediction
    #test_model_api(94)































