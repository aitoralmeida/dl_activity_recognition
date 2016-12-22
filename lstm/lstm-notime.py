# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 16:16:52 2016

@author: gazkune
"""

from collections import Counter
import json
import sys
from copy import deepcopy

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd

from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences

import numpy as np

# Directory of datasets
DIR = '../sensor2vec/kasteren_dataset/'
# Dataset with vectors but without the action timestamps
DATASET_CSV = DIR + 'base_kasteren_reduced.csv'
DATASET_NO_TIME = DIR + 'dataset_no_time.json'
# List of unique activities in the dataset
UNIQUE_ACTIVITIES = DIR + 'unique_activities.json'
# List of unique actions in the dataset
UNIQUE_ACTIONS = DIR + 'unique_actions.json'
# Action vectors
ACTION_VECTORS = DIR + 'actions_vectors.json'

# Maximun number of actions in an activity
ACTIVITY_MAX_LENGHT = 32
# Number of dimensions of an action vector
ACTION_MAX_LENGHT = 50





def save_model(model):
    json_string = model.to_json()
    model_name = 'model_activity_lstm'
    open(model_name + '.json', 'w').write(json_string)
    model.save_weights(model_name + '.h5', overwrite=True)
    
def load_model(model_file, weights_file):
    model = model_from_json(open(model_file).read())
    model.load_weights(weights_file)
    
def check_activity_distribution(y_np, unique_activities):
    activities = []
    for activity_np in y_np:
        index = activity_np.tolist().index(1.0)
        activities.append(unique_activities[index])
    print Counter(activities)
    
def prepare_variable_sequences(df, action_vectors, unique_activities, activity_to_int):
    X = []
    y = []    
    
    current_activity = ""
    actions = []
    aux_actions = []
    for index in df.index:        
        if current_activity == "":
            current_activity = df.loc[index, 'activity']
            
        
        if current_activity != df.loc[index, 'activity']:
            y.append(activity_to_int[current_activity])
            X.append(actions)            
            #print current_activity, aux_actions
            current_activity = df.loc[index, 'activity']
            # reset auxiliary variables
            actions = []
            aux_actions = []
        
        action = df.loc[index, 'action']
        #print 'Current action: ', action
        actions.append(np.array(action_vectors[action]))
        aux_actions.append(action)
        
    # Append the last activity
    y.append(activity_to_int[current_activity])
    X.append(actions)
    
    # Use sequence padding for training samples
    X = pad_sequences(X, maxlen=ACTIVITY_MAX_LENGHT, dtype='float32')    
        
    return X, y
    
    
"""
Function to plot accurary and loss during training
"""

def plot_training_info(metrics, save, history):
    # summarize history for accuracy
    if 'accuracy' in metrics:
        
        plt.plot(history['acc'])
        plt.plot(history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        if save == True:
            plt.savefig('accuracy.png')
            plt.gcf().clear()
        else:
            plt.show()

    # summarize history for loss
    if 'loss' in metrics:
        plt.plot(history['loss'])
        plt.plot(history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        #plt.ylim(1e-3, 1e-2)
        plt.yscale("log")
        plt.legend(['train', 'test'], loc='upper left')
        if save == True:
            plt.savefig('loss.png')
            plt.gcf().clear()
        else:
            plt.show()
        
    
def main(argv):
    
    # Load dataset from csv file
    df_dataset = pd.read_csv(DATASET_CSV, parse_dates=[[0, 1]], header=None, index_col=0, sep=' ')
    df_dataset.columns = ['sensor', 'action', 'event', 'activity']
    df_dataset.index.names = ["timestamp"]
    unique_activities = json.load(open(UNIQUE_ACTIVITIES, 'r'))
    total_activities = len(unique_activities)
    action_vectors = json.load(open(ACTION_VECTORS, 'r'))
    
    # Generate the dict to transform activities to integer numbers
    activity_to_int = dict((c, i) for i, c in enumerate(unique_activities))
    # Generate the dict to transform integer numbers to activities
    int_to_activity = dict((i, c) for i, c in enumerate(unique_activities))

    # Prepare padded variable sequences for activities
    # This data framing approach assumes a perfect segmentation of actions for activities
    X, y = prepare_variable_sequences(df_dataset, action_vectors, unique_activities, activity_to_int)    
    
    # Keep original y (with activity indices) before transforming it to categorical
    y_orig = deepcopy(y)
    # Tranform class labels to one-hot encoding
    y = np_utils.to_categorical(y)
    
    
    # Prepare training and testing datasets
    total_examples = len(X)
    test_per = 0.2
    limit = int(test_per * total_examples)
    #======================================================
    # Be careful here! Training set is built from limit
    # Take into account for visualizations!
    #======================================================
    X_train = X[limit:]
    X_test = X[:limit]
    y_train = y[limit:]
    y_test = y[:limit]
    print 'Total examples:', total_examples
    print 'Train examples:', len(X_train), len(y_train) 
    print 'Test examples:', len(X_test), len(y_test)
    sys.stdout.flush()  
    X = np.array(X_train)
    y = np.array(y_train)
    print 'Activity distribution for training:'
    check_activity_distribution(y, unique_activities)
    #X = X.reshape(X.shape[0], 1, ACTIVITY_MAX_LENGHT, ACTION_MAX_LENGHT)
    X_test = np.array(X_test)
    y_test = np.array(y_test)    

    print 'Activity distribution for testing:'
    check_activity_distribution(y_test, unique_activities)

    #X_test = X_test.reshape(X_test.shape[0], 1, ACTIVITY_MAX_LENGHT, ACTION_MAX_LENGHT) # For multichannel CNN
    X = X.reshape(X.shape[0], ACTIVITY_MAX_LENGHT, ACTION_MAX_LENGHT)
    X_test = X_test.reshape(X_test.shape[0], ACTIVITY_MAX_LENGHT, ACTION_MAX_LENGHT)
    print 'Shape (X,y):'
    print X.shape
    print y.shape
    print 'Training set prepared'  
    sys.stdout.flush()   

    # Build the model
    max_sequence_length = ACTIVITY_MAX_LENGHT
    
    print 'Building model...'
    sys.stdout.flush()
    batch_size = 16
    model = Sequential()
    model.add(LSTM(128, return_sequences=False, dropout_W=0.4, dropout_U=0.4, input_shape=(ACTIVITY_MAX_LENGHT, ACTION_MAX_LENGHT)))
    #model.add(Dropout(0.8))
    #model.add(LSTM(512, return_sequences=False, dropout_W=0.2, dropout_U=0.2))
    #model.add(Dropout(0.8))
    model.add(Dense(total_activities))
    model.add(Activation('softmax'))

    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'mse', 'mae'])
    print 'Model built'
    print(model.summary())
    sys.stdout.flush()
  

    print 'Training...'    
    sys.stdout.flush()
    # Automatic training for Stateless LSTM
    manual_training = False
    history = model.fit(X, y, batch_size=batch_size, nb_epoch=1000, validation_data=(X_test, y_test), shuffle=False)
    
    print 'Saving model...'
    sys.stdout.flush()
    save_model(model)
    print 'Model saved'
    if manual_training == True:
        plot_training_info(['accuracy', 'loss'], True, history)
    else:
        plot_training_info(['accuracy', 'loss'], True, history.history)


if __name__ == "__main__":
   main(sys.argv)
