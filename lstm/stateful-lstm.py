# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 16:16:52 2016

@author: gazkune
"""

from collections import Counter
import json
import sys

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
  
"""
Function to prepare the dataset with individual sequences (simple framing)
"""

def prepare_indiv_sequences(df, action_vectors, unique_activities, activity_to_int):
    print 'Preparing training set...'

    X = []
    y = []    
    
    for index in df.index:
        action = df.loc[index, 'action']        
        X.append(np.array(action_vectors[action]))
        y.append(activity_to_int[df.loc[index, 'activity']])
    
    y = np_utils.to_categorical(y)
    
    return X, y
    
def prepare_variable_sequences(df, action_vectors, unique_activities, activity_to_int):
    # New data framing
    print 'Preparing training set...'

    X = []
    y = []    
    
    current_activity = ""
    actions = []
    for index in df.index:
        action = df.loc[index, 'action']        
        actions.append(np.array(action_vectors[action]))
        
        if current_activity != df.loc[index, 'activity']:
            current_activity = df.loc[index, 'activity']
            if len(actions) > 0:
                X.append(actions)
                y.append(activity_to_int[df.loc[index, 'activity']])
                actions = []

    # Use sequence padding for training samples
    X = pad_sequences(X, maxlen=ACTIVITY_MAX_LENGHT, dtype='float32')
    # Tranform class labels to one-hot encoding
    y = np_utils.to_categorical(y)
    
    return X, y
    
    
def main(argv):
    # Load dataset from csv file
    df_dataset = pd.read_csv(DATASET_CSV, parse_dates=[[0, 1]], header=None, index_col=0, sep=' ')
    df_dataset.columns = ['sensor', 'action', 'event', 'activity']
    df_dataset.index.names = ["timestamp"]
    unique_activities = json.load(open(UNIQUE_ACTIVITIES, 'r'))
    total_activities = len(unique_activities)
    action_vectors = json.load(open(ACTION_VECTORS, 'r'))

    print 'Preparing training set...'
    
    # Generate the dict to transform activities to integer numbers
    activity_to_int = dict((c, i) for i, c in enumerate(unique_activities))
    # Generate the dict to transform integer numbers to activities
    int_to_activity = dict((i, c) for i, c in enumerate(unique_activities))

    
    # Test the simple problem framing
    #X, y = prepare_indiv_sequences(df_dataset, action_vectors, unique_activities, activity_to_int)
    #variable = False
    
    # Test the varaible sequence problem framing approach
    # Remember to change batch_input_size in consequence
    X, y = prepare_variable_sequences(df_dataset, action_vectors, unique_activities, activity_to_int)
    variable = True
    
    total_examples = len(X)
    test_per = 0.2
    limit = int(test_per * total_examples)
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
    
    X_test = np.array(X_test)
    y_test = np.array(y_test)    

    print 'Activity distribution for testing:'
    check_activity_distribution(y_test, unique_activities)
    
    # Current shape of X and y
    print 'X:', X.shape
    print 'y:', y.shape

    # reshape X and X_test to be [samples, time steps, features]
    # In this test we will set timesteps to 1 even though we have padded sequences
    time_steps = 1
        
    #X = X.reshape(X.shape[0], time_steps, ACTION_MAX_LENGHT)
    #X_test = X_test.reshape(X_test.shape[0], time_steps, ACTION_MAX_LENGHT)
    print 'Shape (X,y):'
    print X.shape
    print y.shape
    print 'Training set prepared'  
    sys.stdout.flush()

    # Build the model

        
    print 'Building model...'
    sys.stdout.flush()
    batch_size = 1
    model = Sequential()
    # Test with Stateful layers
    # I read that batch_input_size=(batch_size, None, features) can be used for variable length sequences
    model.add(LSTM(512, return_sequences=False, stateful=True, dropout_W=0.2, dropout_U=0.2, batch_input_shape=(batch_size, time_steps, X.shape[2])))
    #model.add(LSTM(512, return_sequences=False, stateful=True, batch_input_shape=(batch_size, max_sequence_length, ACTION_MAX_LENGHT)))
    #model.add(Dropout(0.8))
    #model.add(LSTM(512, return_sequences=False, dropout_W=0.2, dropout_U=0.2))
    #model.add(Dropout(0.8))
    model.add(Dense(total_activities, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'mse', 'mae'])
    print 'Model built'
    print(model.summary())
    sys.stdout.flush()
  

    print 'Training...'    
    sys.stdout.flush()
    
    # Test manual training
    # we need a manual history dict with 'acc', 'val_acc', 'loss' and 'val_loss' keys
    manual_training = True
    history = {}
    history['acc'] = []
    history['val_acc'] = []
    history['loss'] = []
    history['val_loss'] = []
    """
    for i in range(10):
        print 'epoch: ', i
        model.fit(X, y, nb_epoch=1, batch_size=batch_size, shuffle=False)
        
        hist = model.fit(X, y, nb_epoch=1, batch_size=batch_size, shuffle=False, validation_data=(X_test, y_test))
        history['acc'].append(hist.history['acc'])
        history['val_acc'].append(hist.history['val_acc'])
        history['loss'].append(hist.history['loss'])
        history['val_loss'].append(hist.history['val_loss'])
        
        model.reset_states()
 
    print 'Saving model...'
    sys.stdout.flush()
    save_model(model)
    print 'Model saved'
    """
    # Check data format visually
    print 'X train shape:', X.shape
    print X    

    sample = np.expand_dims(np.expand_dims(X[0][0], axis=0), axis=0)
    print 'sample shape:', sample.shape
    print sample

    other = X[0][0]
    print 'other shape:', other.shape
    print other
    
    
    
    # This training process is to test variable length sequences representing an activity
    # for stateful LSTMs. We train and test batch per batch
    max_len = X.shape[1]
    print 'max length:', max_len
    epochs = 100
    for epoch in range(epochs):
        print '***************'
        print 'Epoch', epoch, '/', epochs
        mean_tr_acc = []
        mean_tr_loss = []
        for i in range(len(X)):
            y_true = y[i]
            #print 'y_true:', np.array([y_true]), np.array([y_true]).shape            
            for j in range(max_len):
                x = np.expand_dims(np.expand_dims(X[i][j], axis=0), axis=0)
                #tr_loss, tr_acc = model.train_on_batch(x, np.array([y_true]))
                hist = model.fit(x, np.array([y_true]), nb_epoch=1, batch_size=1, shuffle=False, verbose=0)
                mean_tr_acc.append(hist.history["acc"])
                mean_tr_loss.append(hist.history["loss"])
            model.reset_states()

        print('accuracy training = {}'.format(np.mean(mean_tr_acc)))
        print('loss training = {}'.format(np.mean(mean_tr_loss)))
        print('___________________________________')
        
        """
        # Comment for now the testing step
        mean_te_acc = []
        mean_te_loss = []
        for i in range(len(X_test)):
            for j in range(max_len):
                te_loss, te_acc = model.test_on_batch(np.expand_dims(np.expand_dims(X_test[i][j], axis=0), axis=0), y_test[i])
                mean_te_acc.append(te_acc)
                mean_te_loss.append(te_loss)
            model.reset_states()

            for j in range(max_len):
                y_pred = model.predict_on_batch(np.expand_dims(np.expand_dims(X_test[i][j], axis=0), axis=0))
            model.reset_states()

        print('accuracy testing = {}'.format(np.mean(mean_te_acc)))
        print('loss testing = {}'.format(np.mean(mean_te_loss)))
        print('___________________________________')    
        """
    
    
    # summarize performance of the model testing the evaluate function
    #scores = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=0)
    #print("Model Accuracy: %.2f%%" % (scores[1]*100))
    
    """
    if manual_training == True:
        plot_training_info(['accuracy', 'loss'], True, history)
    else:
        plot_training_info(['accuracy', 'loss'], True, history.history)
    """

if __name__ == "__main__":
   main(sys.argv)
