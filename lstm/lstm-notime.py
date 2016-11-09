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


from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
import numpy as np

# Directory of datasets
DIR = '../sensor2vec/kasteren_dataset/'
# Dataset with vectors but without the action timestamps
DATASET_NO_TIME = DIR + 'dataset_no_time.json'
# List of unique actions in the dataset
UNIQUE_ACTIVITIES = DIR + 'unique_activities.json'

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
        
    
def main(argv):    
    unique_activities = json.load(open(UNIQUE_ACTIVITIES, 'r'))
    total_activities = len(unique_activities)

    print 'Preparing training set...'
    print '  - Reading dataset'
    sys.stdout.flush()
    with open(DATASET_NO_TIME, 'r') as dataset_file:
        activities = json.load(dataset_file)
    print '  - processing activities'
    X = []
    y = []
    for i, activity in enumerate(activities):
        if i % 100 == 0:
            print '  - Number of activities processed:', i
            sys.stdout.flush()
        actions = []
        for action in activity['actions']:
            actions.append(np.array(action))
        X.append(np.array(actions))
        y.append(np.array(activity['activity']))
    
    
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
    batch_size = 1
    model = Sequential()
    model.add(LSTM(512, return_sequences=False, stateful=False, dropout_W=0.2, dropout_U=0.2, batch_input_shape=(batch_size, max_sequence_length, ACTION_MAX_LENGHT)))
    #model.add(LSTM(512, return_sequences=False, stateful=True, batch_input_shape=(batch_size, max_sequence_length, ACTION_MAX_LENGHT)))
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
    history = model.fit(X, y, batch_size=batch_size, nb_epoch=500, validation_data=(X_test, y_test), shuffle=False)
 
    # Test manual training
    # we need a manual history dict with 'acc', 'val_acc', 'loss' and 'val_loss' keys
    """
    manual_training = True
    history = {}
    history['acc'] = []
    history['val_acc'] = []
    history['loss'] = []
    history['val_loss'] = []
    for i in range(1000):
        print 'epoch: ', i
        hist = model.fit(X, y, nb_epoch=1, batch_size=batch_size, shuffle=False, validation_data=(X_test, y_test))
        history['acc'].append(hist.history['acc'])
        history['val_acc'].append(hist.history['val_acc'])
        history['loss'].append(hist.history['loss'])
        history['val_loss'].append(hist.history['val_loss'])
        model.reset_states()
    """
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
