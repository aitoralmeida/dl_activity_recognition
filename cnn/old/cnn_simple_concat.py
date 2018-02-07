# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 08:47:06 2016

@author: aitor
"""

from collections import Counter
import json
import sys

from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense, Activation, Dropout
from keras.layers import Convolution2D, MaxPooling2D, Flatten
import numpy as np

# Dataset with vectors but without the action timestamps
DATASET_NO_TIME = 'dataset_no_time_2_channels.json'
# List of unique actions in the dataset
UNIQUE_ACTIVITIES = 'unique_activities.json'

# Maximun number of actions in an activity
ACTIVITY_MAX_LENGHT = 32
# Number of dimensions in a action
ACTION_MAX_LENGHT = 50


def save_model(model):
    json_string = model.to_json()
    model_name = 'model_activity_cnn'
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
        
    
    
unique_activities = json.load(open(UNIQUE_ACTIVITIES, 'r'))
total_activities = len(unique_activities)
    
print 'Building model...'
sys.stdout.flush()
classifier = Sequential()
# filter = 2 x ACTION_MAX_LENGHT
# input = 1 channel x 10 X 200
classifier.add(Convolution2D(100, 3, ACTION_MAX_LENGHT, input_shape=(1,ACTIVITY_MAX_LENGHT * 2,ACTION_MAX_LENGHT),border_mode='valid',activation='relu'))
# Same height as the filter heigh
classifier.add(MaxPooling2D(pool_size=(30,1)))
classifier.add(Flatten())
classifier.add(Dropout(0.5))
classifier.add(Dense(total_activities))
classifier.add(Activation('softmax'))
classifier.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'mse', 'mae'])
print 'Model built'
sys.stdout.flush()

print 'Preparing training set...'
print '  - Reading dataset'
sys.stdout.flush()
with open(DATASET_NO_TIME, 'r') as dataset_file:
    activities = json.load(dataset_file)
print '  - processing activities'
X = []
y = []
for i, activity in enumerate(activities):
    if i % 10000 == 0:
        print '  - Number of activities processed:', i
        sys.stdout.flush()
    actions = []    
    for action in activity['previous_actions']:
        actions.append(np.array(action))        
    for action in activity['actions']:
        actions.append(np.array(action))
#    actions_array = np.array(actions)
#    previous_actions_array = np.array(previous_actions)
    activity_actions = np.array(actions)
    X.append(activity_actions)
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
X = X.reshape(X.shape[0], 1, ACTIVITY_MAX_LENGHT * 2, ACTION_MAX_LENGHT)
X_test = np.array(X_test)
y_test = np.array(y_test)
print 'Activity distribution for testing:'
check_activity_distribution(y_test, unique_activities)
X_test = X_test.reshape(X_test.shape[0], 1, ACTIVITY_MAX_LENGHT * 2, ACTION_MAX_LENGHT)
print 'Shape (X,y):'
print X.shape
print y.shape
print 'Training set prepared'  
sys.stdout.flush()  

print 'Training...'
sys.stdout.flush()
classifier.fit(X, y, batch_size=50, nb_epoch=2000, validation_data=(X_test, y_test))
print 'Saving model...'
sys.stdout.flush()
save_model(classifier)
print 'Model saved'
