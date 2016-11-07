# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 11:34:53 2016

@author: aitor
"""

from collections import Counter
import json
import sys

from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense, Activation, Dropout
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Merge
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

branch_previous= Sequential()
branch_previous.add(Convolution2D(100, 3, ACTION_MAX_LENGHT, input_shape=(1,ACTIVITY_MAX_LENGHT,ACTION_MAX_LENGHT),border_mode='valid',activation='relu'))
branch_previous.add(MaxPooling2D(pool_size=(ACTIVITY_MAX_LENGHT-3+1,1))) # 1-max-pool

branch_currrent= Sequential()
branch_currrent.add(Convolution2D(100, 3, ACTION_MAX_LENGHT, input_shape=(1,ACTIVITY_MAX_LENGHT,ACTION_MAX_LENGHT),border_mode='valid',activation='relu'))
branch_currrent.add(MaxPooling2D(pool_size=(ACTIVITY_MAX_LENGHT-3+1,1))) # 1-max-pool

merged = Merge([branch_previous, branch_currrent], mode='concat', concat_axis=2)

classifier = Sequential()
classifier.add(merged)
classifier.add(Convolution2D(100, 2, 1, input_shape=(1,100,2),border_mode='valid',activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2-2+1,1)))
classifier.add(Flatten())
classifier.add(Dropout(0.5))
classifier.add(Dense(total_activities))
classifier.add(Activation('softmax'))
classifier.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'mse', 'mae'])
classifier.summary()
print 'Model built'
sys.stdout.flush()

print 'Preparing training set...'
print '  - Reading dataset'
sys.stdout.flush()
with open(DATASET_NO_TIME, 'r') as dataset_file:
    activities = json.load(dataset_file)
print '  - processing activities'
X_prev = []
X_act = []
y = []
for i, activity in enumerate(activities):
    if i % 10000 == 0:
        print '  - Number of activities processed:', i
        sys.stdout.flush()
    
    previous_actions = []
    for action in activity['previous_actions']:
        previous_actions.append(np.array(action))  
        
    actions = []  
    for action in activity['actions']:
        actions.append(np.array(action))

    prev_activity_actions = np.array(previous_actions)
    X_prev.append(prev_activity_actions)
    activity_actions = np.array(actions)
    X_act.append(activity_actions)
    
    y.append(np.array(activity['activity']))      

total_examples = len(X_act)
test_per = 0.2
limit = int(test_per * total_examples)
X_train_prev = X_prev[limit:]
X_test_prev = X_prev[:limit]
X_train_act = X_act[limit:]
X_test_act = X_act[:limit]
y_train = y[limit:]
y_test = y[:limit]
print 'Total examples:', total_examples
print 'Train examples:', len(X_train_act), len(y_train) 
print 'Test examples:', len(X_test_act), len(y_test)
sys.stdout.flush()  
X_prev = np.array(X_train_prev)
X_act = np.array(X_train_act)
y = np.array(y_train)
print 'Activity distribution for training:'
check_activity_distribution(y, unique_activities)
X_prev = X_prev.reshape(X_prev.shape[0], 1, ACTIVITY_MAX_LENGHT, ACTION_MAX_LENGHT)
X_act = X_act.reshape(X_act.shape[0], 1, ACTIVITY_MAX_LENGHT, ACTION_MAX_LENGHT)
X_test_prev = np.array(X_test_prev)
X_test_act = np.array(X_test_act)
y_test = np.array(y_test)
print 'Activity distribution for testing:'
check_activity_distribution(y_test, unique_activities)
X_test_prev = X_test_prev.reshape(X_test_prev.shape[0], 1, ACTIVITY_MAX_LENGHT, ACTION_MAX_LENGHT)
X_test_act = X_test_act.reshape(X_test_act.shape[0], 1, ACTIVITY_MAX_LENGHT, ACTION_MAX_LENGHT)
print 'Shape (X,y):'
print X_act.shape
print y.shape
print 'Training set prepared'  
sys.stdout.flush()  

print 'Training...'
sys.stdout.flush()
classifier.fit([X_prev, X_act], y, batch_size=50, nb_epoch=2000, validation_data=([X_test_prev, X_test_act], y_test))

print 'Saving model...'
sys.stdout.flush()
save_model(classifier)
print 'Model saved'

