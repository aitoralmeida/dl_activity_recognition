# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 11:41:35 2017

@author: gazkune
"""

from collections import Counter
import json
import sys
from copy import deepcopy

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import math

import pandas as pd

"""
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense, Activation, Embedding, Input, Dropout
from keras.layers import LSTM
from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

from gensim.models import Word2Vec
"""

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
# Word2Vec model
WORD2VEC_MODEL = DIR + 'actions.model'

# Maximun number of actions in an activity
ACTIVITY_MAX_LENGHT = 32
# Number of dimensions of an action vector
ACTION_MAX_LENGHT = 50



def transform_time_cyclic(timestamp, weekday):
    """
    This function transforms a timestamp into a cyclic clock-based time representation

    Parameters
    ----------        
    timestamp : datetime.datetime
        the timestamp to be transformed
    weekday : boolean
        a boolean to say whether the weekday should be treated for the calculation
                    
    Returns
    ----------
    x : float
        x coordinate of the 2D plane defining the clock [-1, 1]
    y : float
        y coordinate of the 2D plane defining the clock [-1, 1]
    """
    # Timestamp comes in datetime.datetime format
    HOURS = 24
    MINUTES = 60
    SECONDS = 60
    
    MAX_SECONDS = 0.0
    total_seconds = -1.0 # For error checking
    
    if weekday == True:    
        MAX_SECONDS = float(6*HOURS*MINUTES*SECONDS + 23*MINUTES*SECONDS + 59*SECONDS + 59)
        total_seconds = float(timestamp.weekday()*HOURS*MINUTES*SECONDS + timestamp.hour*MINUTES*SECONDS + timestamp.minute*SECONDS + timestamp.second)
    else:
        MAX_SECONDS = float(23*MINUTES*SECONDS + 59*SECONDS + 59)
        total_seconds = float(timestamp.hour*MINUTES*SECONDS + timestamp.minute*SECONDS + timestamp.second)
    
        
    angle = (total_seconds*2*math.pi) / MAX_SECONDS
    
    x = math.cos(angle)
    y = math.sin(angle)
    
    return x, y
    
    
def transform_time_linear(times):
    """
    This function transforms a timestamp into a linear representation

    Parameters
    ----------        
    times : pandas.tseries.index.DatetimeIndex
        a pandas series with all the timestamps to be transformed
                    
    Returns
    ----------
    linear_times : list of int
        a list of the transformed timestamps    
    """
    diff_times = times - times[0]
    linear_times = diff_times / diff_times[len(diff_times) - 1]
    
    return linear_times
    

# Main function
def main(argv):
    
    # Load dataset from csv file
    df_dataset = pd.read_csv(DATASET_CSV, parse_dates=[[0, 1]], header=None, index_col=0, sep=' ')
    df_dataset.columns = ['sensor', 'action', 'event', 'activity']
    df_dataset.index.names = ["timestamp"]
    
    linear_times = transform_time_linear(df_dataset.index)
    
    for i in xrange(10):
        t = df_dataset.index[i]
        print '--------------------------------------------------------'
        print t, 'day:', t.to_datetime().weekday()
        x, y = transform_time_cyclic(df_dataset.index[i].to_datetime(), True)
        print 'Weekday True: (', x, ',', y, ')'
        x, y = transform_time_cyclic(df_dataset.index[i].to_datetime(), False)
        print 'Weekday False: (', x, ',', y, ')'
        print 'Linear:', linear_times[i]
    
    # Test also the last element as a special case    
    i = len(df_dataset)-1    
    t = df_dataset.index[i]
    print '--------------------------------------------------------'
    print t, 'day:', t.to_datetime().weekday()
    x, y = transform_time_cyclic(df_dataset.index[i].to_datetime(), True)
    print 'Weekday True: (', x, ',', y, ')'
    x, y = transform_time_cyclic(df_dataset.index[i].to_datetime(), False)
    print 'Weekday False: (', x, ',', y, ')'
    print 'Linear:', linear_times[i]
    
if __name__ == "__main__":
   main(sys.argv)