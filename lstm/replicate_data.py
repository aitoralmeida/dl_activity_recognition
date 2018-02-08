#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 15:38:49 2018

@author: gazkune
"""
import sys
import numpy as np
from collections import Counter


# Directory of formatted datasets
INPUT_DIR = 'formatted_data/'
ROOT_NAME = 'aruba_continuous_no_t_50_10_stratified'
INPUT_ROOT_NAME = INPUT_DIR + ROOT_NAME
INPUT_X_TAIL = '_60_x_val.npy'
INPUT_Y_TAIL = '_60_y_val.npy'

# Store results in
OUTPUT_X_FILE = INPUT_DIR + 'balanced_' + ROOT_NAME + INPUT_X_TAIL
OUTPUT_Y_FILE = INPUT_DIR + 'balanced_' + ROOT_NAME + INPUT_Y_TAIL


# Main function
def main(argv):
    
    print 'Loading data'
    
    # We load here the action indices
    X = np.load(INPUT_ROOT_NAME + INPUT_X_TAIL)
    # We load here activity labels using one-hot vector encoding
    y = np.load(INPUT_ROOT_NAME + INPUT_Y_TAIL)
    
    # We transform one-hot vector to integer codes
    y_code = np.array([np.argmax(y[x]) for x in xrange(len(y))])
    
    print 'X shape:', X.shape
    print 'y shape:', y.shape
    print 'y_code shape:', y_code.shape
    
    # We calculate the distribution of activities
    distro_dict = Counter(y_code)
    print 'Activity distribution:', distro_dict 
    max_instances = max(distro_dict.values())
    print 'Maximum number of instances:', max_instances
    
    # 'mew_x' and 'new_y' are used to generate the replicated and balanced dataset
    new_x = np.copy(X)
    new_y = np.copy(y)
    for key in distro_dict.keys():
        instances = distro_dict[key]
        print 'Instances for activity', key, ':', instances
        # Obtain the indices where y_train_code equals the activity in key
        indices = np.where(y_code == key)
        # As np.where returns a tuple we will extract the internal np.array
        indices = indices[0]
        # Obtain the samples we can use for replication
        samples = X[indices]
        labels = y[indices]
        print '  Shape of samples:', samples.shape
        print '  Shape of labels:', labels.shape
        
        # We will randomly pick up a sample from 'samples' and append to X_train
        # In consequence, we will also append the activity label to y_train
        # We will use np.random.choice for that purpose
        if instances < max_instances:
            rep_indices = np.random.choice(len(samples), max_instances - instances)
            print '  Number of instances to replicate:', len(rep_indices)
            print '  Shape of new_x:', new_x.shape
            print '  Shape of new_y:', new_y.shape
            new_x = np.append(new_x, samples[rep_indices], axis=0)
            new_y = np.append(new_y, labels[rep_indices], axis=0)
            print '  After append: shape of new_x:', new_x.shape
            print '  After append: shape of new_y:', new_y.shape
        
          
        print 'After replication'
        new_y_codes = np.array([np.argmax(new_y[x]) for x in xrange(len(new_y))])
        count = np.where(new_y_codes == key)
        count = count[0]
        print 'Instances for activity', key, len(count)
    
    new_distro_dict = Counter(new_y_codes)
    print 'New activity distribution:'
    print new_distro_dict
    
    print 'New x_train shape:', new_x.shape
    print 'New y_train shape:', new_y.shape
    
    print 'Save new_x to', OUTPUT_X_FILE
    print 'Save new_y to', OUTPUT_Y_FILE
    np.save(OUTPUT_X_FILE, new_x)
    np.save(OUTPUT_Y_FILE, new_y)
    
    
if __name__ == "__main__":
   main(sys.argv)