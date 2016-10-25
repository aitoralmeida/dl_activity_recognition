# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 14:07:36 2016

@author: aitor
"""
from collections import Counter
import csv
import json

from gensim.models import Word2Vec
import numpy


# Dataset generated with the synthetic generator
DATASET = 'action_dataset.csv'
# Text file used to create the action vectors with word2vec
ACTION_TEXT = 'actions.txt'
# List of unique actions in the dataset
UNIQUE_ACTIONS = 'unique_actions.json'
# Word2vec model generated with gensim
ACTIONS_MODEL = 'actions.model'
# Vector values for each action
ACTIONS_VECTORS = 'actions_vectors.json'
# File with the activities ordered
ACTIVITIES_ORDERED = 'activities.json'
# Dataset with vectors but without the action timestamps
DATASET_NO_TIME = 'dataset_no_time.json'

# When there is no activity
NONE = 'None'
# Separator for the text file
SEP = ' '

# Generates the text file from the csv
def process_csv():
    actions = ''    
    actions_set = set()
    with open(DATASET, 'rb') as csvfile:
        print 'Processing:', DATASET
        reader = csv.reader(csvfile)        
        i = 0
        for row in reader:
            i += 1
            if i == 1:
                continue        
            action = row[1]
            activity = row[2]
            if activity != NONE:
                actions += action + SEP
                actions_set.add(action)
            if i % 10000 == 0:
                print '  -Actions processed:', i
        print 'Total actions processed:', i
    
    with open(ACTION_TEXT, 'w') as textfile: 
        textfile.write(actions)     
    json.dump(list(actions_set), open(UNIQUE_ACTIONS, 'w'))
    print 'Text file saved'

# creates a json file with the action vectors from the gensim model
def create_vector_file():
    print 'Creating the vector file...'
    actions = json.load(open(UNIQUE_ACTIONS, 'r'))
    print 'Total unique actions:', len(actions)
    model = Word2Vec.load(ACTIONS_MODEL)
    actions_vectors = {}
    for action in actions:
        actions_vectors[action] = model[action].tolist()
     
    json.dump(actions_vectors, open(ACTIONS_VECTORS, 'w'), indent=2)
    print 'Saved action vectors'

# Processes the csv and orders the activities in a json    
def order_activities():
    with open(DATASET, 'rb') as csvfile:
        print 'Processing:', DATASET
        reader = csv.reader(csvfile)        
        i = 0        
        current_activity = {
            'name':'',
            'actions': []
        }
        activities = []
        for row in reader:
            i += 1
            if i == 1:
                continue        
            date = row[0]            
            action = row[1]
            activity = row[2]
            if activity != NONE and activity != '':
                if activity == current_activity['name']:
                    action_data = {
                        'action':action,
                        'date':date                
                    }
                    current_activity['actions'].append(action_data)
                else:
                    activities.append(current_activity)
                    current_activity = {
                        'name':activity,
                        'actions': []
                    }
                    action_data = {
                        'action':action,
                        'date':date                
                    }
                    current_activity['actions'].append(action_data)
            if i % 10000 == 0:
                print 'Actions processed:', i
        json.dump(activities, open(ACTIVITIES_ORDERED, 'w'), indent=1)
    print 'Ordered activities'
    
def median(lst):
    return numpy.median(numpy.array(lst))
 
# Statistics about the activities   
def calculate_statistics():
    print 'Calculating statistics'
    activities = json.load(open(ACTIVITIES_ORDERED, 'r'))
    total_activities = len(activities)
    print 'Total activities:', total_activities
    action_lengths = []
    for activity in activities:
        action_lengths.append(len(activity['actions']))
    print 'Avg activity lenght:', sum(action_lengths)/total_activities
    print 'Median activity lenght:', median(action_lengths)
    print 'Longest activity:', max(action_lengths)
    print 'Shortest activity:', min(action_lengths)
    distribution = Counter(action_lengths)
    print 'Distribution:', json.dumps(distribution, indent=2)
    
def create_vector_dataset_no_time():
    print 'Creating dataset...'
    dataset = []
    action_vectors = json.load(open(ACTIONS_VECTORS, 'r'))
    activities = json.load(open(ACTIVITIES_ORDERED, 'r'))
    for activity in activities:
        training_example = {
            'activity' : activity['name'],
            'actions' : []         
        }
        for action in activity['actions']:
            training_example['actions'].append(action_vectors[action['action']])

        # Padding        
        if len(training_example['actions']) < 10:
            for i in range(10 - len(training_example['actions'])):
              training_example['actions'].append([0] * 200)
              
        dataset.append(training_example)
    print 'Writing file'
    json.dump(dataset, open(DATASET_NO_TIME,'w'))
    print 'Created dataset'
            
    
    
        
    
    
if __name__ == '__main__':
    print 'Start...'
    #process_csv()
    #create_vector_file()
    #order_activities()
    #calculate_statistics()
    create_vector_dataset_no_time()
    print 'Fin'



            

