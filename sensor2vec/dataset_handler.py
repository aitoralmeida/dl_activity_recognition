# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 14:07:36 2016

@author: aitor
"""
import csv
import json
from gensim.models import Word2Vec

DATASET = 'action_dataset.csv'
ACTION_TEXT = 'actions.txt'
UNIQUE_ACTIONS = 'unique_actions.json'
ACTIONS_MODEL = 'actions.model'
ACTIONS_VECTORS = 'actions_vectors.json'
NONE = 'None'
SEP = ' '

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
    
    
if __name__ == '__main__':
    print 'Start...'
    #process_csv()
    create_vector_file()
    print 'Fin'



            


