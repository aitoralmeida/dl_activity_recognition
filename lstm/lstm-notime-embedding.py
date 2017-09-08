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
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Activation, Embedding, Input, Dropout
from keras.layers import LSTM
from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

from gensim.models import Word2Vec

import numpy as np

from sklearn import metrics
from sklearn.metrics import confusion_matrix

# Directory of datasets
DIR = '../sensor2vec/casas_aruba_dataset/'
# Choose the specific dataset
#DATASET_CSV = DIR + 'aruba_complete_numeric.csv'
DATASET_CSV = DIR + 'aruba_no_t.csv'
#DATASET_CSV = DIR + 'aruba_no_t_testsplit.csv'

# TODO: Action vectors -> Ask Aitor!!
# ACTION_VECTORS = DIR + 'action2vec/actions_vectors.json'
# Word2Vec model
#WORD2VEC_MODEL = DIR + 'action2vec/continuous_complete_numeric_200_10.model' # d=200, win=10
WORD2VEC_MODEL = DIR + 'action2vec/continuous_no_t_50_10.model' # d=50, win=10

# Maximun number of actions in an activity
#ACTIVITY_MAX_LENGTH = 32 # Extract from the dataset itself

# Number of dimensions of an action vector
#ACTION_MAX_LENGTH = 200 # Make coherent with selected WORD2VEC_MODEL
ACTION_MAX_LENGTH = 50 # Make coherent with selected WORD2VEC_MODEL





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
Function which implements the data framing to use an embedding layer
Input:
    df -> Pandas DataFrame with timestamp, action and activity
    activity_to_int -> dict with the mappings between activities and integer indices
    delta -> integer to control the segmentation of actions for sequence generation
Output:
    X -> array with action index sequences
    y -> array with activity labels as integers
    tokenizer -> instance of Tokenizer class used for action/index convertion

"""
def prepare_embeddings(df, activity_to_int, delta = 0):
    # Numpy array with all the actions of the dataset
    actions = df['action'].values
    print "prepare_embeddings: actions length:", len(actions)
    
    # Use tokenizer to generate indices for every action
    # Very important to put lower=False, since the Word2Vec model
    # has the action names with some capital letters
    # Very important to remove '.' and '_' from filters, since they are used
    # in action names (T003_21.5)
    tokenizer = Tokenizer(lower=False, filters='!"#$%&()*+,-/:;<=>?@[\\]^`{|}~\t\n')
    tokenizer.fit_on_texts(actions)
    action_index = tokenizer.word_index
    print "prepare_embeddings: action_index:"
    print action_index.keys()
    
    # Build new list with action indices    
    trans_actions = np.zeros(len(actions))
    for i in xrange(len(actions)):
        #print "prepare_embeddings: action:", actions[i]        
        trans_actions[i] = action_index[actions[i]]

    #print trans_actions
    X = []
    y = []
    # Depending on delta, we generate sequences in different ways
    if delta == 0:
        # Each sequence is composed by the actions of that
        # activity instance
        current_activity = ""
        actionsdf = []
        aux_actions = []
        i = 0
        ACTIVITY_MAX_LENGTH = 0
        for index in df.index:        
            if current_activity == "":
                current_activity = df.loc[index, 'activity']
            
        
            if current_activity != df.loc[index, 'activity']:
                y.append(activity_to_int[current_activity])
                X.append(actionsdf)            
                #print current_activity, aux_actions
                current_activity = df.loc[index, 'activity']
                # reset auxiliary variables
                actionsdf = []
                aux_actions = []
        
            #print 'Current action: ', action
            actionsdf.append(np.array(trans_actions[i]))
            aux_actions.append(trans_actions[i])
            i = i + 1
        
        # Append the last activity
        y.append(activity_to_int[current_activity])
        X.append(actionsdf)
        if len(actionsdf) > ACTIVITY_MAX_LENGTH:
            ACTIVITY_MAX_LENGTH = len(actionsdf)
    else:
        # TODO: use delta value as the time slice for action segmentation
        # as Kasteren et al.
        print 'prepare_embeddings: delta value =', delta
        
        current_index = df.index[0]
        last_index = df.index[len(df) - 1]
        i = 0
        DYNAMIC_MAX_LENGTH = 0
        while current_index < last_index:
            current_time = df.loc[current_index, 'timestamp']
            #print 'prepare_embeddings: inside while', i
            #print 'prepare_embeddings: current time', current_time
            i = i + 1            
            
            """
            if i % 10 == 0:
                print '.',
            """
            actionsdf = []
            
            #auxdf = df.iloc[np.logical_and(df.index >= current_index, df.index < current_index + pd.DateOffset(seconds=delta))]
            auxdf = df.loc[np.logical_and(df.timestamp >= current_time, df.timestamp < current_time + pd.DateOffset(seconds=delta))]
            
            #print 'auxdf'
            #print auxdf
                        
            #first = df.index.get_loc(auxdf.index[0])
            first = auxdf.index[0]
            #last = df.index.get_loc(auxdf.index[len(auxdf)-1])
            last = auxdf.index[len(auxdf)-1]
            #print 'First:', first, 'Last:', last
            #actionsdf.append(np.array(trans_actions[first:last]))
            
            if first == last:
                actionsdf.append(np.array(trans_actions[first]))
            else:
                for j in xrange(first, last+1):            
                    actionsdf.append(np.array(trans_actions[j]))
            
            if len(actionsdf) > DYNAMIC_MAX_LENGTH:
                print " "
                DYNAMIC_MAX_LENGTH = len(actionsdf)
                print "MAX LENGTH =", DYNAMIC_MAX_LENGTH
                print 'First:', auxdf.loc[first, 'timestamp'], 'Last:', auxdf.loc[last, 'timestamp']
                print 'first index:', first, 'last index:', last
                print 'Length:', len(auxdf)
                #print auxdf
                #print actionsdf
                
                
            X.append(actionsdf)
            # Find the dominant activity in the time slice of auxdf
            activity = auxdf['activity'].value_counts().idxmax()
            y.append(activity_to_int[activity])
            
            # Update current_index            
            #pos = df.index.get_loc(auxdf.index[len(auxdf)-1])
            #current_index = df.index[pos+1]
            if last < last_index:
                current_index = last + 1
            else:
                current_index = last_index
            
                

    # Pad sequences
    max_sequence_length = 0
    if delta != 0:
        X = pad_sequences(X, maxlen=DYNAMIC_MAX_LENGTH, dtype='float32')
        max_sequence_length = DYNAMIC_MAX_LENGTH
    else:            
        X = pad_sequences(X, maxlen=ACTIVITY_MAX_LENGTH, dtype='float32')
        max_sequence_length = ACTIVITY_MAX_LENGTH
    
    return X, y, tokenizer, max_sequence_length

# Function to create the embedding matrix, which will be used to initialize
# the embedding layer of the network
def create_embedding_matrix(tokenizer):
    model = Word2Vec.load(WORD2VEC_MODEL)    
    action_index = tokenizer.word_index
    embedding_matrix = np.zeros((len(action_index) + 1, ACTION_MAX_LENGTH))
    unknown_words = {}    
    for action, i in action_index.items():
        try:            
            embedding_vector = model[action]
            embedding_matrix[i] = embedding_vector            
        except Exception as e:
            #print type(e) exceptions.KeyError
            if action in unknown_words:
                unknown_words[action] += 1
            else:
                unknown_words[action] = 1
    print "Number of unknown tokens: " + str(len(unknown_words))
    print unknown_words
    
    return embedding_matrix
    
    
def calculate_evaluation_metrics(y_gt, y_preds):
    
    """Calculates the evaluation metrics (precision, recall and F1) for the
    predicted examples. It calculates the micro, macro and weighted values
    of each metric.
            
    Usage example:
        y_gt = ['make_coffe', 'brush_teeth', 'wash_hands']
        y_preds = ['make_coffe', 'wash_hands', 'wash_hands']
        metrics = calculate_evaluation_metrics (y_ground_truth, y_predicted)
        
    Parameters
    ----------
        y_gt : array, shape = [n_samples]
            Classes that appear in the ground truth.
        
        y_preds: array, shape = [n_samples]
            Predicted classes. Take into account that the must follow the same
            order as in y_ground_truth
           
    Returns
    -------
        metric_results : dict
            Dictionary with the values for the metrics (precision, recall and 
            f1)    
    """
        
    metric_types =  ['micro', 'macro', 'weighted']
    metric_results = {
        'precision' : {},
        'recall' : {},
        'f1' : {},
        'acc' : -1.0        
    }
            
    for t in metric_types:
        metric_results['precision'][t] = metrics.precision_score(y_gt, y_preds, average = t)
        metric_results['recall'][t] = metrics.recall_score(y_gt, y_preds, average = t)
        metric_results['f1'][t] = metrics.f1_score(y_gt, y_preds, average = t)
        metric_results['acc'] = metrics.accuracy_score(y_gt, y_preds) 
                
    return metric_results

# Main function
def main(argv):
    
    # Load dataset from csv file
    df_dataset = pd.read_csv(DATASET_CSV, parse_dates=[0], header=None)
    df_dataset.columns = ["timestamp", 'action', 'activity']    
    
    #df = df_dataset[0:1000] # reduce dataset for tests
    df = df_dataset # complete dataset
    unique_activities = df['activity'].unique()
    print "Unique activities:"
    print unique_activities

    total_activities = len(unique_activities)
    #action_vectors = json.load(open(ACTION_VECTORS, 'r'))
    
    # Generate the dict to transform activities to integer numbers
    activity_to_int = dict((c, i) for i, c in enumerate(unique_activities))
    # Generate the dict to transform integer numbers to activities
    int_to_activity = dict((i, c) for i, c in enumerate(unique_activities))

        
    # Prepare sequences using action indices
    # Each action will be an index which will point to an action vector
    # in the weights matrix of the Embedding layer of the network input
    # Use 'delta' to establish slicing time; if 0, slicing done on activity type basis
    delta = 60 # To test the same time slicing as Kasteren (60)
    X, y, tokenizer, max_sequence_length = prepare_embeddings(df, activity_to_int, delta=delta)
    """
    for i in range(10):
        print '-----------------------'
        print 'Activity', i
        print X[i], '->', int_to_activity[y[i]]
    """
    # Create the embedding matrix for the embedding layer initialization
    embedding_matrix = create_embedding_matrix(tokenizer)
    #print "Embedding matrix"
    #print embedding_matrix
    
    # Keep original y (with activity indices) before transforming it to categorical
    y_orig = deepcopy(y)
    # Tranform class labels to one-hot encoding
    y = np_utils.to_categorical(y)
    
    
    # Prepare training and testing datasets
    # Insert code to split data into train, validation and test
    total_examples = len(X)
    train_per = 0.6
    val_per = 0.2
    # test_per = 0.2 # Not needed
    
    train_limit = int(train_per * total_examples)
    val_limit = train_limit + int(val_per * total_examples)    
    X_train = X[0:train_limit]
    X_val = X[train_limit:val_limit]
    X_test = X[val_limit:]
    y_train = y[0:train_limit]
    y_val = y[train_limit:val_limit]
    y_test = y[val_limit:]
    print 'Max sequence length:', max_sequence_length
    print 'Total examples:', total_examples
    print 'Train examples:', len(X_train), len(y_train) 
    print 'Validation examples:', len(X_val), len(y_val)
    print 'Test examples:', len(X_test), len(y_test)
    sys.stdout.flush()  
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    print 'Activity distribution for training:'
    check_activity_distribution(y_train, unique_activities)

    X_val = np.array(X_val)
    y_val = np.array(y_val)
    print 'Activity distribution for validation:'
    check_activity_distribution(y_val, unique_activities)
    
    X_test = np.array(X_test)
    y_test = np.array(y_test)    

    print 'Activity distribution for testing:'
    check_activity_distribution(y_test, unique_activities)

    # Reshape for using with static action vectors
    #X = X.reshape(X.shape[0], ACTIVITY_MAX_LENGTH, ACTION_MAX_LENGTH)
    #X_test = X_test.reshape(X_test.shape[0], ACTIVITY_MAX_LENGTH, ACTION_MAX_LENGTH)
    # If we are using an Embedding layer, there is no need to reshape
    print 'Shape (X,y):'
    print X_train.shape
    print y_train.shape
    print 'Training set prepared'  
    sys.stdout.flush()   

    # Build the model    
    print 'Building model...'
    sys.stdout.flush()
    batch_size = 16
    model = Sequential()
    #model.add(Input(shape=(ACTIVITY_MAX_LENGTH,), dtype='int32'))
    model.add(Embedding(input_dim=embedding_matrix.shape[0], output_dim=embedding_matrix.shape[1], weights=[embedding_matrix], input_length=max_sequence_length, trainable=True))
    # Change input shape when using embeddings
    model.add(LSTM(512, return_sequences=False, dropout_W=0.2, dropout_U=0.2, input_shape=(max_sequence_length, ACTION_MAX_LENGTH)))
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
    # Define the callbacks to be used (EarlyStopping and ModelCheckpoint)
    earlystopping = EarlyStopping(monitor='val_loss', patience=2, verbose=0)    
    weights = 'lstm-notime-embedding-weights.hdf5' # TODO: improve file naming for multiple architectures
    modelcheckpoint = ModelCheckpoint(weights, monitor='val_loss', save_best_only=True, verbose=0)
    callbacks = [earlystopping, modelcheckpoint]
    
    # Automatic training for Stateless LSTM
    manual_training = False    
    history = model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=100, validation_data=(X_val, y_val), shuffle=False, callbacks=callbacks)
        
    # Use the test set to calculate precision, recall and F-Measure with the bet model
    model.load_weights(weights)
    yp = model.predict(X_test, batch_size=1, verbose=1)
    print "Predictions on test set:"
    print "yp shape:", yp.shape
    print yp
    ypreds = np.argmax(yp, axis=1)
    print "After using argmax:"
    print ypreds
    print "y_test shape:", y_test.shape
    print y_test
    print "y_test activity indices:"
    ytrue = np.array(y_orig[val_limit:])
    print ytrue
    
    # Use scikit-learn metrics to calculate confusion matrix, accuracy, precision, recall and F-Measure
    # TODO: test with the whole dataset
    cm = confusion_matrix(ytrue, ypreds)
    
    # Normalize the confusion matrix by row (i.e by the number of samples
    # in each class)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    np.set_printoptions(precision=3, linewidth=1000)
    print('Confusion matrix')
    print(cm)
    
    print('Normalized confusion matrix')
    print(cm_normalized)
    
    #Dictionary with the values for the metrics (precision, recall and f1)    
    metrics = calculate_evaluation_metrics(ytrue, ypreds)
    print "Scikit metrics"
    print 'accuracy: ', metrics['acc']
    print 'precision:', metrics['precision']
    print 'recall:', metrics['recall']
    print 'f1:', metrics['f1']    
    
    #print 'Saving model...'
    #sys.stdout.flush()
    #save_model(model)
    #print 'Model saved'
    
    if manual_training == True:
        plot_training_info(['accuracy', 'loss'], True, history)
    else:
        plot_training_info(['accuracy', 'loss'], True, history.history)
    

if __name__ == "__main__":
   main(sys.argv)
