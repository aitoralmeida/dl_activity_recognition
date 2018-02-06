# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 16:16:52 2016

@author: gazkune
"""

from collections import Counter
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from keras.models import Sequential, model_from_json, Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Activation, Dense, Dropout, Embedding, LSTM, Bidirectional, Convolution1D, Convolution2D, MaxPooling2D, AveragePooling2D, GlobalMaxPooling1D,GlobalMaxPooling2D, Flatten, Concatenate, Input, Reshape, TimeDistributed, Multiply, GRU

import numpy as np

from sklearn import metrics
from sklearn.metrics import confusion_matrix

# Directory of formatted datasets
INPUT_DIR = 'formatted_data/'
INPUT_BALANCED_ROOT_NAME = INPUT_DIR + 'balanced_aruba_continuous_no_t_50_10'
INPUT_ROOT_NAME = INPUT_DIR + 'aruba_continuous_no_t_50_10'


# ID for the experiment which is being run -> used to store the files with
# appropriate naming
EXPERIMENT_ID = '04'

# File name for best model weights storage
WEIGHTS_FILE = EXPERIMENT_ID + '_cnn_parallel_withattention_notime.hdf5'

#number of input actions for the model
INPUT_ACTIONS = 125
#Number of elements in the action's embbeding vector
ACTION_EMBEDDING_LENGTH = 50



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
        lgd = plt.legend(['train', 'val'], bbox_to_anchor=(1.04,1), loc="upper left")
        if save == True:
            plt.savefig(EXPERIMENT_ID + '-' + 'accuracy.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
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
        plt.legend(['train', 'val'], bbox_to_anchor=(1.04,1), loc="upper left")
        if save == True:
            plt.savefig(EXPERIMENT_ID + '-' + 'loss.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
            plt.gcf().clear()
        else:
            plt.show()
        

    
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
    
    # fix random seed for reproducibility
    np.random.seed(7)
    
    
    print 'Loading data'
    
    embedding_matrix = np.load(INPUT_ROOT_NAME + '_60_embedding_weights.npy')
    X_train = np.load(INPUT_ROOT_NAME + '_60_x_train.npy')
    X_val = np.load(INPUT_ROOT_NAME + '_60_x_val.npy')
    X_test = np.load(INPUT_ROOT_NAME + '_60_x_test.npy')
    
    y_train = np.load(INPUT_ROOT_NAME + '_60_y_train.npy')
    y_val = np.load(INPUT_ROOT_NAME + '_60_y_val.npy')
    y_test = np.load(INPUT_ROOT_NAME + '_60_y_test.npy')
    
    max_sequence_length = X_train.shape[1]
    total_activities = y_train.shape[1]
    ACTION_MAX_LENGTH = embedding_matrix.shape[1]
    
    print 'X shape:', X_train.shape
    print 'y shape:', y_train.shape
    
    print 'max sequence length:', max_sequence_length
    print 'features per action:', embedding_matrix.shape[0]
    print 'Action max length:', ACTION_MAX_LENGTH
    
    # Show the activity distribution for each set
    # We transform one-hot vector to integer codes
    y_train_code = np.array([np.argmax(y_train[x]) for x in xrange(len(y_train))])    
    y_val_code = np.array([np.argmax(y_val[x]) for x in xrange(len(y_val))])
    y_test_code = np.array([np.argmax(y_test[x]) for x in xrange(len(y_test))])
    
    print "Activity distribution for training:"
    print Counter(y_train_code)
    
    print "Activity distribution for validation:"
    print Counter(y_val_code)
    
    print "Activity distribution for testing:"
    print Counter(y_test_code)
    
    #sys.exit()
    # Build the model    
    print 'Building model...'
    sys.stdout.flush()
    batch_size = 1024

    #input pipeline
    input_actions = Input(shape=(INPUT_ACTIONS,), dtype='int32', name='input_actions')
    embedding_actions = Embedding(input_dim=embedding_matrix.shape[0], output_dim=embedding_matrix.shape[1], weights=[embedding_matrix], input_length=INPUT_ACTIONS, trainable=True, name='embedding_actions')(input_actions)        
    #attention mechanism
    gru = GRU(128, input_shape=(INPUT_ACTIONS, ACTION_EMBEDDING_LENGTH), return_sequences=True, name='gru')(embedding_actions)
    # total units = 128 * INPUT_ACTIONS
    dense_att_1 = TimeDistributed(Dense(128, activation = 'tanh',name = 'dense_att_1'))(gru)
    # total units = 1 * INPUT_ACTIONS
    dense_att_2 = TimeDistributed(Dense(1))(dense_att_1)
    # to undo the time distribution and have 1 value for each action
    reshape_distributed = Reshape((INPUT_ACTIONS,))(dense_att_2) 
    attention = Activation('softmax')(reshape_distributed)
    #so we can multiply it with embeddings
    reshape_att = Reshape((INPUT_ACTIONS, 1), name = 'reshape_att')(attention) 
    #apply the attention to the embeddings
    apply_att = Multiply()([embedding_actions, reshape_att])
    #add channel dimension for the CNNs
    reshape = Reshape((INPUT_ACTIONS, ACTION_EMBEDDING_LENGTH, 1), name = 'reshape')(apply_att) 
       
    #branching convolutions
    ngram_2 = Convolution2D(200, 2, ACTION_EMBEDDING_LENGTH, border_mode='valid',activation='relu', name = 'conv_2')(reshape)
    maxpool_2 = MaxPooling2D(pool_size=(INPUT_ACTIONS-2+1,1), name = 'pooling_2')(ngram_2)
    ngram_3 = Convolution2D(200, 3, ACTION_EMBEDDING_LENGTH, border_mode='valid',activation='relu', name = 'conv_3')(reshape)
    maxpool_3 = MaxPooling2D(pool_size=(INPUT_ACTIONS-3+1,1), name = 'pooling_3')(ngram_3)
    ngram_4 = Convolution2D(200, 4, ACTION_EMBEDDING_LENGTH, border_mode='valid',activation='relu', name = 'conv_4')(reshape)
    maxpool_4 = MaxPooling2D(pool_size=(INPUT_ACTIONS-4+1,1), name = 'pooling_4')(ngram_4)
    ngram_5 = Convolution2D(200, 5, ACTION_EMBEDDING_LENGTH, border_mode='valid',activation='relu', name = 'conv_5')(reshape)
    maxpool_5 = MaxPooling2D(pool_size=(INPUT_ACTIONS-5+1,1), name = 'pooling_5')(ngram_5)
     
    #1 branch again
    merged = Concatenate(axis=2)([maxpool_2, maxpool_3, maxpool_4, maxpool_5])
    flatten = Flatten(name = 'flatten')(merged)
#    batch_norm = BatchNormalization()(flatten)
    dense_1 = Dense(256, activation = 'relu',name = 'dense_1')(flatten)
    drop_1 = Dropout(0.8, name = 'drop_1')(dense_1)
#    dense_2 = Dense(1024, activation = 'relu',name = 'dense_2')(drop_1)
#    drop_2 = Dropout(0.8, name = 'drop_2')(dense_2)
    output_actions = Dense(total_activities, activation='softmax', name='main_output')(drop_1)
         
    model = Model(input=[input_actions], output=[output_actions])
   
    
#    model = Sequential()
#    
#    model.add(Embedding(input_dim=embedding_matrix.shape[0], output_dim=embedding_matrix.shape[1], weights=[embedding_matrix], input_length=max_sequence_length, trainable=True))
#    # Change input shape when using embeddings
#    model.add(LSTM(512, return_sequences=False, dropout_W=0.5, dropout_U=0.5, input_shape=(max_sequence_length, embedding_matrix.shape[1])))
#    #model.add(Dropout(0.5))
#    #model.add(LSTM(512, return_sequences=False, dropout_W=0.2, dropout_U=0.2))
#    #model.add(Dropout(0.8))
#    model.add(Dense(total_activities))
#    model.add(Activation('softmax'))
    
    # TODO: Test TimeDistributed(dense) in the final dense layer
    # TODO: reshape y to be 3D tensor (samples, timesteps, categories)
    """    
    # This should not be done!!
    y_train_td = np.zeros(dtype='float', shape=[y_train.shape[0], max_sequence_length, y_train.shape[1]])
    for i in xrange(len(y_train)):
        y_train_td[i][:] = y_train[i]
    y_val_td = np.zeros(dtype='float', shape=[y_val.shape[0], max_sequence_length, y_val.shape[1]])
    for i in xrange(len(y_val)):
        y_val_td[i][:] = y_val[i]
    
    y_train = y_train_td
    y_val = y_val_td
    print 'y shape (for TimeDistributed):', y_train.shape
    # TODO: Have a look at the difference between input_dim and input_shape for the LSTM layer
    # https://github.com/fchollet/keras/issues/2613
    model.add(Embedding(input_dim=embedding_matrix.shape[0], output_dim=embedding_matrix.shape[1], weights=[embedding_matrix], input_length=max_sequence_length, trainable=True))
    # Change input shape when using embeddings
    model.add(LSTM(512, return_sequences=True, dropout_W=0.5, dropout_U=0.5, input_shape=(max_sequence_length, embedding_matrix.shape[1])))    
    model.add(TimeDistributed(Dense(128, activation='relu')))
    #model.add(Flatten())
    model.add(Reshape(128*max_sequence_length))
    model.add(Dense(128, activation='relu'))
    model.add(Droput(0.5))
    model.add(Dense(total_activities, activation='softmax'))    
    """

    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'mse', 'mae'])
    print 'Model built'
    print(model.summary())
    sys.stdout.flush()
  

    print 'Training...'    
    sys.stdout.flush()
    # Define the callbacks to be used (EarlyStopping and ModelCheckpoint)
    earlystopping = EarlyStopping(monitor='val_loss', patience=100, verbose=0)    
    # TODO: improve file naming for multiple architectures
    modelcheckpoint = ModelCheckpoint(WEIGHTS_FILE, monitor='val_loss', save_best_only=True, verbose=0)
    callbacks = [earlystopping, modelcheckpoint]
    
    # Automatic training for Stateless LSTM
    manual_training = False    
    history = model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=1000, validation_data=(X_val, y_val), shuffle=True, callbacks=callbacks)
        
    # Print best val_acc and val_loss
    print 'Validation accuracy:', max(history.history['val_acc'])
    print 'Validation loss:', min(history.history['val_loss'])
    # Use the test set to calculate precision, recall and F-Measure with the bet model
    model.load_weights(WEIGHTS_FILE)
    yp = model.predict(X_test, batch_size=batch_size, verbose=1)
    # TODO: tidy up those prints
    print "Predictions on test set:"
    print "yp shape:", yp.shape
    print yp
    ypreds = np.argmax(yp, axis=1)
    print "After using argmax:"
    print ypreds
    print "y_test shape:", y_test.shape
    print y_test
    print "y_test activity indices:"
    #ytrue = np.array(y_orig[val_limit:]) # the priginal way
    ytrue = np.argmax(y_test, axis=1)
    print ytrue
    
    # Use scikit-learn metrics to calculate confusion matrix, accuracy, precision, recall and F-Measure    
    cm = confusion_matrix(ytrue, ypreds)
    
    # Normalize the confusion matrix by row (i.e by the number of samples
    # in each class)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    np.set_printoptions(precision=3, linewidth=1000, suppress=True)
    print('Confusion matrix')
    print(cm)
    # Save also the cm to a txt file
    np.savetxt(EXPERIMENT_ID+'-cm.txt', cm, fmt='%.0f')
    
    print('Normalized confusion matrix')
    print(cm_normalized)
    np.savetxt(EXPERIMENT_ID+'-cm-normalized.txt', cm_normalized, fmt='%.3f')
    
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
