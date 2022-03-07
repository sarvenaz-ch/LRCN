# -*- coding: utf-8 -*-
"""
This file contains functions that work with the object oriented format of the 
DL algorithm
"""
import os
import numpy as np
import random
import matplotlib.pyplot as plt
from livelossplot import PlotLossesKeras
from keras import callbacks
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from datetime import datetime
def train(model, train_x, train_label, epochs, val_x = [], val_label = [],
          verbose = 1, train_plot = False, save_best = False, filepath = []):
    tb = callbacks.TensorBoard(histogram_freq=1, write_grads = True)
    print('- Training model...' )
    if save_best == False:
        if val_x == []:
            history = model.fit(train_x, train_label, verbose = verbose, epochs=epochs).history
        else:
            history = model.fit(train_x, train_label, validation_data = (val_x, val_label),
                                verbose = verbose, epochs=epochs, callbacks=[PlotLossesKeras()]).history
            # history = model.fit(train_x, train_label, validation_data = (val_x, val_label),
            #                     verbose = verbose, epochs=epochs).history
    else: # save the best trained model at each epoch
        if filepath == []:
            filepath = os.getcwd()+'\\model\\'+datetime.now().strftime("%Y-%m-%d_%H-%M")+'\\'+datetime.now().strftime("%Y-%m-%d_%H-%M")+'_best_model.epoch{epoch:02d}-loss{val_loss:.3f}.hdf5'
        checkpoint = callbacks.ModelCheckpoint(filepath=filepath, monitor='val_loss',
                             verbose=1, save_best_only=True, mode='min')
        if val_x == []:
            history = model.fit(train_x, train_label, verbose = verbose, epochs=epochs, callbacks=[checkpoint]).history
            
        else:
            history = model.fit(train_x, train_label, validation_data = (val_x, val_label),
                                verbose = verbose, epochs=epochs, callbacks=[checkpoint]).history

        
    # plt_loss = plt.plot(history['loss'][2:], label = 'overall loss')
    if train_plot == True:
        plt_fz1 = plt.plot(history['fz1_loss'][80:], label = 'fz1 loss')
        plt_fz2 = plt.plot(history['fz2_loss'][80:], label = 'fz2 loss')
        # plt_fz1 = plt.plot(history_bn['fz1_loss'][2:], label = 'fz1 loss after activation')
        # plt_fz2 = plt.plot(history_bn['fz2_loss'][2:], label = 'fz2 loss after activation')
        plt.legend(loc='upper right'); plt.title('Training loss')
        plt.show()
        # history, results = train_crnn_model(train_x = chopped_train_x, outputs_train = chopped_train_label,
        #                   num_outputs = 2, model = crnn_model, n_splits = n_splits,
        #                 epochs = epochs, callback_file_path = callback_file_path)
    return history
 
    
def train_backtest(model, train_x, train_label, epochs, k_fold = 5,
                   verbose = 1, train_plot = False, filepath=[]):
    ''' This function trains a temporal network based on k-fold algorithm. It 
    saves the best network up untill anypoint based on the validation loss
    in the filepath'''
    if k_fold<2:
        raise ValueError(" train_backtest(): The number of folds should be greater than 2")
        
    batch_size = len(train_x)
    history = []

    if filepath == []:
        filepath = os.getcwd()+'/model/'+datetime.now().strftime("%Y-%m-%d_%H-%M")+'_best_model.epoch{epoch:02d}-loss{val_loss:.3f}.hdf5'

    for fold in range(0,k_fold):
        train_ind, val_ind = split_index(k_fold, batch_size)
    
        train_input =  train_x[train_ind]
        train_output = train_label[train_ind]
        val_input = train_x[val_ind]
        val_output = train_label[val_ind]
        model.model.load_weights('model.h5')
        history.append(train(model = model.model, train_x = train_input, train_label = train_output,
                         epochs = epochs, val_x = val_input, val_label = val_output,
                         save_best = True, verbose = 1, train_plot = False, filepath=filepath))
        
    return history
       
def train_k_fold(model, train_x, train_label, epochs, k_fold = 5,
                   verbose = 1, train_plot = False, filepath=[]):
    if filepath == []:
        filepath = os.getcwd()+'/model/'+datetime.now().strftime("%Y-%m-%d_%H-%M")+'_best_model.epoch{epoch:02d}-loss{val_loss:.3f}.hdf5'
    history = []
    kf = KFold(n_splits = k_fold)
    for train_ind, val_ind in kf.split(train_x):
        train_input, val_input = train_x[train_ind], train_x[val_ind]
        train_output, val_output = train_label[train_ind], train_label[val_ind]
        model.model.load_weights('model.h5')
        history.append(train(model = model.model, train_x = train_input, train_label = train_output,
                         epochs = epochs, val_x = val_input, val_label = val_output,
                         save_best = True, verbose = 1, train_plot = False, filepath=filepath))



def split_index(k_fold, batch_size):
    # Choosing indecis of batches of data for validation set
    for fold in range(0, k_fold):
        # partition the training set indecis into validation and train set
        val_ind = random.sample(list(np.linspace(0, batch_size-1, batch_size)), int(batch_size/k_fold))
        for i in range(0, len(val_ind)): val_ind[i] = int(val_ind[i]) # converting the list values to integers
        train_ind = []
        for ind in range(0, batch_size):
            if ind not in val_ind:
                train_ind.append(ind)
                
    return train_ind, val_ind        
    
