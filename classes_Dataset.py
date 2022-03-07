"""
Created on Sat Apr  3 12:09:15 2021

@author: local_ergo
"""
import os
import pickle 
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split

from functions_preprocessing import sliding_window_numbers, sliding_window_image, sliding_window_2outputs

def load_multi_data(subject_n, visualization = False):
    ''' This function loads the datasets from multiple pickle files, normalize 
    them and save them in sensals and FPs variable and assign them as attributes
    of subject self.
    '''
    print('- Loading the data ...')
    FPs = []
    sensals = []
    for [n,t] in subject_n:
        sensal_filename = os.path.dirname(os.getcwd())+'/datasets/subject_'+str(n)+'_t'+str(t)+'_sensal'
        FP_filename = os.path.dirname(os.getcwd())+'/datasets/subject_'+str(n)+'_t'+str(t)+'_FP'
        sensal = pickle.load(open(sensal_filename, 'rb')); sensal = sensal['sensal']
        FP = pickle.load(open(FP_filename, 'rb'))
        FPs.append(FP)
        sensals.append(sensal)
    del FP, sensal,n, t
    if visualization == True:
        # VISUALIZATION
        print('- Visualization of the label data ...')
        linewidth = 1
        alpha = 1
        for sub in range(len(FPs)):
        # for sub in range(0,1):
            plt.figure(dpi = 200)
            plt.rcParams.update({'font.size': 6})
            plt.suptitle('Force for subject '+str(subject_n[sub][0]));
            # plt.suptitle(' Ground Reaction Forces', fontsize= 12)
            plt.subplot(311)
            plt.plot(FPs[sub][:,0],label = 'fX1',alpha = alpha,  color = 'red', linewidth = linewidth)
            plt.plot(FPs[sub][:,3], label = 'fx2',alpha = alpha,  color = 'black', linewidth = linewidth)
            plt.ylabel('F_x', fontsize= 10)
            plt.legend( bbox_to_anchor=(1, 1))
            plt.subplot(312)
            plt.plot(FPs[sub][:,1], label = 'fy1', alpha = alpha, color = 'red', linewidth=linewidth)
            plt.plot(FPs[sub][:,4], label = 'fy2', alpha = alpha, color = 'black', linewidth=linewidth)
            plt.legend( bbox_to_anchor=(1, 1))
            plt.ylabel('F_y', fontsize= 10)
            plt.subplot(313)
            plt.plot(FPs[sub][:,2], label = 'fz1', alpha = alpha, color = 'red', linewidth = linewidth)
            plt.plot(FPs[sub][:,5], label = 'fz2', alpha = alpha, color = 'black', linewidth = linewidth)
            plt.legend( bbox_to_anchor=(1, 1));
            plt.ylabel('F_z', fontsize= 10)
            plt.xlabel('data point', fontsize= 8)
            plt.show() 
        # for i in range(0,1102):
        #     im = Image.fromarray(np.uint8(sensals[0][i,:,:]/50),'L')
        #     plt.imshow(im, cmap="magma") 
        #     plt.title('A sample of the input data')
        #     plt.show()
    
    #------------ INPUT DATA NORMALIZATION TO BE BETWEEN (0,1)
    print('- Normalizing the data ...')
    for sub in range(len(subject_n)):
        max_sensal = np.amax(sensals[sub])
        sensals[sub] = sensals[sub]/max_sensal
        temp1 = np.amax(FPs[sub][:,0])
        temp2 = np.amax(FPs[sub][:,1])
        temp3 = np.amax(FPs[sub][:,3])
        temp4 = np.amax(FPs[sub][:,4])
        max_FP = max(temp1, temp2, temp3, temp4) #find the maximum value of shear force (over x and y)
        
        for i in range(FPs[sub].shape[1]):
            FPs[sub][:,i] = (FPs[sub][:,i]- np.amin(FPs[sub][:,i]))/(np.amax(FPs[sub][:,i])- np.amin(FPs[sub][:,i]))
        del temp1, temp2, temp3, temp4, i
        # print('- Visualization of the label after normalization data ...')
        # for sub in range(len(self.FPs)):
        #     plt.plot(self.FPs[sub][:,0], label='fx1')
        #     plt.plot(self.FPs[sub][:,1], label='fy1')
        #     plt.plot(self.FPs[sub][:,3], label='fx2')
        #     plt.plot(self.FPs[sub][:,4], label='fy2')
        #     plt.title('Shear forces for subject'+str(sub+1));
        #     plt.legend(loc='upper right')
        #     plt.show()
    return FPs, sensals


            


class Multi_Datasets():
    def __init__(self, model, subject_n, seq_length = 30, shift = 10, output = 'fz',
                 test_size = 0.2, val_size = 0, visualization = False):
        self.model = model
        self.subject_n = subject_n
        self.seq_length = seq_length
        
        self.FPs = []
        self.sensals = []
        print('- Loading the data ...')
        self.FPs, self.sensals = load_multi_data(subject_n, visualization = visualization)
        
        print('- Creating test and train datasets for', model,' model...')
        self.train_xs = []; self.test_xs = [] ;
        self.train_labels =[]; self.test_labels = [];
        self.val_xs = []; self.val_labels = [];
        self.chopped_train_xs = []; self.chopped_train_labels = [];
        self.chopped_test_xs = []; self.chopped_test_labels = [];
        self.chopped_val_xs = []; self.chopped_val_labels = [];
        
        shuffle = False if model in ['lrcn', 'c3d', 'convlstm2d'] else True
        
        for sub in range(len(subject_n)):
            train_x, test_x, train_output, test_label = train_test_split(self.sensals[sub], self.FPs[sub],
                                                              test_size = test_size, shuffle = shuffle)
         
            #Components of the force           
            fx1_train_label = train_output[:,0]
            fy1_train_label = train_output[:,1]                                                   
            fz1_train_label = train_output[:,2]
            fx2_train_label = train_output[:,3]
            fy2_train_label = train_output[:,4]
            fz2_train_label = train_output[:,5]
                
            fx1_test_label = test_label[:,0]
            fy1_test_label = test_label[:,1]
            fz1_test_label = test_label[:,2]
            fx2_test_label = test_label[:,3]
            fy2_test_label = test_label[:,4]
            fz2_test_label = test_label[:,5]
            del test_label
            
            if output == 'fxfy': # developing one model for fx and fy
                train_label = [fx1_train_label, fy1_train_label, fx2_train_label, fy2_train_label]
                test_label = np.transpose(np.array([fx1_test_label, fy1_test_label,
                                                fx2_test_label, fy2_test_label]))
            elif output == 'fx':
                train_label = [fx1_train_label, fx2_train_label]            
                test_label = np.transpose(np.array([fx1_test_label, fx2_test_label]))
            elif output == 'fy':
                train_label = [fy1_train_label, fy2_train_label]
                test_label = np.transpose(np.array([fy1_test_label, fy2_test_label]))
            elif output == 'fz':
                train_label = [fz1_train_label, fz2_train_label] 
                test_label = np.transpose(np.array([fz1_test_label, fz2_test_label]))
                                   
            # If we have validation set
            if val_size > 0:
    
                train_x, val_x, train_label, val_label = train_test_split(train_x, train_output,
                                                                          test_size = val_size,
                                                                          shuffle = shuffle)
                # Components of the force
                fx1_train_label = train_label[:,0]
                fy1_train_label = train_label[:,1]                                                   
                fz1_train_label = train_label[:,2]
                fx2_train_label = train_label[:,3]
                fy2_train_label = train_label[:,4]
                fz2_train_label = train_label[:,5]                                                         
                
                fx1_val_label = val_label[:,0]
                fy1_val_label = val_label[:,1]                                                   
                fz1_val_label = val_label[:,2]
                fx2_val_label = val_label[:,3]
                fy2_val_label = val_label[:,4]
                fz2_val_label = val_label[:,5]
                
                if output == 'fxfy': # developing one model for fx and fy
                    train_output = [fx1_train_label, fy1_train_label,
                                   fx2_train_label, fy2_train_label]
                    val_output = [fx1_val_label, fy1_val_label,
                                                    fx2_val_label, fy2_val_label]
                elif output == 'fx':
                    train_output = [fx1_train_label, fx2_train_label]            
                    val_output = [fx1_val_label, fx2_val_label]
                elif output == 'fy':
                    train_output= [fy1_train_label, fy2_train_label]
                    val_output = [fy1_val_label, fy2_val_label]
                elif output == 'fz':
                    train_output = [fz1_train_label, fz2_train_label] 
                    val_output = [fz1_val_label, fz2_val_label]
                
                del val_label, train_label
                train_label = np.transpose(np.array(train_output))
                val_label = np.transpose(np.array(val_output))
                self.val_xs.append(val_x)
                self.val_labels.append(val_label);
                
            else:
                train_label = np.transpose(np.array(train_label))    
                
            self.image_size = (train_x.shape[1], train_x.shape[2]) # The size of the image matrix at each frame  
            
            # Chop the data for temporal networks
            if model in ['lrcn', 'c3d', 'convlstm2d']:
#                print('- Chopping the original data to sequeces of length of ', seq_length,'...')
                assert len(train_x) >= seq_length
                
                '''Chopping Train Data'''
                n_frames = len(train_x) # number of frames
                n_windows, _ = sliding_window_numbers(n_samples = n_frames, sequence_length = seq_length, shift = shift)
                chopped_train_x = sliding_window_image(original_data = train_x, n_windows = n_windows, sequence_length = seq_length, shift = shift, plot_samples = False)
                chopped_train_label = sliding_window_2outputs(original_data = train_label, n_windows = n_windows, sequence_length = seq_length, shift = shift)
                # n_frames = len(train_x) # number of frames
                # n_windows, _ = sliding_window_numbers(n_samples = n_frames, sequence_length = seq_length, shift = shift)
                # chopped_train_x = sliding_window_image(original_data = train_x, n_windows = n_windows, sequence_length = seq_length, shift = shift, plot_samples = False)
                # chopped_train_label = sliding_window_2outputs(original_data = train_label, n_windows = n_windows, sequence_length = seq_length, shift = shift)
                ''' Chopping Test Data'''
                assert len(test_x) >= seq_length
                n_frames = len(test_x) # number of frames
                n_windows, _ = sliding_window_numbers(n_samples = n_frames, sequence_length = seq_length, shift = shift)
                chopped_test_x = sliding_window_image(original_data = test_x, n_windows = n_windows, sequence_length = seq_length, shift = shift, plot_samples = False)    
                chopped_test_label = sliding_window_2outputs(original_data = test_label, n_windows = n_windows, sequence_length = seq_length, shift = shift)
            
                if val_size > 0:
                    '''Chopping Validation Data'''
                    n_frames = len(val_x) # number of frames
                    # print("len val_x:", len(val_x))
                    assert len(val_x) >= seq_length
                    n_windows, _ = sliding_window_numbers(n_samples = n_frames, sequence_length = seq_length, shift = shift)
                    chopped_val_x = sliding_window_image(original_data = val_x, n_windows = n_windows, sequence_length = seq_length, shift = shift, plot_samples = False)    
                    chopped_val_label = sliding_window_2outputs(original_data = val_label, n_windows = n_windows, sequence_length = seq_length, shift = shift)
                    self.chopped_val_xs.append(chopped_val_x); self.chopped_val_labels.append(chopped_val_label); 
                
                self.chopped_train_xs.append(chopped_train_x); self.chopped_train_labels.append(chopped_train_label);
                self.chopped_test_xs.append(chopped_test_x); self.chopped_test_labels.append(chopped_test_label);
                del chopped_test_x, chopped_train_x, chopped_test_label, chopped_train_label
                
            self.train_xs.append(train_x); self.test_xs.append(test_x)
            self.train_labels.append(train_label); self.test_labels.append(test_label)
    
            del sub, train_x, test_x, train_output, test_label,train_label, fz1_test_label, fz2_test_label, fz1_train_label, fz2_train_label
        
        #___________ END of for loop __________________
        
        
        if val_size > 0:
            print('- This datset contains a validation dataset. To disable the validation set, assign 0 to val_size')
            self.seq_lim = min(map(len,self.val_labels)) # The maximum sequence length I can choose
        else:    
            self.seq_lim = min(map(len,self.test_labels)) # The maximum sequence length I can choose
    
        
        ''' CONCATINATING THE DATA'''
        self.train_x = np.vstack(self.train_xs); self.train_label = np.vstack(self.train_labels);
        self.test_x = np.vstack(self.test_xs); self.test_label = np.vstack(self.test_labels);
        if val_size > 0:
            self.val_x = np.vstack(self.val_xs); self.val_label = np.vstack(self.val_labels);
        if model in ['lrcn', 'c3d', 'convlstm2d']:
            self.chopped_train_x = np.vstack(self.chopped_train_xs); 
            self.chopped_train_label = np.vstack(self.chopped_train_labels);
            self.chopped_test_x = np.vstack(self.chopped_test_xs);
            self.chopped_test_label = np.vstack(self.chopped_test_labels);
            if val_size > 0:
                self.chopped_val_x = np.vstack(self.chopped_val_xs); 
                self.chopped_val_label = np.vstack(self.chopped_val_labels);
                
    

class Dataset_No_Test_Set():
    ''' This class creates just train and validation sets, no test set. It is
    used for training the model when we have a specific set for training and 
    test'''
    def __init__(self, model, subject_n, seq_length = 40, shift = 15,
                 val_size = 0.2, output = 'fxfy', visualization = False):
        self.model = model
        self.subject_n = subject_n
        self.seq_length = seq_length
        self.shift = shift
        self.val_size = val_size
        self.FPs, self.sensals = load_multi_data(subject_n, visualization = visualization)
        
        
        print('- Creating test and validation datasets for', model,' model...')
        self.train_xs = []; self.train_labels =[]; 
        self.val_xs = []; self.val_labels = [];
        self.chopped_train_xs = []; self.chopped_train_labels = [];
        self.chopped_val_xs = []; self.chopped_val_labels = [];
        
        shuffle = False if model in ['lrcn', 'c3d', 'convlstm2d'] else True
        
        for sub in range(len(subject_n)):    
            train_x = self.sensals[sub]
            # Components of the force                                                              
            fx1_train_label = self.FPs[sub][:,0]
            fy1_train_label = self.FPs[sub][:,1]
            fz1_train_label = self.FPs[sub][:,2]
            fx2_train_label = self.FPs[sub][:,3]
            fy2_train_label = self.FPs[sub][:,4]
            fz2_train_label = self.FPs[sub][:,5]
            train_output = []
            
            
            if output == 'fxfy': # developing one model for fx and fy
                train_output.append(fx1_train_label);train_output.append(fy1_train_label)
                train_output.append(fx2_train_label);train_output.append(fy2_train_label)

            elif output == 'fz':
                train_output.append(fz1_train_label);train_output.append(fz2_train_label);

            elif output == 'fx':
                train_output.append(fx1_train_label);train_output.append(fx2_train_label);
 
            elif output == 'fy':
                train_output.append(fy1_train_label);train_output.append(fy2_train_label);
  
            
            train_label = np.transpose(np.array(train_output))
            
            self.image_size = (train_x.shape[1], train_x.shape[2]) # The size of the image matrix at each frame  
            
            # Chop the data for temporal networks
            if model in ['lrcn', 'c3d', 'convlstm2d']:
                assert len(train_x) >= seq_length
                
                '''Chopping Train Data'''
                n_frames = len(train_x) # number of frames
                n_windows, _ = sliding_window_numbers(n_samples = n_frames, sequence_length = seq_length, shift = shift)
                chopped_train_x = sliding_window_image(original_data = train_x, n_windows = n_windows, sequence_length = seq_length, shift = shift)
                chopped_train_label = sliding_window_2outputs(original_data = train_label, n_windows = n_windows, sequence_length = seq_length, shift = shift)
            
                self.chopped_train_xs.append(chopped_train_x); self.chopped_train_labels.append(chopped_train_label);
                
                del chopped_train_x, chopped_train_label
                
            self.train_xs.append(train_x); 
            self.train_labels.append(train_label);
            
            del sub, train_x, train_output
        
        #___________ END of for loop __________________
        
        self.seq_lim = min(map(len,self.train_labels)) # The maximum sequence length I can choose
        
        ''' CONCATINATING THE DATA'''
        self.train_x = np.vstack(self.train_xs); self.train_label = np.vstack(self.train_labels);
        if model in ['lrcn', 'c3d', 'convlstm2d']:
            self.chopped_train_x = np.vstack(self.chopped_train_xs); 
            self.chopped_train_label = np.vstack(self.chopped_train_labels);        
      