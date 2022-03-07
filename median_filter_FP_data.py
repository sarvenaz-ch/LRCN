# -*- coding: utf-8 -*-
"""
This file applies median filter to FP data and plots the filtered data against 
raw data

@author:Sarvenaz
"""
import os
import pickle 
import numpy as np
from scipy.signal import medfilt, medfilt2d
import matplotlib.pyplot as plt
from PIL import Image
from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.model_selection import train_test_split
from functions_preprocessing import sliding_window_numbers, sliding_window_image, sliding_window_2outputs

subject_n = [[1,1],[4,1],[7,1],[8,1],[8,2],[10,1],[10,2],[11,1], [11,2],
             [12,1],[13,1],[13,2],[15,1],[16,1],[17,1],[18,1],[19,1],[19,2]] #subject_n = [[subject number,trial number]]
# subject_n = [[13,2]]
model = 'cnn'
test_size = 0.2
val_size = 0

FPs = []
sensals = []
print('- Loading the data ...')
for [n,t] in subject_n:

    # sensal_filename = os.path.dirname(os.getcwd())+'\\datasets\\subject_'+str(n)+'_t'+str(t)+'_sensal'
    FP_filename = os.path.dirname(os.getcwd())+'\\datasets\\subject_'+str(n)+'_t'+str(t)+'_FP'
    # sensal = pickle.load(open(sensal_filename, 'rb')); sensal = sensal['sensal']
    FP = pickle.load(open(FP_filename, 'rb'))
    FPs.append(FP)
    # sensals.append(sensal)
# del FP, sensal,n, t

FP_filt = []
for sub in range(len(FPs)):
    FP_filt.append([])
    for ax in range(FPs[sub].shape[1]):
        FP_filt[sub].append(medfilt(volume = FPs[sub][:,ax],kernel_size = 9))
# for ax in range(FPs[0].shape[1]):
#     FPs[0][:,ax] = medfilt(volume = FPs[0][:,ax],kernel_size = 59)
    
#----------------- VISUALIZATION
# print('- Visualization of the training data ...')\
    
    

linewidth = 1
alpha = 0.3
# for sub in range(len(FPs)):
for sub in range(2,4):
    print(('Force for subject '+str(subject_n[sub][0])))
    plt.figure(dpi = 200)
    plt.rcParams.update({'font.size': 6})
    plt.suptitle('Force for subject '+str(subject_n[sub][0]));
    plt.subplot(311)
    # plt.plot(FPs[sub][:,0],label = 'fX1',alpha = alpha,  color = 'blue', linewidth = linewidth)
    plt.plot(FP_filt[sub][0][500:],label = 'fx1', color = 'blue', linewidth = linewidth)
    # plt.plot(FPs[sub][:,3], label = 'fx2',alpha = alpha,  color = 'red', linewidth = linewidth)
    plt.plot(FP_filt[sub][3][500:], label = 'fx2', color = 'red', linewidth = linewidth)
    plt.legend( bbox_to_anchor=(1, 1))
    
    plt.subplot(312)
    # plt.plot(FPs[sub][:,1], label = 'fy1', alpha = alpha, color = 'blue', linewidth=linewidth)
    plt.plot(FP_filt[sub][1][500:], label = 'fy1', color = 'blue', linewidth=linewidth)
    # plt.plot(FPs[sub][:,4], label = 'fy2', alpha = alpha, color = 'red', linewidth=linewidth)
    plt.plot(FP_filt[sub][4][500:], label = 'fy2', color = 'red', linewidth=linewidth)
    # plt.title('Force for subject'+str(sub+1));
    plt.ylabel('Force (N)', fontsize = 'xx-large')
    plt.legend( bbox_to_anchor=(1, 1))
    
    plt.subplot(313)
    # plt.plot(FPs[sub][:,2], label = 'fz1', alpha = alpha, color = 'blue', linewidth = linewidth)
    plt.plot(FP_filt[sub][2][500:], label = 'fz1', color = 'blue', linewidth = linewidth)
    # plt.plot(FPs[sub][:,5], label = 'fz2', alpha = alpha, color = 'red', linewidth = linewidth)
    plt.plot(FP_filt[sub][5][500:], label = 'fz2', color = 'red', linewidth = linewidth)
    plt.xlabel('data point', fontsize = 'xx-large'); 
    # plt.title('Force for subject'+str(sub+1));
    print(' -------------- MIN & MAX --------------')
    print('- X: fp1[',min(FP_filt[sub][0][:]), max(FP_filt[sub][0][:]), ']fp2:[',min(FP_filt[sub][3][:]), max(FP_filt[sub][3][:]))
    print('- Y: fp1[',min(FP_filt[sub][1][:]), max(FP_filt[sub][1][:]), ']fp2:[',min(FP_filt[sub][4][:]), max(FP_filt[sub][4][:]))
    print('- Z: fp1[',min(FP_filt[sub][2][:]), max(FP_filt[sub][2][:]), ']fp2:[',min(FP_filt[sub][5][:]), max(FP_filt[sub][5][:]))
    print(' -------------- MEAN & STDEV --------------')
    print('- X: fp1[',np.mean(FP_filt[sub][0][:]), np.std(FP_filt[sub][0][:]), ']fp2:[',np.mean(FP_filt[sub][3][:]), np.std(FP_filt[sub][3][:]))
    print('- Y: fp1[',np.mean(FP_filt[sub][1][:]), np.std(FP_filt[sub][1][:]), ']fp2:[',np.mean(FP_filt[sub][4][:]), np.std(FP_filt[sub][4][:]))
    print('- Z: fp1[',np.mean(FP_filt[sub][2][:]), np.std(FP_filt[sub][2][:]), ']fp2:[',np.mean(FP_filt[sub][5][:]), np.std(FP_filt[sub][5][:]))
    plt.legend( bbox_to_anchor=(1, 1));
    plt.show()

# ''' RELACING THE FORCE DATA OF SPECIFIC SUBJETCTS WITH THE FILTERED ONE'''
# output_FP_filename = os.path.dirname(os.getcwd())+'\\datasets\\subject_0_t0_FP'
# FPs[0][:] = np.transpose(np.array(FP_filt[0]))
# sub = 0
# plt.figure(dpi = 200)
# plt.rcParams.update({'font.size': 6})
# plt.suptitle('Force for subject '+str(subject_n[sub][0]));
# plt.subplot(311)
# plt.plot(FPs[sub][:,0],label = 'fX1',alpha = alpha,  color = 'blue', linewidth = linewidth)
# plt.plot(FP_filt[sub][0][:],'--', label = 'fx1_filt', color = 'blue', linewidth = linewidth)
# plt.plot(FPs[sub][:,3], label = 'fx2',alpha = alpha,  color = 'red', linewidth = linewidth)
# plt.plot(FP_filt[sub][3][:], '--', label = 'fx2_filt', color = 'red', linewidth = linewidth)
# plt.legend( bbox_to_anchor=(1, 1))
# plt.subplot(312)
# plt.plot(FPs[sub][:,1], label = 'fy1', alpha = alpha, color = 'blue', linewidth=linewidth)
# plt.plot(FP_filt[sub][1][:],'--', label = 'fy1_filt', color = 'blue', linewidth=linewidth)
# plt.plot(FPs[sub][:,4], label = 'fy2', alpha = alpha, color = 'red', linewidth=linewidth)
# plt.plot(FP_filt[sub][4][:],'--', label = 'fy2_filt', color = 'red', linewidth=linewidth)
# # plt.title('Force for subject'+str(sub+1));
# plt.legend( bbox_to_anchor=(1, 1))
# plt.subplot(313)
# plt.plot(FPs[sub][:,2], label = 'fz1', alpha = alpha, color = 'blue', linewidth = linewidth)
# plt.plot(FP_filt[sub][2][:],'--', label = 'fZ1_filt', color = 'blue', linewidth = linewidth)
# plt.plot(FPs[sub][:,5], label = 'fz2', alpha = alpha, color = 'red', linewidth = linewidth)
# plt.plot(FP_filt[sub][5][:],'--', label = 'fZ2_filt', color = 'red', linewidth = linewidth)
# # plt.title('Force for subject'+str(sub+1));
# plt.legend( bbox_to_anchor=(1, 1));
# plt.show()

# pickle.dump(FPs, open(output_FP_filename,'wb'))
# print('- The data is saved in pickle format.')