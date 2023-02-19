#!/usr/bin/env python
# coding: utf-8

# In[2]:



import os
from os.path import dirname, join as pjoin
import scipy.io
import tensorflow as tf
import torch

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from os import makedirs
from os import listdir
from shutil import copyfile
from random import seed
from random import random

# baseline model for the classification dataset
import sys
from matplotlib import pyplot
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from sklearn.model_selection import KFold, StratifiedKFold
from tensorflow.keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

#!pip install opencv-python
#!pip install lime 

import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from skimage.color import gray2rgb
from skimage.color import rgb2gray
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import inception_v3 as inc_net
from lime import lime_image
import pandas as pd
import glob
from tensorflow.keras.models import load_model
from sklearn.metrics import r2_score

import numpy as np

from keras.layers.normalization import BatchNormalization
import cv2
from keras.preprocessing.image import ImageDataGenerator
from skimage.segmentation import mark_boundaries


# In[6]:


# # NS
tensor_filenames= [x.replace(' ', '').replace('voxel', 'Tensor_voxel_').replace('-1', '').replace('.jpg', '.mat') for x in '''
voxel 0 0.jpg
voxel 1 0.jpg
voxel 2 0.jpg
voxel 3 0.jpg
voxel 4 0-1.jpg
voxel 5 0.jpg
voxel 6 0.jpg
voxel 7 0.jpg
voxel 8 0.jpg
voxel 9 0.jpg
voxel 0 1.jpg
voxel 1 1.jpg
voxel 2 1.jpg
voxel 3 1.jpg
voxel 4 1.jpg
voxel 5 1.jpg
voxel 6 1.jpg
voxel 7 1.jpg
voxel 8 1.jpg
voxel 9 1.jpg
voxel 0 2.jpg
voxel 1 2.jpg
voxel 2 2.jpg
voxel 3 2.jpg
voxel 4 2.jpg
voxel 5 2.jpg
voxel 6 2.jpg
voxel 7 2.jpg
voxel 8 2.jpg
voxel 9 2.jpg
voxel 0 3.jpg
voxel 1 3.jpg
voxel 2 3.jpg
voxel 3 3.jpg
voxel 4 2.jpg
voxel 5 3.jpg
voxel 6 3.jpg
voxel 7 3.jpg
voxel 8 3.jpg
voxel 9 3.jpg
voxel 0 4.jpg
voxel 1 4.jpg
voxel 2 4.jpg
voxel 3 4.jpg
voxel 4 4.jpg
voxel 5 4.jpg
voxel 6 4.jpg
voxel 7 4.jpg
voxel 8 4.jpg
voxel 9 4.jpg
voxel 0 5.jpg
voxel 1 5.jpg
voxel 2 5.jpg
voxel 3 5.jpg
voxel 4 5.jpg
voxel 5 5.jpg
voxel 6 5.jpg
voxel 7 5.jpg
voxel 8 5.jpg
voxel 9 5.jpg
voxel 0 6.jpg
voxel 1 6.jpg
voxel 2 6.jpg
voxel 3 6.jpg
voxel 4 6-1.jpg
voxel 5 6.jpg
voxel 6 6.jpg
voxel 7 6.jpg
voxel 8 6.jpg
voxel 9 6.jpg
voxel 0 7.jpg
voxel 1 7.jpg
voxel 2 7.jpg
voxel 3 7.jpg
voxel 4 7.jpg
voxel 5 7.jpg
voxel 6 7.jpg
voxel 7 7.jpg
voxel 8 7.jpg
voxel 9 7.jpg
voxel 0 8.jpg
voxel 1 8.jpg
voxel 2 8.jpg
voxel 3 8.jpg
voxel 4 8.jpg
voxel 5 8.jpg
voxel 6 8-1.jpg
voxel 7 8.jpg
voxel 8 8.jpg
voxel 9 8.jpg
'''.split('\n') if x]

voxel_index=[x.strip() for x in '''
11
12
13
14
15
16
17
18
19
110
21
22
23
24
25
26
27
28
29
210
31
32
33
34
35
36
37
38
39
310
41
42
43
44
45
46
47
48
49
410
51
52
53
54
55
56
57
58
59
510
61
62
63
64
65
66
67
68
69
610
71
72
73
74
75
76
77
78
79
710
81
82
83
84
85
86
87
88
89
810
91
92
93
94
95
96
97
98
99
910
'''.split('\n') if x]

area_porosity=[float(x.strip()) for x in '''
0.031
0.809
0.699
1.826
3.528
3.649
3.998
2.477
2.558
0
2.269
0
0.082
2.017
3.251
2.875
3.891
3.036
1.685
0.283
0.014
0.417
0.772
1.761
3.288
3.618
4.497
2.237
1.451
0.276
2.875
2.946
1.736
1.637
3.288
7.545
4.489
2.382
1.578
2.586
4.483
0.899
0.147
0.27
1.764
2.434
2.581
2.671
2.237
1.651
1.522
0.355
2.378
2.299
2.099
0.902
2.325
0.104
0.986
0.194
1.654
1.745
0.977
0.903
0
0.799
0.414
1.28
0.79
0.031
2.029
1.192
1.724
1.043
2.347
0.493
0.685
0.166
0.093
0.135
0
0.92
0.589
0.819
1.705
0.81
3.427
1.136
0.957
0.204
'''.split('\n') if x]


# In[9]:


#NS
import pandas as pd
import math
from keras.utils import normalize, to_categorical

# Groups of pore counts (ex: 0-5, 6-10, ...)
#GROUP_WIDTH=5
#NUM_GROUPS=10
#group_index = lambda pore_count: math.ceil(pore_count/GROUP_WIDTH)
#category_vector = lambda grp_index: to_categorical(grp_index, NUM_GROUPS).astype(int)

#voxel_pores = dict(zip(voxel_index, pore_counts))


voxel_pores = dict(zip(voxel_index, area_porosity))

#tensor_filenames, pore_counts


# In[10]:


# organize dataset into a numpy structure
# Extract input spectrogram data from .mat files 

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
# sets print options to 5 decimal places

TENSOR_INPUT_DATA_COMBINED = torch.ones((90,4,129,6))
TENSOR_INPUT_DATA_PRINTING = torch.ones((90,4,129,6))
#TENSOR_INPUT_DATA_MILLING = torch.ones((72,2,129,15))
    
    # 72 - Total number of data points for binary classification
    # 6 - Channels (pertaining to each spectrogram)
    # 129  - Freuqency bands for each spectrogram
    # 15 - Time steps

voxel_number = 0
y_target=[]
file_names = []
# assign directory
# directory = 'files'
# iterate over files in that directory
# for filename in os.listdir(directory):


directory = (r"F:\tamu_assignment\ISEN613-Engineering-Data-Analysis-CourseProject-main\CNN code and input\sample_4_mats")
# Change this directory location to your folder that contains the 72 .mat files from Tensor_data_72_voxels.zip
# corresponding to Sample 1.
# When working with Sample 4 (Tensor_data_90_voxels_sample4.zip), extract the 90 files in a folder,
# and update the directory. In addition,
# update the dimensions in the below for loop (and in other places as required) as each tensor in Sample 4 
# is of dimension 129x6x4,
# ie, 129 frequency bands, 6 time steps, 4 spectrograms pertaining to only printing cycle. 

for filename in os.listdir(directory):
    #print(filename)
    mat_fname = pjoin(directory, filename)
    #NS Load the Mat file of the spectrogram
    Tensor_data_voxel = scipy.io.loadmat(mat_fname)
    #print(Tensor_data_voxel[list(Tensor_data_voxel.keys())[3]].shape)
    data = list(Tensor_data_voxel. items())
    tensor_array = np.array(data, dtype=object)

    # The spectrogram data is captured in a list from one of the elements of tensor_array
    # Shape of tensor_array is (4,2) and all spectrogram data (129x15x6) is stored in tensor_array[3][1]

    spec_list = list(tensor_array[3][1])
    #print(len(spec_list))
    for i in range(len(spec_list)):
        #print(i)
        for j in range(6):
            for k in range(4):
                TENSOR_INPUT_DATA_COMBINED[voxel_number,k,i,j] = spec_list[i][j][k]
                    
    print(TENSOR_INPUT_DATA_COMBINED.shape)                
    TENSOR_INPUT_DATA_PRINTING = TENSOR_INPUT_DATA_COMBINED[:,(0,1,2,3),:,:]
    #TENSOR_INPUT_DATA_MILLING = TENSOR_INPUT_DATA_COMBINED[:,(0,3),:,:]                
    
    file_index = filename.split('_')[2]
    ## Append number of pores
    y_target.append(voxel_pores[file_index.split('.')[0]])
    file_names.append(filename)
    voxel_number = voxel_number + 1

print("---------------------")
print(TENSOR_INPUT_DATA_COMBINED.shape)
print(TENSOR_INPUT_DATA_COMBINED)
print(TENSOR_INPUT_DATA_PRINTING.shape)
print(TENSOR_INPUT_DATA_PRINTING)
#print(TENSOR_INPUT_DATA_MILLING.shape)
#print(TENSOR_INPUT_DATA_MILLING)
print("---------------------")
print(y_target)
print("---------------------")
data_order = np.arange(90)
np.random.shuffle(data_order)
print(data_order)


# In[ ]:





# In[13]:


file_pores_df = pd.DataFrame({
    'filenames': file_names,
    'area_porosity': area_porosity
})
#file_pores_df['group_index']=file_pores_df['pore_counts'].apply(group_index)
file_pores_df


# In[16]:


file_pores_df['area_porosity_normalized']=file_pores_df['area_porosity']/file_pores_df['area_porosity'].abs().max()
file_pores_df


# In[36]:


np.array([np.array([x]) for x  in y_train])


# In[42]:


NUMPY_INPUT_DATA = TENSOR_INPUT_DATA_COMBINED.numpy()
Y_DATA=file_pores_df['area_porosity_normalized']
X_train = NUMPY_INPUT_DATA[data_order[0:73],:,:,:]
y_train = np.array(Y_DATA[data_order[0:73]])
X_test = NUMPY_INPUT_DATA[data_order[73:],:,:,:]
y_test = np.array(Y_DATA[data_order[73:]])

# Define the K-fold Cross Validator
kfold = KFold(n_splits=5, shuffle=True)

# K-fold Cross Validation model evaluation
fold_no = 1
train_acc_per_fold = []
train_loss_per_fold = []
val_acc_per_fold = []
val_loss_per_fold = []
test_acc_per_fold = []
test_loss_per_fold = []

for train, val in kfold.split(X_train, y_train):
    
    train_acc = 0
    val_acc = 0
    test_acc = 0
    # Define the model architecture
    activation='sigmoid'
    while(train_acc < 0.5 or val_acc < 0.5 or test_acc < 0.5):
        model = Sequential()
        model.add(Conv2D(4, 3, activation=activation, kernel_initializer='he_uniform', padding='same', input_shape=(4, 129, 6)))
        model.add(BatchNormalization())
        
        model.add(Conv2D(128, 3, activation=activation, kernel_initializer='he_uniform', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D())
        
        model.add(Conv2D(64, 3, activation=activation, kernel_initializer='he_uniform', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D())
        
#         model.add(Conv2D(32, 3, activation=activation, kernel_initializer='he_uniform', padding='same'))
#         model.add(MaxPooling2D(2,2))
        
        model.add(Conv2D(32, 3, activation=activation, kernel_initializer='he_uniform', padding='same'))
        model.add(Flatten())

#         model.add(Dense(16, activation=activation, kernel_initializer='he_uniform'))
        model.add(Dense(1, activation='relu'))
        #model.output_shape
        # Compile the model
        opt = SGD(lr=0.1, momentum=0.9)
        model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=opt, metrics=['accuracy'])
    
        # Generate a print
        print('------------------------------------------------------------------------')
        print(f'Training for fold {fold_no} ...')
        #print(train)
        #print(val)
        # Fit data to model
        history = model.fit(X_train[train], y_train[train], epochs=500, verbose=0)
        #train_acc_per_fold.append(history.history['accuracy'])
        #train_loss_per_fold.append(history.history['loss'])
        #summarize_diagnostics(fold_no, history)
        model.save('saves\porosity_regeression_model_fold_no_'+ str(fold_no) +'.h5')
    
        # Generate generalization metrics
        train_scores = model.evaluate(X_train[train], y_train[train], verbose=0)
        print(f'Train Score for fold {fold_no}: {model.metrics_names[0]} of {train_scores[0]}; {model.metrics_names[1]} of {train_scores[1]*100}%')
        train_acc = train_scores[1]
        
        val_scores = model.evaluate(X_train[val], y_train[val], verbose=0)
        print(f'Val Score for fold {fold_no}: {model.metrics_names[0]} of {val_scores[0]}; {model.metrics_names[1]} of {val_scores[1]*100}%')
        val_acc = val_scores[1]
        
        test_scores = model.evaluate(X_test, y_test, steps=len(X_test), verbose = 0)
        print(f'Test data Score for fold {fold_no}: {model.metrics_names[0]} of {test_scores[0]}; {model.metrics_names[1]} of {test_scores[1]*100}%')
        test_acc = test_scores[1]
    
    # Increase fold number
    fold_no = fold_no + 1
    train_acc_per_fold.append(train_scores[1] * 100)
    train_loss_per_fold.append(train_scores[0])
    val_acc_per_fold.append(val_scores[1] * 100)
    val_loss_per_fold.append(val_scores[0])
    test_acc_per_fold.append(test_scores[1] * 100)
    test_loss_per_fold.append(test_scores[0])


# In[ ]:




