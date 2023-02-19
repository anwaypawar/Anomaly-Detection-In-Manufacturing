#!/usr/bin/env python
# coding: utf-8

# In[5]:



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


# In[3]:


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

pore_counts=[int(x.strip()) for x in '''
1
1
4
5
22
17
20
15
5
0
1
0
2
4
5
22
25
13
17
4
3
9
7
10
26
33
26
24
16
9
7
8
5
7
26
35
25
17
2
1
11
3
6
6
15
22
28
12
9
4
3
3
6
8
9
10
20
3
7
1
5
7
3
8
0
3
4
5
4
1
4
5
5
7
7
6
4
3
2
1
0
3
1
5
7
5
4
4
5
4
'''.split('\n') if x]


# In[9]:


#NS
import pandas as pd
import math
from keras.utils import normalize, to_categorical

# Groups of pore counts (ex: 0-5, 6-10, ...)
GROUP_WIDTH=5
NUM_GROUPS=10
group_index = lambda pore_count: math.ceil(pore_count/GROUP_WIDTH)
#category_vector = lambda grp_index: to_categorical(grp_index, NUM_GROUPS).astype(int)

voxel_pores = dict(zip(voxel_index, pore_counts))


#tensor_filenames, pore_counts


# In[5]:


# organize dataset into a numpy structure
# Extract input spectrogram data from .mat files 

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
# sets print options to 5 decimal places

TENSOR_INPUT_DATA_COMBINED = torch.ones((72,6,129,15))
TENSOR_INPUT_DATA_PRINTING = torch.ones((72,4,129,15))
TENSOR_INPUT_DATA_MILLING = torch.ones((72,2,129,15))
    
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


directory = (r"C:\Users\Admin\Desktop\613 project\input")
# Change this directory location to your folder that contains the 72 .mat files from Tensor_data_72_voxels.zip
# corresponding to Sample 1.
# When working with Sample 4 (Tensor_data_90_voxels_sample4.zip), extract the 90 files in a folder,
# and update the directory. In addition,
# update the dimensions in the below for loop (and in other places as required) as each tensor in Sample 4 
# is of dimension 129x6x4,
# ie, 129 frequency bands, 6 time steps, 4 spectrograms pertaining to only printing cycle. 

for filename in os.listdir(directory):
    # print(filename)
    mat_fname = pjoin(directory, filename)
    #NS Load the Mat file of the spectrogram
    Tensor_data_voxel = scipy.io.loadmat(mat_fname)

    data = list(Tensor_data_voxel. items())
    tensor_array = np.array(data, dtype=object)

    # The spectrogram data is captured in a list from one of the elements of tensor_array
    # Shape of tensor_array is (4,2) and all spectrogram data (129x15x6) is stored in tensor_array[3][1]

    spec_list = list(tensor_array[3][1])

    for i in range(len(spec_list)):
        for j in range(15):
            for k in range(6):
                TENSOR_INPUT_DATA_COMBINED[voxel_number,k,i,j] = spec_list[i][j][k]
                    
                    
    TENSOR_INPUT_DATA_PRINTING = TENSOR_INPUT_DATA_COMBINED[:,(1,2,4,5),:,:]
    TENSOR_INPUT_DATA_MILLING = TENSOR_INPUT_DATA_COMBINED[:,(0,3),:,:]                
    
    file_index = filename.split('_')[2]
    ## Append number of pores
    y_target.append(voxel_pores[file_index.split('.')[0]])
    file_names.append(filename)
    voxel_number = voxel_number + 1
    

# y_target = np.zeros(72)
# y_target[4:10] = 1
# y_target[13:18] = 1
# y_target[22:34] = 1
# y_target[37:41] = 1
# y_target[43:46] = 1
# y_target[47] = 1
# y_target[57] = 1
# y_target[59] = 1
# y_target[70] = 1

print("---------------------")
print(TENSOR_INPUT_DATA_COMBINED.shape)
print(TENSOR_INPUT_DATA_COMBINED)
print(TENSOR_INPUT_DATA_PRINTING.shape)
print(TENSOR_INPUT_DATA_PRINTING)
print(TENSOR_INPUT_DATA_MILLING.shape)
print(TENSOR_INPUT_DATA_MILLING)
print("---------------------")
print(y_target)
print("---------------------")
data_order = np.arange(72)
np.random.shuffle(data_order)
print(data_order)


# In[7]:


file_pores_df = pd.DataFrame({
    'filenames': file_names,
    'pore_counts': y_target
})
file_pores_df['group_index']=file_pores_df['pore_counts'].apply(group_index)
file_pores_df


# In[39]:


model = tf.keras.Sequential()
model.add(Conv2D(6, (1,1), activation=activation, kernel_initializer='he_uniform', padding='same', input_shape=(6, 129, 15)))
model.add(Conv2D(64, (7, 7), activation=activation, kernel_initializer='he_uniform', padding='same'))
model.add(MaxPooling2D(3,3))
model.add(Conv2D(4, 2, activation=activation, kernel_initializer='he_uniform', padding='same'))
model.output_shape


# In[101]:


model = Sequential()
model.add(Conv2D(6, 3, activation=activation, kernel_initializer='he_uniform', padding='same', input_shape=(6, 129, 15)))
model.add(BatchNormalization())

model.add(Conv2D(128, 3, activation=activation, kernel_initializer='he_uniform', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Conv2D(64, 3, activation=activation, kernel_initializer='he_uniform', padding='same'))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(64, 3, activation=activation, kernel_initializer='he_uniform', padding='same'))

model.add(Flatten())
model.add(Dense(128, activation=activation, kernel_initializer='he_uniform'))
#model.add(Conv2D(16, 2, activation=activation, kernel_initializer='he_uniform', padding='same'))


model.add(Dense(10, activation='softmax'))
model.output_shape

# model.add(Conv2D(64, (4, 4), activation=activation, kernel_initializer='he_uniform', padding='same'))
# # model.add(MaxPooling2D(2,2))
# model.output_shape
# model.add(Conv2D(16, 2, activation=activation, kernel_initializer='he_uniform', padding='same'))


# In[106]:


NUMPY_INPUT_DATA = TENSOR_INPUT_DATA_COMBINED.numpy()
Y_DATA=to_categorical(file_pores_df['group_index'], num_classes=NUM_GROUPS)
X_train = NUMPY_INPUT_DATA[data_order[0:64],:,:,:]
y_train = Y_DATA[data_order[0:64]]
X_test = NUMPY_INPUT_DATA[data_order[64:72],:,:,:]
y_test = Y_DATA[data_order[64:72]]

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
#         model = Sequential()
#         model.add(Conv2D(6, (7, 7), activation=activation, kernel_initializer='he_uniform', padding='same', input_shape=(6, 129, 15)))
#         model.add(BatchNormalization())
        
#         model.add(Conv2D(128, (7, 7), activation=activation, kernel_initializer='he_uniform', padding='same'))
#         model.add(MaxPooling2D())
#         model.add(Conv2D(64, (4, 4), activation=activation, kernel_initializer='he_uniform', padding='same'))
#         model.add(MaxPooling2D())
#         model.add(Conv2D(16, (2,2), activation=activation, kernel_initializer='he_uniform', padding='same'))
#         model.add(Conv2D(64, (4, 4), activation=activation, kernel_initializer='he_uniform', padding='same'))
#         model.add(MaxPooling2D())
#         model.add(Conv2D(16, 2, activation=activation, kernel_initializer='he_uniform', padding='same'))
# #         model.add(Conv2D(64, (3, 3), activation=activation, kernel_initializer='he_uniform', padding='same'))
# #         model.add(MaxPooling2D((2,2), padding='same'))
        
#         # model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
#         # model.add(MaxPooling2D((2, 2)))
#         model.add(Flatten())
# #         model.add(Dense(16, activation=activation, kernel_initializer='he_uniform'))
#         model.add(Dense(10, activation='softmax'))
        model = Sequential()
        model.add(Conv2D(6, 3, activation=activation, kernel_initializer='he_uniform', padding='same', input_shape=(6, 129, 15)))
        model.add(BatchNormalization())

        model.add(Conv2D(128, 3, activation=activation, kernel_initializer='he_uniform', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D())
        model.add(Conv2D(64, 3, activation=activation, kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPooling2D(2,2))
        model.add(Conv2D(64, 3, activation=activation, kernel_initializer='he_uniform', padding='same'))

        model.add(Flatten())
        model.add(Dense(128, activation=activation, kernel_initializer='he_uniform'))
        #model.add(Conv2D(16, 2, activation=activation, kernel_initializer='he_uniform', padding='same'))


        model.add(Dense(10, activation='softmax'))
        #model.output_shape
        # Compile the model
       #  opt = SGD(lr=0.01, momentum=0.9)
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    
        # Generate a print
        print('------------------------------------------------------------------------')
        print(f'Training for fold {fold_no} ...')

        # Fit data to model
        history = model.fit(X_train[train], y_train[train], epochs=500, verbose=0)
        #train_acc_per_fold.append(history.history['accuracy'])
        #train_loss_per_fold.append(history.history['loss'])
        #summarize_diagnostics(fold_no, history)
        model.save('saves\porosity_dense_categorical_model_fold_no_'+ str(fold_no) +'.h5')
    
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


# In[2]:


get_ipython().system('pip install keras')


# In[6]:


get_ipython().system('pip install keras layer')


# In[ ]:





# In[ ]:





# In[ ]:




