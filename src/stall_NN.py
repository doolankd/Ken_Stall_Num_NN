#! /usr/bin/env python

import math
import numpy as np
import re
import random
import cv2

from collections import Counter
from matplotlib import pyplot as plt
from PIL import Image
from os import listdir
from os.path import isfile, join

# CHANGE PATH
PATH = "/home/fizzer/ros_ws/src/my_controller/pictures/NN_stall_num_pics/all_pics"
folder0 = PATH
files0 = [f for f in listdir(PATH) if isfile(join(PATH, f))]

#print(files0)

# ./stall_NN.py

# this is very important for later on. Your 80% validation considered the FIRST
# 80% of alphabetized license plates, so it's likely it would've been trained 
# with a lot more A's and B's than Y's and Z's
random.shuffle(files0) 

all_stall_nums = []
for file in files0:
  stall_num = list(file)
  i = -5
  all_stall_nums.append(stall_num[i])

print(len(all_stall_nums))
#print(all_stall_nums)

# put the appropriate characters and letters into the proper bins
first = 1
count = 1
for num in all_stall_nums:
  i = 0 # general index
  j = 0 # ascii indexing
  # Ascii numbers start at 48 (0) - so 1 will be at 49
  #print(num)
  ascii_num = 49 #49
  individual_num_array = np.array([[0,0,0,0,0,0]]) #np.array([[0,0,0,0,0,0,0,0]])
  current_char_to_num = ord(num) # gets ascii equivalent
  while current_char_to_num > ascii_num:
    ascii_num = ascii_num + 1
    j = j + 1
  individual_num_array[0,j] = 1
  row_num_array = individual_num_array
  #print(count)
  count = count + 1
  if first == 1:
    first = 0
    num_bin_label = row_num_array
  else:
    intermediate_bin = np.array(row_num_array)
    num_bin_label = np.concatenate((num_bin_label, intermediate_bin))

# test code above

#num_bin_label.size
#ord("9")
#plate
#quarter_bin_label[0:4]
#quarter_bin_label.shape # 725 plates, 4 arrays denoting bin allocation, 36 bins
#quarter_bin_label
print(num_bin_label.shape)
print(num_bin_label[0])
print(len(files0))
#print(all_stall_nums) -all good

#### move from labels to dealing with actual image/dataset

# Load the license plates to imgset0
#imgset0 = np.array([[np.array(Image.open(f'{folder0}/{file}')), 0]
#                   for file in files0[:]])

first = True
for i in range(len(files0)):
    sim_img = np.asarray(Image.open(folder0 + "/" + files0[i]))
    sim_img = sim_img.reshape(1,sim_img.shape[0],sim_img.shape[1],1)
    if first:
        imgset0 = sim_img
        print(sim_img.shape)
        first = False
    else:
        imgset0 = np.vstack((imgset0,sim_img))
        #print("hello")

print("Loaded {:} images from folder:\n{}".format(imgset0.shape[0], folder0))
print(imgset0.shape)
print(folder0 + "/" + files0[0])

# ./stall_NN.py

i = 0
for img in imgset0:
  img = cv2.cvtColor(img[0], cv2.COLOR_GRAY2RGB)

  if i == 0:
    i = 1
    value = np.array([img])
    new_set = np.array([img])
  else:
    interm = img
    new_set = np.concatenate((new_set,[interm]))

print(new_set.shape)
print(img.shape)

# run for generated data
print(imgset0.shape)
X_dataset_orig = np.array([data[0] for data in new_set[:]])
X_dataset_orig[0].shape # ORIGINAL shape of a SINGLE image
print(X_dataset_orig.shape)

#print(x)
#imgset0
#plt.imshow(imgset0[0,0])

print(new_set[0].shape)

# run for sim generated data
print(imgset0.shape)
X_dataset_orig = np.array([data for data in new_set[:]])
X_dataset_orig[0].shape # ORIGINAL shape of a SINGLE image
print(X_dataset_orig.shape)

#print(x)
#imgset0
#plt.imshow(imgset0[0,0])

# will normalize data:

X_dataset_norm = X_dataset_orig/255
max = np.amax(X_dataset_orig)
print("max: " + str(max))
max = np.amax(X_dataset_norm)
print("norm: " + str(max))

Y_final_set = num_bin_label
X_final_set = np.copy(X_dataset_norm)

from ipywidgets import interact
import ipywidgets as ipywidgets

def displayImage(index):
  plt.imshow(X_final_set[index])
  caption = ("y = " + str(num_bin_label[index])) # str(np.squeeze(Y_dataset_orig[:, index])))
  plt.text(0.5, 0.5, caption, 
           color='orange', fontsize = 20,
           horizontalalignment='left', verticalalignment='top')


displayImage(1) # (first number is plate # in folder:pictures, second number is which quarter to display)

VALIDATION_SPLIT = 0.2 # 20% are reserved for validation

print("Total examples: {:d}\nTraining examples: {:d}\nTest examples: {:d}".
      format(X_final_set.shape[0],
             math.ceil(X_final_set.shape[0] * (1-VALIDATION_SPLIT)),
             math.floor(X_final_set.shape[0] * VALIDATION_SPLIT)))
print("X shape: " + str(X_final_set.shape))
print("Y shape: " + str(Y_final_set.shape))

X_final_set.shape
X_final_set[0,0,0]

master_data_set = np.copy(X_final_set)
master_label_set = np.copy(Y_final_set) 

print("master_label_set shape: " + str(master_label_set.shape))
print("master_data_set shape: " + str(master_data_set.shape))
'''