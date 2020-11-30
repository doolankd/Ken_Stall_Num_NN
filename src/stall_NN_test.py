#! /usr/bin/env python
import math
import numpy as np
import re
import random
from sklearn.metrics import confusion_matrix
import cv2 as cv2

from collections import Counter
from matplotlib import pyplot as plt
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator

from os import listdir
from os.path import isfile, join

#training CNN
from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tensorflow.python.keras import optimizers

from tensorflow.python.keras.utils import plot_model
from tensorflow.python.keras import backend

import sys

import seaborn as sn
import pandas as pd

# ./stall_NN_test.py

# load the NN object
conv_model = models.load_model("NN_object")

PATH_testing = "/home/fizzer/ros_ws/src/my_controller/pictures/NN_stall_num_pics/all_pics"
folder0 = PATH_testing
files0 = [f for f in listdir(PATH_testing) if isfile(join(PATH_testing, f))]

all_stall_nums = []
for file in files0:
  stall_num = list(file)
  i = -5
  all_stall_nums.append(stall_num[i])

print(len(all_stall_nums))

# ./stall_NN_test.py

# put the appropriate characters and letters into the proper bins
first = 1
count = 1
for num in all_stall_nums:
  i = 0 # general index
  j = 0 # ascii indexing
  # Ascii numbers start at 48 (0) - so 1 will be at 49
  #print(num)
  ascii_num = 49 
  individual_num_array = np.array([[0,0,0,0,0,0,0,0]]) #np.array([[0,0,0,0,0,0,0,0]])
  current_char_to_num = ord(num) # gets ascii equivalent
  while current_char_to_num > ascii_num:
    ascii_num = ascii_num + 1
    j = j + 1
  individual_num_array[0,j] = 1
  row_num_array = individual_num_array
  count = count + 1
  if first == 1:
    first = 0
    num_bin_label = row_num_array
  else:
    intermediate_bin = np.array(row_num_array)
    num_bin_label = np.concatenate((num_bin_label, intermediate_bin))

# test code above

print(num_bin_label.shape)
print(num_bin_label[0])
print(len(files0))

# something from Dannon's code, adapted for my set:
classes = np.array([])
for i in range(1,9):
  classes = np.append(classes, i)

#classes = "12345678"

first = True
for i in range(len(files0)):
    sim_img = np.asarray(Image.open(folder0 + "/" + files0[i]))
    sim_img = sim_img.reshape(1,sim_img.shape[0],sim_img.shape[1],1)
    img = cv2.cvtColor(sim_img[0], cv2.COLOR_GRAY2RGB)
    if first:
      #print("sim_img: " + str(sim_img.shape))
      #print("gray: " + str(img.shape))
      first = False
      new_set = np.array([img])
      #print("img: " + str(img.shape))
      #print("new: " + str(new_set.shape))
    else:
      new_set = np.vstack((new_set,[img]))

print("Loaded {:} images from folder:\n{}".format(new_set.shape[0], folder0))
print(new_set.shape)
print(new_set[0].shape)

def mapPredictionToCharacter(y_predict):
    #maps NN predictions to the numbers based on the max probability.
    y_predicted_max = np.max(y_predict)
    index_predicted = np.where(y_predict == y_predicted_max)
    character = classes[index_predicted]
    return character[0]

def testNN(files):
    y_pred = np.array([])
    y_true = np.array([])
    #for i in range(len(files)):
    for j in range(len(new_set)):

        img_aug = np.expand_dims(new_set[j], axis=0)
        y_predict = conv_model.predict(img_aug)[0]

        predicted_character = mapPredictionToCharacter(y_predict)
        y_pred = np.append(y_pred,int(predicted_character))

        true_character = int(all_stall_nums[j]) #num_bin_label[j,(np.where(1 == 1)+1)]
        #num_bin_label[j]
        y_true = np.append(y_true,true_character)
        #print(y_true)
        #print(y_pred)
        #break
    	#break

        print("predicted: ", predicted_character)
        print("actual: ", true_character)
        print("\n")

    return y_true, y_pred

# ./stall_NN_test.py

y_true, y_pred = testNN(files0)
np.set_printoptions(threshold=sys.maxsize)
print("************************")
print(y_true.shape)
print(y_pred.shape)
#from https://stackoverflow.com/questions/35572000/how-can-i-plot-a-confusion-matrix
confusion_matrix = confusion_matrix(y_true,y_pred)
print(confusion_matrix)
df_cm = pd.DataFrame(confusion_matrix, index = [i for i in "12345678"],
                  columns = [i for i in "12345678"])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True)
plt.show()

'''
def testAllImages(dataset):
    for image in dataset:
        img_aug = np.expand_dims(img, axis=0)
        y_predict = conv_model.predict(img_aug)[0]
        y_predicted_max = np.max(y_predict)
        index_predicted = np.where(y_predict == y_predicted_max)
        print(index_predicted)
    return None
'''