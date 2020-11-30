#! /usr/bin/env python

import math
import numpy as np
import re
import random
import cv2
import sys

from collections import Counter
from matplotlib import pyplot as plt
from PIL import Image
from os import listdir
from os.path import isfile, join
from sklearn.metrics import confusion_matrix

# PATH
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
  individual_num_array = np.array([[0,0,0,0,0,0,0,0]]) #np.array([[0,0,0,0,0,0,0,0]])
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

print(num_bin_label.shape)
print(num_bin_label[0])
print(len(files0))

#### move from labels to dealing with actual image/dataset

# This did not work here - used dannon's code that he gave in the lab book
# Load the license plates to imgset0
#imgset0 = np.array([[np.array(Image.open(f'{folder0}/{file}')), 0]
#                   for file in files0[:]])

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

# ./stall_NN.py

# copying the sim generated data
X_dataset_orig = np.array([data for data in new_set[:]])
X_dataset_orig[0].shape # ORIGINAL shape of a SINGLE image
print("X_dataset: " + str(X_dataset_orig.shape))

# will normalize data:
X_dataset_norm = X_dataset_orig/255
max = np.amax(X_dataset_orig)
print("original max: " + str(max))
max = np.amax(X_dataset_norm)
print("norm max: " + str(max))

Y_final_set = num_bin_label
X_final_set = np.copy(X_dataset_norm)


VALIDATION_SPLIT = 0.2 # 20% are reserved for validation

print("Total examples: {:}\nTraining examples: {:}\nTest examples: {:}".
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

# ./stall_NN.py
'''
from keras import layers
from keras import models
from keras import optimizers

from keras.utils import plot_model
from keras import backend
'''

from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tensorflow.python.keras import optimizers

from tensorflow.python.keras.utils import plot_model
from tensorflow.python.keras import backend

def reset_weights(model):
  for ix, layer in enumerate(model.layers):
      if (hasattr(model.layers[ix], 'kernel_initializer') and 
          hasattr(model.layers[ix], 'bias_initializer')):
          weight_initializer = model.layers[ix].kernel_initializer
          bias_initializer = model.layers[ix].bias_initializer

          old_weights, old_biases = model.layers[ix].get_weights()

          model.layers[ix].set_weights([
              weight_initializer(shape=old_weights.shape),
              bias_initializer(shape=len(old_biases))])

conv_model = models.Sequential()
conv_model.add(layers.Conv2D(32, (4, 4), activation='relu',
                             input_shape=(100, 100, 3)))
conv_model.add(layers.MaxPooling2D((12, 12)))
conv_model.add(layers.Conv2D(48, (4, 4), activation='relu')) # 48
conv_model.add(layers.MaxPooling2D((4, 4)))
#conv_model.add(layers.Conv)
conv_model.add(layers.Flatten())
conv_model.add(layers.Dropout(0.40)) # are also ok:0.60, 0.48
conv_model.add(layers.Dense(128, activation='relu')) # 100
conv_model.add(layers.Dense(8, activation='softmax'))

print(conv_model.summary())

LEARNING_RATE = 1e-4
conv_model.compile(loss='categorical_crossentropy',
                   optimizer=optimizers.RMSprop(lr=LEARNING_RATE),
                   metrics=['acc'])

# ./stall_NN.py

history_conv = conv_model.fit(master_data_set, master_label_set, 
                              validation_split=VALIDATION_SPLIT, 
                              epochs=50, 
                              batch_size=24)

plt.plot(history_conv.history['loss'])
plt.plot(history_conv.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train loss', 'val loss'], loc='upper left')
plt.show()

plt.plot(history_conv.history['acc'])
plt.plot(history_conv.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy (%)')
plt.xlabel('epoch')
plt.legend(['train accuracy', 'val accuracy'], loc='upper left')
plt.show()

# confusion matrix

true_index = []
pred_index = []
index = 0
for label in master_label_set:
  ### find the index that has the 1, find the index that has the highest probability, graph those for each label

  # navigate through set of bin for the 1 entry
  i = 0
  for bin in label:
    if bin == 1:
      true_index.append(i)
      break
    else:
      i = i + 1

  img = master_data_set[index]
  img_aug = np.expand_dims(img, axis=0)
  y_predict = conv_model.predict(img_aug)[0]
  # find the highest probability index
  highest_prob = 0
  largest_i = 0
  i = 0
  for prediction in y_predict:
    if prediction > highest_prob:
      highest_prob = prediction
      largest_i = i
      i = i + 1
    else:
      i = i + 1
  pred_index.append(largest_i)

  index = index + 1

np.set_printoptions(threshold=sys.maxsize)

i = 0
true_count = 0
false_count = 0
while i < len(pred_index):
  if true_index[i] == pred_index[i]:
    true_count = true_count + 1
  else:
    false_count = false_count + 1
    #print(str(true_index[i]) + " " + str(pred_index[i]))
  i = i + 1

print("True: " + str(true_count))
print("False: " + str(false_count))
print(" ")
print("Confusion Matrix: ")
print(confusion_matrix(true_index, pred_index))

# ./stall_NN.py

# Display images in the training data set. 
def displayImage(index):
  img = master_data_set[index]
  
  img_aug = np.expand_dims(img, axis=0)
  y_predict = conv_model.predict(img_aug)[0]
  #print(y_predict)
  
  plt.imshow(img)  

  # find the highest probability
  highest_prob = 0
  for prediction in y_predict:
    if prediction > highest_prob:
      highest_prob = prediction

  print("Highest probability of input: " + str(highest_prob))

  # navigate through label data set for the 1 entry
  one_index = 0
  i = 0
  for bin in master_label_set[index]:
    if bin == 1:
      one_index = i
    else:
      i = i + 1
  print("index location: " + str(one_index))

  # get prediction at real index value
  guess_prediction = y_predict[one_index]
  print("Guess prediction: " + str(guess_prediction))

displayImage(3)

# saving the NN as an object
models.save_model(conv_model,"NN_object")

# ./stall_NN.py