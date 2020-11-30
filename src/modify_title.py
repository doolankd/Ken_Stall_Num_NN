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

# get PATH
PATH = "/home/fizzer/ros_ws/src/my_controller/pictures/NN_stall_num_pics/all_7"
write_location = "/home/fizzer/ros_ws/src/my_controller/pictures/NN_stall_num_pics/all_7/"
folder0 = PATH
files0 = [f for f in listdir(PATH) if isfile(join(PATH, f))]

def convert_to_str(input_seq, seperator):
   # Join all the strings in list
   final_str = seperator.join(input_seq)
   return final_str

# get the names of all files and save as a list
all_stall_names = []
files_written = 0
for file in files0:
  stall_name = list(file)
  i = -5
  stall_name[i] = '7' 
  
  print(''.join(stall_name))

  sim_img = np.asarray(Image.open(folder0 + "/" + file))
  cv2.imwrite(write_location + ''.join(stall_name), sim_img)
  print("files_written: " + str(files_written+1))
  #print(str(all_stall_names[i]))
  #all_stall_names.append(stall_num)

# get each image and write new title right away
first = True
#print(all_stall_names[0])
#print(str(all_stall_names[0]))
#print(''.join(str(all_stall_names[0])))
#print(convert_to_str(str(all_stall_names[0]),","))

# ./modify_title.py

'''
for i in range(len(files0)):
	sim_img = np.asarray(Image.open(folder0 + "/" + files0[i]))
	cv2.imwrite(write_location + str(all_stall_names[i]), sim_img)
	print("files_written: " + str(files_written+1))
	print(str(all_stall_names[i]))
	break
'''
'''
	#sim_img = sim_img.reshape(1,sim_img.shape[0],sim_img.shape[1],1)
	if first:
	  first = False
	  new_set = np.array([img])
	else:
	  new_set = np.vstack((new_set,[img]))

print("Loaded {:} images from folder:\n{}".format(new_set.shape[0], folder0))
print(new_set.shape)
print(new_set[0].shape)

i = 0
for file in new_set:
	cv2.imwrite(write_location + all_stall_names[i], file)
	files_written = files_written + 1
	print("files_written: " + str(files_written))
'''