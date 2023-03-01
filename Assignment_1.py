# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 11:56:43 2023

@author: Pranav Agrawal
"""

from keras.datasets import fashion_mnist
import numpy as np
import pdb


import wandb

wandb.login()

(xtrain, ytrain), (xtest, ytest) = fashion_mnist.load_data()

# Extract unique labels from the dataset for visualization

yunique , index = np.unique(ytrain[:100], return_index = True)

labels = ['Tshirts', 'Trouser', 'Pullover',
          'Dress', 'Coat', 'Sandal', 'Shirt',
          'Sneaker', 'Bag', 'Ankle Boot']


# Sending the image to the wandb project 
# Question 1

run = wandb.init(project = 'CS6910-Assignment-1')
example = []
for i in index:
  img = xtrain[i, :, :]
  img = wandb.Image(img, caption = labels[ytrain[i]])
  example.append(img)
wandb.log({'examples': example})
wandb.finish()


##  Adding new comment




