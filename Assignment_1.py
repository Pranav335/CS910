# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 11:56:43 2023

@author: Pranav Agrawal
"""

from keras.datasets import fashion_mnist
import numpy as np
import pdb
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from tqdm import tqdm



import wandb

wandb.login()

(xtrain, ytrain), (xtest, ytest) = fashion_mnist.load_data()

# Extract unique labels from the dataset for visualization

yunique, index = np.unique(ytrain[:100], return_index = True)

labels = ['Tshirts', 'Trouser', 'Pullover',
          'Dress', 'Coat', 'Sandal', 'Shirt',
          'Sneaker', 'Bag', 'Ankle Boot']


# Sending the image to the wandb project 
# Question 1

# run = wandb.init(project = 'CS6910-Assignment-1')
# example = []
# for i in index:
#   img = xtrain[i, :, :]
#   img = wandb.Image(img, caption = labels[ytrain[i]])
#   example.append(img)
# wandb.log({'examples': example})
# wandb.finish()


# Reshaping the array into a new row.
xtrain = np.reshape(xtrain, (xtrain.shape[0], -1))/255
xtest = np.reshape(xtest, (xtest.shape[0], -1))/255

# One-hot encoding
encoder = OneHotEncoder()
ytrain_oh = encoder.fit_transform(ytrain.reshape(-1,1))
ytest_oh = encoder.fit_transform(ytest.reshape(-1,1))

ytrain_oh = ytrain_oh.toarray()
ytest_oh = ytest_oh.toarray()

# Define sweep configuration

class neural_network:

  def __init__(self, n_x , hidden_layer = [2]):
    '''

      Parameters
      ----------
      n_x: Number of input features
      
      hidden_layer : List, The length of the list
      is the number of hidden layers and element of the list 
      is the number of neuron in the ith layer. The default is [2].
      
      Returns
      -------
      None.

      '''
    self.n_x = n_x
    self.L = len(hidden_layer) # 1
    self.neuron = [self.n_x] + hidden_layer + [10] # [784, 1, 10 ]
    self.W = {}
    self.B = {}
    self.A = {}
    self.H = {}
    for i in range(self.L + 1):
      self.W[i+1] = np.random.randn(self.neuron[i+1], self.neuron[i]) #[n_y x n_x]
      self.B[i+1] = np.zeros((self.neuron[i+1], 1)) # [n_y, 1]

  def positive_sigmoid(self, x):
    ''' Calculate the sigmoid function when x > 0'''
    return 1.0/(1.0 + np.exp(-x))

  def negative_sigmoid(self, x):
    '''Calculate the sigmoid function when x < 0'''
    exp = np.exp(x)
    return exp/ (1 + exp)

  def sigmoid(self, x):
    ''' Calculate the sigmoid function'''
    positives = x >=0  # Extract the index of the positive values
    negative = ~positives

    result = np.empty_like(x)
    result[positives] = self.positive_sigmoid(x[positives]) # Find the sigmoid of the the positive 
    result[negative] = self.negative_sigmoid(x[negative])

    return result

  def softmax(self, x):
    m = np.max(x)
    exps = np.exp(x-m)
    #return exps/ np.sum(exps, axis = -1, keepdims = True)
    return exps/ np.sum(exps)

  def forward_pass(self,x):
    x = x.reshape(1, -1)
    self.H[0] = x.T # (784 x N)
    for i in range(self.L):
      self.A[i+1] = self.W[i+1] @ self.H[i]  + self.B[i+1] # (2, 784) x (784, N) + (2, 1) -> (2, N)
      self.H[i+1] = self.sigmoid(self.A[i+1]) #(2, N)

    self.A[self.L+1] = self.W[self.L+1] @ self.H[self.L] + self.B[self.L+1] #(k, k-1) x (k-1, N) + (k, 1)-> (k, N)
    self.H[self.L+1] = self.softmax(self.A[self.L + 1]) # (k, N)

    return self.H[self.L+1] 

  def grad_sigmoid(self, x):
    return x * (1 - x)

  def gradient(self, x, y):
    y = y.reshape(-1, 1)
    self.dW = {}
    self.dB = {}
    self.dA = {}
    self.dH = {}
    # y = (N, k)
    self.forward_pass(x)
    self.dA[self.L + 1] = self.H[self.L+1] - y # (k, N)
    for k in range(self.L+1, 0, -1):
      self.dW[k] = self.dA[k] @ self.H[k-1].reshape(1, -1) # (2, N) x (N, 2) -> (2, 2)
      #self.dB[k] = np.sum(self.dA[k], axis = 1).reshape(-1, 1) # (2, N) -> (2, 1)
      self.dB[k] = self.dA[k]
      if k > 1:
        self.dH[k-1] = self.W[k].T @ self.dA[k] 
        self.dA[k-1] = self.dH[k-1] * self.grad_sigmoid(self.H[k-1])
        
  
  def cross_entropy(self, ytrue, ypred, eps = 0.0001):
    ytrue = ytrue.reshape(-1, 1)
    index = np.argmax(ytrue)
    ypred = ypred.reshape(-1, 1)
    y = ypred[index,0]
    loss  =  -1 * np.log(y + eps)
    
    return loss


  def gradient_descent(self, x, y, lr, batch_size):
    n_samples = x.shape[0]
    num_seen_points = 0
    # Initializing dw and db to store the sum of gradient
    dW = {}
    dB = {}
    for s in range(self.L + 1):
          dW[s+1] = np.zeros((self.neuron[s+1], self.neuron[s])) #[n_y x n_x]
          dB[s+1] = np.zeros((self.neuron[s+1], 1))
    for j in range(n_samples):    
      # Computing the gradient 
      
      self.gradient(x[j,:], y[j,:])
      # Accumulating the gradient for the batch
      for k in range(self.L + 1):
          dW[k+1] += self.dW[k+1]
          dB[k+1] += self.dB[k+1]
      num_seen_points += 1
      # If the batch size is reached, making the update and setting the 
      # gradient to zero for next batch.
      ## What will happen if we do not set the gradient back to zero.
      if num_seen_points%batch_size == 0:
        self.W[k+1] -= lr * dW[k+1]
        self.B[k+1] -= lr * dB[k+1]
        for s in range(self.L + 1):
          dW[s+1] = np.zeros((self.neuron[s+1], self.neuron[s])) #[n_y x n_x]
          dB[s+1] = np.zeros((self.neuron[s+1], 1))

  def momentum_based_gradient_descent(self, x, lr, batch_size, beta):
    pass


  def accuracy(self, X, Y):
    n_samples = X.shape[0]
    accuracy = 0
    for j in range(n_samples):
      yhat = self.forward_pass(X[j,:])
      pred_class = np.argmax(yhat)
      true_class = np.argmax(Y[j,:])
      #print(pred_class, true_class)
      if pred_class == true_class:
        accuracy += 1

    accuracy /= n_samples
    return accuracy



  def fit_neural_network(self,X, Y, epochs = 10, lr = 0.00001,
                         batch_size = 40,
                         optimizer = 'GD', beta = 0.05,
                         eps = 1e-5, train_plot = False):
    
  
    loss = []                          

    n_samples = X.shape[0]
    train_accuracy = []
    # Training the model.
    for i in tqdm(range(epochs), desc = 'Epochs'):
      epoch_loss = 0
      if optimizer == 'GD':
        self.gradient_descent(X,Y, lr = lr, batch_size = batch_size)

      for j in range(n_samples):
        yhat = self.forward_pass(X[j, :])
        epoch_loss += self.cross_entropy(Y[j,:], yhat)
      
      loss.append(epoch_loss/n_samples)
      train_accuracy.append(self.accuracy(X, Y))
      
      # if i > 10:
      #     plt.figure()
      #     plt.plot([*range(i)], loss, label = 'Loss')
      #     plt.plot([*range(i)], train_accuracy, label = 'Accuracy')
      #     plt.xlabel('Epochs')
      #     plt.ylabel('Loss')
      #     plt.legend()
          
   
          
    # Plotting the loss
    # if train_plot == True:
    #     plt.plot([*range(epochs)], loss, label = 'Loss')
    #     plt.plot([*range(epochs)], train_accuracy, label = 'Accuracy')
    #     plt.xlabel('Epochs')
    #     plt.ylabel('Loss')
    #     plt.legend()
    
    return epoch_loss



  def fit_transform(self, x, y):
    n_samples = x.shape[0]
    accuracy = 0
    for j in range(n_samples):
      yhat = self.forward_pass(x[j,:])
      pred_class = np.argmax(yhat)
      true_class = np.argmax(y[j,:])
      #print(pred_class, true_class)
      if pred_class == true_class:
        accuracy += 1

    accuracy /= n_samples
    return accuracy
      


sweep_configuration = {'method': 'random',
                       'name': 'sweep',
                       'metric': {'goal': 'minimize', 'name': 'epoch_loss'},
                       'parameters':
                           {
                             'batch_size': {'values':[16, 32, 64]},
                             'epochs': {'values': [20, 30, 40]},
                             'lr': {'values': [1e-10,1e-7, 1e-5, 1e-3]}
                             }
    }
    

def wandbsweeps():
    wandb.init(project = 'CS6910-Assignment-1')
    nn = neural_network(784)
    epoch_loss = nn.fit_neural_network(xtrain, ytrain_oh, lr = wandb.config.lr,
                                       epochs = wandb.config.epochs,
                                       batch_size = wandb.config.batch_size)
    wandb.log({'epoch_loss': epoch_loss})

nn = neural_network(784)
#epoch_loss = nn.fit_neural_network(xtrain, ytrain_oh, epochs = 50, optimizer = 'GD')

sweep_id = wandb.sweep(sweep= sweep_configuration, project = 'CS6910-Assignment-1')
wandb.agent(sweep_id, function = wandbsweeps, count = 10)
## Lets go debug mode.
## Lets do another debug







