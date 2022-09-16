# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 15:55:37 2022

@author: chonk
"""
#%%
#tensorflow imports
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import linear, relu, sigmoid

#matplotlib imports
import matplotlib.pyplot as plt

#maths imports
import numpy as np
import scipy as sp

#imaging
import PIL
from PIL import Image, ImageDraw

#os paths
import os.path
#%%
#process data
#data
def process_data():
    data = np.genfromtxt('data/train_data.csv', delimiter=',') #uses MNIST database, link:http://yann.lecun.com/exdb/mnist/
    
    n = len(data)
    x1 = []
    for i in range(n):
        value = data[i, 1:].astype(np.float64)
        x1.append(value)
    
    x = np.asarray(x1)
    y1 = data[0:, 0].astype(np.uint8)
    y = np.reshape(y1, (-1, 1))
    
    np.save('data/x.npy', x)
    np.save('data/y.npy', y)

#%%
#data
def load_data():
    if not os.path.exists('data/x.npy'):
        x, y = process_data()
    else:
        x = np.load('data/x.npy')
        y = np.load('data/y.npy')
    
    return x,y
#%%
#neural network definition
def create_network():
    tf.random.set_seed(1234) 
    model = Sequential([               
            tf.keras.Input(shape=(784,)), Dense(units=25, activation='relu'),
            Dense(units=15, activation='relu'), Dense(units=10, activation='linear'),
            ], name = "Neural")

    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

    x, y = load_data()
    history = model.fit(x, y, epochs=40)
    
    return model
#%% 
def image_processing(model, filename):
    image = Image.open(filename).convert('L')
    img = 255 - np.array(image)
    prediction = model.predict(img.reshape(1,784))  
    prediction_p = tf.nn.softmax(prediction)
    yhat = np.argmax(prediction_p)
    return yhat
