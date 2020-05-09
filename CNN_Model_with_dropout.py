# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 10:20:51 2020

@author: Bidemi Adedokun, Hammed Ilupeju, Adetoye Abodunrin
"""

import time
import tensorflow as tf
from tensorflow import keras 
from keras import layers
from keras import models
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Flatten, Convolution2D, MaxPooling2D
import pickle
from keras.optimizers import RMSprop



#===============================================================
def CNN_Model_with_dropout():
    
    pickle_in = open("X_train.pickle", "rb")
    X_train = pickle.load(pickle_in)
    
    pickle_in = open("X_test.pickle", "rb")
    X_test = pickle.load(pickle_in)
    
    pickle_in = open("y_train.pickle", "rb")
    y_train= pickle.load(pickle_in)
    
    pickle_in = open("y_test.pickle", "rb")
    y_test = pickle.load(pickle_in)
    
    train_datagen = ImageDataGenerator(horizontal_flip=True)
    test_datagen = ImageDataGenerator()
    
    #train_generator = train_datagen.fit(X_train)
    #validation_generator = test_datagen.fit(X_test)
             
    
    img_size = 50
    for epochs in [15, 30]:
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu',
        input_shape=(img_size, img_size, 3)))
        #model.add(Dropout(0.2, input_shape=(img_size, img_size, 3)))
        model.add(layers.Conv2D(32, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(layers.Dense(1, activation='sigmoid'))
        
        print(model.summary()) 
        
        from keras import optimizers
        model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])
        
        model.fit_generator(train_datagen.flow(X_train, y_train, batch_size=32),
                           steps_per_epoch=len(X_train) / 32, epochs= epochs, validation_data = test_datagen.flow(X_test, y_test, batch_size=20),
                           validation_steps=len(X_test) / 20)
    
    if epochs == 15:    
        # save the model
        model.save('CNN_dropout_Chile_Disease_vs_Normal_15')
    else:
         model.save('CNN_dropout_Chile_Disease_vs_Normal_30')



