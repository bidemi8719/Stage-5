# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 16:58:59 2020

@author:  Bidemi Adedokun, Hammed Ilupeju, Adetoye Abodunrin
"""

#import tensorflow as tf
from tensorflow import keras 
from keras import layers
from keras import models
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard
import time
import pickle

def CNN_Model_with_DataAug():
        
    pickle_in = open("X_train.pickle", "rb")
    X_train = pickle.load(pickle_in)
    
    pickle_in = open("X_test.pickle", "rb")
    X_test = pickle.load(pickle_in)
    
    pickle_in = open("y_train.pickle", "rb")
    y_train= pickle.load(pickle_in)
    
    pickle_in = open("y_test.pickle", "rb")
    y_test = pickle.load(pickle_in)
     
    # setup the data augmentation parameter      
    from keras.preprocessing.image import ImageDataGenerator    
    train_datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,)
    
    test_datagen = ImageDataGenerator()


    train_generator = train_datagen.fit(X_train)    
    # since we see binary_crossentropy loss, we need binary labels
    validation_generator = test_datagen.fit(X_test)
    
    
    img_size = 50
    for epochs in [15, 30]:
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu',
        input_shape=(img_size, img_size, 3)))        
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
        model.add(layers.Dense(1, activation='sigmoid'))
        
        print(model.summary()) 
        
        
        from keras import optimizers
        #model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
        model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])
                    
        model.fit_generator(train_datagen.flow(X_train, y_train, batch_size=32),
                           steps_per_epoch=len(X_train) / 32, epochs= epochs, validation_data = test_datagen.flow(X_test, y_test, batch_size = 20),
                           validation_steps=len(X_test) / 20)
    
    if epochs == 15:    
        # save the model
        model.save('CNN_DataAugmentation_Chile_Disease_vs_Normal_15')
    else:
         model.save('CNN_DataAugmentation_Chile_Disease_vs_Normal_30')
        
        
        
        