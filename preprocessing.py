# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 10:20:34 2020

@author: TEMITAYO
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator



DATAdir = '.../All_chili_data' #'C:/Users/TEMITAYO/Pictures/All_chili_data' # #'C:/Users/TEMITAYO/Pictures/dddd'
img_size = 50

CATEGORIES = ["disease", "normal"]

#for category in CATEGORIES :
path = os.path.join(DATAdir) # path to normal and disease
for img in os.listdir(path):
    img_array = cv2.imread(os.path.join(path, img))
    new_array = cv2.resize(img_array, (img_size, img_size))
    #img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
    plt.imshow(new_array)
    plt.show()
    break
#break
    

training_data = []

def create_training_data():
    for category in CATEGORIES :
        path = os.path.join(DATAdir, category) # path to normal and disease
        class_num =CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img))
                #img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (img_size, img_size))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass
        
       
create_training_data()

print(len(training_data))


import random
random.shuffle(training_data)

x = []
y = []


for features, lable in training_data:
    x.append(features)
    y.append(lable)
    
X = np.array(x).reshape(-1, img_size, img_size, 3)


print(y[0:10])
    
    
import keras.backend
print('The Baackend is: ', keras.backend.backend())

print('The variable type of data is:', X.dtype)
print('the shape of X is: ', X.shape)

X = X/255.0
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

train_datagen = ImageDataGenerator(horizontal_flip=True)
test_datagen = ImageDataGenerator()

train_generator = train_datagen.fit(X_train)
validation_generator = test_datagen.fit(X_test)

pickle_out = open("X_train.pickle", "wb")
pickle.dump(X_train, pickle_out)
pickle_out.close()

pickle_out = open("X_test.pickle", "wb")
pickle.dump(X_test, pickle_out)
pickle_out.close()


pickle_out = open("y_train.pickle", "wb")
pickle.dump(y_train, pickle_out)
pickle_out.close()

pickle_out = open("y_test.pickle", "wb")
pickle.dump(y_test, pickle_out)
pickle_out.close()


# for PCA
from sklearn.decomposition import PCA    
#############################################################
num_samples = X_train.shape[0]
X_train = X_train.reshape(num_samples, -1)

num_samples = X_test.shape[0]
X_test = X_test.reshape(num_samples, -1)
     
pca= PCA(n_components=4)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.fit_transform(X_test)
            

#saving whAT we have done

pickle_out = open("X_train_pca.pickle", "wb")
pickle.dump(X_train_pca, pickle_out)
pickle_out.close()

pickle_out = open("X_test_pca.pickle", "wb")
pickle.dump(X_test_pca, pickle_out)
pickle_out.close()

