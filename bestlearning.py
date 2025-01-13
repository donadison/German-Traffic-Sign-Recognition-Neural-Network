#uczenie za pomocą danych, wzorowane na oryginalnym sposobie proponowanym przez autora w gicie

import keras
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing import image
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout,Activation
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from PIL import Image
cur_path = r"C:\\Users\\Adrian\\.cache\\kagglehub\\datasets\\meowmeowmeowmeowmeow\\gtsrb-german-traffic-sign\\versions\\1"
data = []
labels = []
classes = 43
 

#Retrieving the images and their labels 
for i in range(classes):
    path = os.path.join(cur_path,'train',str(i))
    images = os.listdir(path)

    for a in images:
        try:
            image = Image.open(path + '\\'+ a)
            image = image.resize((30,30))
            image = np.array(image)
    
            #data.append(image)
            #labels.append(i)
            data.append([image,i]) #appending all value together 
        except:
            print("Error loading image")
random.shuffle(data)
print(len(data))
x = []
y = []

for features,label in data:
    x.append(features)
    y.append(label)
#Converting lists into numpy arrays
x = np.array(x)
y = np.array(y)
X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)
X_train = X_train/255.0
X_val = X_val/255.0
#x = np.array(x).reshape(-1, 30, 30, 1) 
print("Shape of train images is:", X_train.shape)
print("Shape of labels is:", y_train.shape)
#Building the model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(30, 30, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(43))
model.add(Activation('softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
aug = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.15,
    horizontal_flip=False,
    vertical_flip=False,
    fill_mode="nearest")

model.fit(aug.flow(X_train, y_train, batch_size=32), epochs=15, validation_data=(X_val, y_val))

model.save('newmodel.h5')  # always save your weights after training or during training
