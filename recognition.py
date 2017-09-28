#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 10:26:54 2017
License: MIT
@author: easton
"""

import points
import numpy as np
import keras
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization

IMG_SIZE = 28
subdir = './Shapes'
features, images, names = points.process_image(subdir, IMG_SIZE)

le = preprocessing.LabelEncoder()
names_encoded = le.fit_transform(names)
num_classes = len(le.classes_)

# Train test split and some preprocessing
x_train, x_test, y_train, y_test = train_test_split(
        features, names_encoded, test_size=0.2, random_state=42
        )
x_train_i, x_test_i, y_train_i, y_test_i = train_test_split( # i: image
        images, names_encoded, test_size=0.2, random_state=42
        )
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
y_train_i = keras.utils.to_categorical(y_train_i, num_classes)
y_test_i = keras.utils.to_categorical(y_test_i, num_classes)

img_rows, img_cols = (IMG_SIZE, IMG_SIZE)
input_shape_i = (img_rows, img_cols, 1)
x_train_i = x_train_i.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test_i = x_test_i.reshape(x_test.shape[0], img_rows, img_cols, 1)

# Neural network with calculated attributes (angles, sides, etc.)
model = Sequential()
model.add(Dense(100, activation='relu', input_dim=18))
model.add(Dropout(0.3))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=32, epochs=2000, verbose=1,
          validation_data=(x_test, y_test))

# Convolutional neural network with just image input
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',
                 input_shape=input_shape_i))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
model.fit(x_train_i, y_train_i, batch_size=32, epochs=100, verbose=1, 
          validation_data=(x_test_i, y_test_i))

# Combined model with image data and calculated features

# Image input
main_input = Input(shape=(28,28, 1), name='main_input')
x = Conv2D(32, kernel_size=(3, 3), activation='relu')(main_input)
x = Conv2D(32, kernel_size=(3, 3), activation='relu')(x)
x = MaxPooling2D(pool_size=(2,2))(x)
x= Conv2D(64, kernel_size=(3,3), activation='relu')(x)
x= Conv2D(64, kernel_size=(3,3), activation='relu')(x)
x= MaxPooling2D(pool_size=(2,2))(x)
x = Dropout(0.2)(x)
x = Flatten()(x)
# This will be concatenated with the other inputs
conv_out = BatchNormalization(name = 'conv_out')(x)
# Auxiliary output to help with training
conv_aux = Dense(256, activation='relu')(conv_out)
conv_aux = Dense(128, activation='relu')(conv_aux)
conv_aux = Dropout(0.5)(conv_aux)
aux_output = Dense(num_classes, activation='softmax', 
                   name = 'aux_output')(conv_aux)
# Auxiliary input (calculated features)
aux_input = Input(shape=(18,), name='aux_input')
aux = Dense(100, activation='relu')(aux_input)
aux = Dense(100, activation='relu')(aux)
aux = Dropout(0.2)(aux)
# Concatenate inputs
x = keras.layers.concatenate([conv_out, aux])
# Dense layers on top of other inputs
x = Dense(256, activation='relu')(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
main_output = Dense(num_classes, activation='softmax', name='main_output')(x)
# Compile and train
model = Model(inputs=[main_input, aux_input], outputs=[main_output, aux_output])
model.compile(optimizer='adadelta', loss='categorical_crossentropy',
              loss_weights=[1., 0.2], metrics=['accuracy'])
model.fit([x_train_i, x_train], [y_train, y_train],
          epochs=100, batch_size=32, 
          validation_data=([x_test_i, x_test], [y_test_i, y_test]))

# Same idea, but using calculated features as primary input
# This model doesn't do quite as well as the previous one

# Angles and side lengths
main_input = Input(shape=(18,), name='main_input')
main = Dense(100, activation='relu')(main_input)
main = Dense(100, activation='relu')(main)
main_out = Dropout(0.2)(main)
aux_output = Dense(num_classes, activation='softmax', name = 'aux_output')(main_out)
# Image input
aux_input = Input(shape=(28,28, 1), name='aux_input')
aux = Conv2D(32, kernel_size=(3, 3), activation='relu')(aux_input)
aux = Conv2D(32, kernel_size=(3, 3), activation='relu')(aux)
aux = MaxPooling2D(pool_size=(2,2))(aux)
aux= Conv2D(64, kernel_size=(3,3), activation='relu')(aux)
aux= Conv2D(64, kernel_size=(3,3), activation='relu')(aux)
aux= MaxPooling2D(pool_size=(2,2))(aux)
aux = Dropout(0.2)(aux)
aux = Flatten()(aux)
aux_out = BatchNormalization(name = 'aux_out')(aux)
# Concatenate secondary input
main = keras.layers.concatenate([main_out, aux_out])
# Dense layers
main = Dense(256, activation='relu')(main)
main = Dense(128, activation='relu')(main)
main = Dropout(0.5)(main)
main_output = Dense(num_classes, activation='softmax', name='main_output')(main)
# Compile and train
model = Model(inputs=[main_input, aux_input], outputs=[main_output, aux_output])
model.compile(optimizer='adadelta', loss='categorical_crossentropy',
              loss_weights=[1., 0.2], metrics=['accuracy'])
model.fit([x_train, x_train_i], [y_train, y_train_i],
          epochs=100, batch_size=32, 
          validation_data=([x_test, x_test_i], [y_test, y_test_i]))