import os
import csv
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, ELU
from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers import (
    Input,
    Dense,
    Flatten,
    merge,
    Lambda,
    Dropout,
    Activation
)
from keras.layers.convolutional import (
    Convolution2D,
    MaxPooling2D,
    AveragePooling2D,
    ZeroPadding2D,
    MaxPooling1D, 
    Convolution1D
)
samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        if(line[0]!="center"):
           samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import cv2
import numpy as np
import sklearn

ch, row, col = 3, 30, 100  # Trimmed image format
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, int(batch_size/6)):
            batch_samples = samples[offset:offset+int(batch_size/6)]
            images = []
            angles = []
            # Here I pass the data and do data augmentation
            # using left right images for recovery 
            # and flipped images
            for batch_sample in batch_samples:
                name = './data/IMG/'+batch_sample[0].split('/')[-1]
                name_l = './data/IMG/'+batch_sample[1].split('/')[-1]
                name_r = './data/IMG/'+batch_sample[2].split('/')[-1]
                center_image = cv2.imread(name)
                left_image = cv2.imread( name_l)
                right_image = cv2.imread( name_r)
                center_angle = float(batch_sample[3])
                center_image = center_image[30:130,0:320]
                center_image = cv2.resize(center_image,(64, 64), interpolation=cv2.INTER_AREA)  
                left_image = left_image[30:130,0:320]
                left_image = cv2.resize(left_image,(64, 64), interpolation=cv2.INTER_AREA)  
                right_image = right_image[30:130,0:320]
                right_image = cv2.resize(right_image,(64, 64), interpolation=cv2.INTER_AREA) 
                correction= 0.29
                steering_left = center_angle + correction
                steering_right = center_angle - correction
                images.append(center_image)
                angles.append(center_angle)
                images.append(left_image)
                angles.append(steering_left)
                images.append(right_image )
                angles.append(steering_right)
                f_center = cv2.flip(center_image,1)
                images.append(f_center)
                angles.append(-center_angle)
                f_left = cv2.flip(left_image,1)
                images.append(f_left)
                angles.append(-steering_left)
                f_right= cv2.flip(right_image,1)
                images.append(f_right)
                angles.append(-steering_right)
            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=60)
validation_generator = generator(validation_samples, batch_size=60)


#I use these function as my bases to build my inception plus a fully connected layer

def BNConv(nb_filter, nb_row, nb_col, w_decay, subsample=(1, 1), border_mode="same"):
    def f(input):
        conv = Convolution2D(nb_filter=nb_filter, nb_row=nb_row, nb_col=nb_col, subsample=subsample,
                      border_mode=border_mode)(input)
        BN = BatchNormalization(mode=0, axis=1)(conv)
        return ELU()(BN)
    return f


#Here I define My architecture. First of all I normalize the input,  
#Normalization of the input is feeded to
#a 3x3x12 CNN with batch normalization and ELU activation
#Then another  3x3x32 CNN with batch normalization and ELU activation
#Then Max pooling for downsampling is introduced with a 3x3 kernel and 2x2 strides
#Then another  3x3x64 CNN with batch normalization and ELU activation
#Then Max pooling for downsampling is introduced with a 3x3 kernel and 2x2 strides
#Following is a dropout layer 0.2
#Then the output is flatten and feeded to a feed forward layer
#with two hidden layers, with 300 and 60 neurons
#1 layer has dropout 0.2
#One last neuron for steering

def mymodel(w_decay=None):
    input = Input(shape=(64, 64,ch))
    #Normalize input
    inp_norm = Lambda(lambda x: (x / 255.0) - 0.5)(input)

    conv_1_3 = BNConv(12, 3, 3, w_decay, border_mode="valid")( inp_norm )
    conv_2_3 = BNConv(32, 3, 3, w_decay, border_mode="valid")(conv_1_3  )
    pool_2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode="valid")(conv_2_3)
    conv_3 = BNConv(64, 3, 3, w_decay, border_mode="valid")(pool_2 )
    pool_3 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode="valid")(conv_3)
    #dropout
    pool_3 = Dropout(0.2)(pool_3)
    #Fully connected
    x = Flatten()(pool_3)
    fully=Dense(300)(x)
    fully_act=Activation('relu')(fully)
    fully_act = Dropout(0.2)(fully_act)
    fully2=Dense(60)(fully_act)
    fully2_act=Activation('relu')(fully2)
    steering=Dense(1, name= 'Steering_Angle')(fully2_act)
 
    model = Model(input, steering)
 
    return model

model = mymodel() 

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch=6*len(train_samples), validation_data=validation_generator, nb_val_samples=6*len(validation_samples), nb_epoch=3)

model.save_weights("./model.h5", True)
model_json = model.to_json()
json_file=open("model.json", "w")
json_file.write(model_json)
