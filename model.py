import os
import csv
from keras.layers import Lambda
from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.layers import Dropout
from keras.layers import Cropping2D
from math import ceil

samples = []
with open('video_unity/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
from sklearn.utils import shuffle
import cv2
import numpy as np
import sklearn

def generator(samples, batch_size=64):
    num_samples = len(samples)
    # the generator was needed due to high memory usage
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        batche = 0
        for offset in range(0, num_samples, batch_size):
            batche = batche + 1
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = batch_sample[0]
                left_name = batch_sample[1]
                right_name = batch_sample[2]
                
                correction = float(0.2)
		# center camera image is read
                center_image = cv2.imread(name)
                center_image_rgb = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
                center_angle = float(batch_sample[3])
                images.append(center_image_rgb)
                angles.append(center_angle)
		
		# center camera image flipped
                flipped_image = np.fliplr(center_image_rgb)
                flipped_angle = -center_angle
                images.append(flipped_image)
                angles.append(flipped_angle)
                
                # left camera image
                left_image = cv2.imread(left_name)
                left_image_rgb = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
                left_angle = float(center_angle + correction)
                images.append(left_image_rgb)
                angles.append(left_angle)
                
                # right camera image
                right_image = cv2.imread(right_name)
                right_image_rgb = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)
                right_angle = center_angle - correction
                images.append(right_image_rgb)
                angles.append(right_angle)
                
            
            X_train = np.array(images)
            y_train = np.array(angles)
	    # data shuffle
            yield sklearn.utils.shuffle(X_train, y_train)


batch_size=128

train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)


model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Conv2D(24,(5,5),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides = (2,2)))
model.add(Conv2D(36,(5,5),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides = (2,2)))
model.add(Conv2D(48,(5,5),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides = (2,2)))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides = (2,2)))
model.add(Flatten())
model.add(Dense(120))
model.add(Dropout(0.15))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator,
            steps_per_epoch=ceil(len(train_samples)/batch_size),
            validation_data=validation_generator,
            validation_steps=ceil(len(validation_samples)/batch_size),
            epochs=5, verbose=1)
model.save('model_all_4conv_RGB.h5')