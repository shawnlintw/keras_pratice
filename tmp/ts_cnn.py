# Dataset saved in https://github.com/Abeni18/Keras-TensorFlow-backend-CNN-Finger-Counting-deep_learning.git
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential, load_model
import tensorflow as tf
import numpy as np
import os

import matplotlib
import matplotlib.pyplot as plt

plt.rcParams['axes.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14

nbatch =32

train_datagen = ImageDataGenerator(rescale=1./255,
		rotation_range=12.,
		width_shift_range=0.2,
		height_shift_range=0.2,
		zoom_range=0.15,
		horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
		'images/train/',
		target_size=(256,256),
		color_mode="grayscale",
		batch_size=nbatch,
		classes=['NONE','ONE','TWO','THREE','FOUR','FIVE'],
		class_mode='categorical'
		)
test_gen = test_datagen.flow_from_directory(
		'images/test/',
		target_size=(256,256),
		color_mode="grayscale",
		batch_size=nbatch,
		classes=['NONE','ONE','TWO','THREE','FOUR','FIVE'],
		class_mode='categorical'
		)

for x,y in train_gen:
	plt.figure(figsize=(10,10))
	for i in range(25):
		plt.subplot(5,5,i+1)
		plt.axis('off')
		plt.title('Label: %d' % np.argmax(y[i]))
		img =np.uint8(255*x[i,:,:,0])
		plt.imshow(img, cmap='gray')
	break
plt.pause(-1)
plt.close()

model = Sequential()

model.add(Conv2D(32, (3,3), activation='relu', input_shape=(256,256,1)))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(256, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))

model.add(Flatten())
model.add(Dense(150, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(6, activation='softmax'))

model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

callbacks_list =[
		EarlyStopping(monitor='val_loss', patience=10),
		ModelCheckpoint(filepath='model_6cat_2.h6', monitor='val_loss', save_best_only=True)]

history = model.fit_generator(
		train_gen,
		steps_per_epoch=64,
		epochs=5,
		validation_data=test_gen,
		validation_steps=28,
		callbacks=callbacks_list
		)


plt.figure(figsize=(10,3))
plt.subplot(1,2,1)
nepochs=len(history.history['loss'])
plt.plot(range(nepochs), history.history['loss'], 'g-', label='train')
plt.plot(range(nepochs), history.history['val_loss'], 'c-', label='test')

plt.legend(prop={'size':20})
plt.ylabel('loss')
plt.xlabel('number of epochs')

plt.subplot(1,2,2)
plt.plot(range(nepochs), history.history['accuracy'], 'g-', label='train')
plt.plot(range(nepochs), history.history['val_accuracy'], 'c-', label='test')
plt.legend(prop={'size': 20})
plt.ylabel('accuracy')
plt.xlabel('number of epochs')

plt.pause(-1)
plt.close()


