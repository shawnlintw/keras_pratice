# vgg16 model
import sys
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

# define cnn model
def define_model():
	model =VGG16(include_top=False, input_shape=(224,224,3))
	# mark loaded layers as not trainable
	for layer in model.layers:
		layer.trainable=False
	# add new classifire layers
	flat1 = Flatten()(model.layers[-1].output)
	class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
	output = Dense(1, activation='sigmoid')(class1)
	# define new model
	model = Model(inputs=model.inputs, outputs=output)

	# compile model
	opt = SGD(lr=0.001, momentum=0.9)
	model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
	return model

# plot diagnostic learning curvers
def summarize_diagnostics(history):
	# plot loss
	plt.subplot(211)
	plt.title('Cross Entropy Loss')
	plt.plot(history.history['loss'], color='blue', label='train')
	plt.plot(history.history['val_loss'], color='orange', label='test')
	# plot accuracy
	plt.subplot(212)
	plt.title('Classification Accuracy')
	plt.plot(history.history['accuracy'], color='blue', label='train')
	plt.plot(history.history['val_accuracy'], color='orange', label='test')
	# save plot to file
	filename = sys.argv[0].split('/')[-1]
	plt.savefig(filename + '_plot.png')
	plt.close()

# run the test harness for evaluating a model
def run_test_harness():
	model= define_model()
	# create data generator
	datagen = ImageDataGenerator(featurewise_center=True)
	# specify imagenet mean values for centering
	'''
	The model also expects images to be centered. 
	That is, to have the mean pixel values from each channel (red, green, and blue) as calculated on the ImageNet training dataset subtracted from the input. 
	Keras provides a function to perform this preparation for individual photos via the preprocess_input() function. 
	Nevertheless, we can achieve the same effect with the ImageDataGenerator by setting the “featurewise_center” argument to “True” 
	and manually specifying the mean pixel values to use when centering as the mean values from the ImageNet training dataset: [123.68, 116.779, 103.939]
	'''
	datagen.mean=[123.68, 116.779, 103.939]

	# prepare iterators
	train_it = datagen.flow_from_directory('dataset_dogs_vs_cats/train/',
			class_mode='binary',
			batch_size=64,
			target_size=(224,224))
	test_it = datagen.flow_from_directory('dataset_dogs_vs_cats/test/',
			class_mode='binary',
			batch_size=64,
			target_size=(224,224))
	# fit model
	history = model.fit_generator(train_it, steps_per_epoch=len(train_it),
			validation_data= test_it,
			validation_steps=len(test_it),
			epochs=10,
			verbose=1)
	_, acc= model.evaluate_generator(test_it, steps=len(test_it),verbose=1)
	print('> %.3f' % (acc * 100.0))
	summarize_diagnostics(history)
run_test_harness()
