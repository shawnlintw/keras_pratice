import numpy as np
import pandas as pd
from keras.utils import np_utils
np.random.seed(10)
from keras.datasets import mnist

(x_train_image, y_train_label), (x_test_image, y_test_label)= mnist.load_data()

import matplotlib.pyplot as plt

def plot_image(image):
	fig=plt.gcf()
	fig.set_size_inches(2,2)
	plt.imshow(image, cmap='binary')
	plt.show()

#plot_image(x_train_image[1])

def plot_image_labels(images, labels, idx, num=10):
	fig = plt.gcf()
	fig.set_size_inches(12,14)
	if num>25: num=25
	for i in range(0,num):
		ax=plt.subplot(5,5,1+i)
		ax.imshow(images[idx], cmap='binary')
		title = "label=" +str(labels[idx])
		ax.set_title(title, fontsize=10)
		ax.set_xticks([])
		ax.set_yticks([])
		idx+=1
	plt.show()

#plot_image_labels(x_train_image, y_train_label, idx=0)
#plot_image_labels(x_test_image,  y_test_label,  idx=0)

x_Train = x_train_image.reshape(60000,784).astype('float32')
x_Test = x_test_image.reshape(10000, 784).astype('float32')
x_Train_normalize=x_Train/255
x_Test_normalize = x_Test/255

y_Train_OneHot= np_utils.to_categorical(y_train_label)
y_Test_OneHot=np_utils.to_categorical(y_test_label)

from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
model.add(Dense(units=256, input_dim=784, kernel_initializer='normal',activation='relu'))
model.add(Dense(units=10, kernel_initializer='normal', activation='softmax'))
print(model.summary())

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

train_history=model.fit(x=x_Train_normalize, y=y_Train_OneHot, validation_split=0.2, epochs=10, batch_size=200, verbose=1)

def show_train_history(train_history, train, validation):
	plt.plot(train_history.history[train])
	plt.plot(train_history.history[validation])
	plt.title('Train History')
	plt.ylabel(train)
	plt.xlabel('Epoch')
	plt.legend(['train','validation'], loc='upper left')
	plt.show()

#show_train_history(train_history, 'accuracy', 'val_accuracy')
#show_train_history(train_history, 'loss', 'val_loss')

scores = model.evaluate(x_Test_normalize, y_Test_OneHot)
print('\naccuracy=',scores[1])

def plot_images_labels_prediction(images,labels, prediction, idx, num=10):
	fig=plt.gcf()
	fig.set_size_inches(6,7)
	if num>25 : num=25
	for i in range(0,num):
		ax=plt.subplot(5,5, 1+i)
		ax.imshow(images[idx], cmap='binary')
		title = 'label='+str(labels[idx])
		if len(prediction)>0:
			title+= " ,predict=" +str(prediction[idx])

		ax.set_title(title, fontsize=10)
		ax.set_xticks([])
		ax.set_yticks([])
		idx+=1
	plt.show()

prediction = model.predict_classes(x_Test)
#plot_images_labels_prediction(x_test_image, y_test_label, prediction, idx=340)

import pandas as pd
confucmatrix=pd.crosstab(y_test_label, prediction, rownames=['label'], colnames=['predict'])
print(confucmatrix)

df= pd.DataFrame({'label':y_test_label, 'predict':prediction})
df[:2]
print(df)
#df[(df.label==5)&(df.predict==3)]
#plot_images_labels_prediction(x_test_image, y_test_label, prediction, idx=340, num=1)
