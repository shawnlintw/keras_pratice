import numpy as np
from keras.datasets import cifar10
from keras.utils import np_utils
np.random.seed(10)

#----- dataset perpare ------
(X_img_train, y_label_train), (X_img_test, y_label_test)=cifar10.load_data()
print('train data:','\timages:',X_img_train.shape, '\tlabels:', y_label_train.shape)
print('test data: ','\timages:',X_img_test.shape,'\tlabels:', y_label_test.shape)

X_img_train_normalize = X_img_train.astype('float32')/255.0
X_img_test_normalize = X_img_test.astype('float32')/255.0

y_label_train_OneHot = np_utils.to_categorical(y_label_train)
y_label_test_OneHot = np_utils.to_categorical(y_label_test)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D


model = Sequential()

model.add(Conv2D(filters=32,
	kernel_size=(3,3),
	input_shape=(32,32,3),
	activation = 'relu',
	padding='same'))

model.add(Dropout(0.25))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=64,
	kernel_size=(3,3),
	activation='relu',
	padding='same'))

model.add(Dropout(0.25))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=128,
	kernel_size=(3,3),
	activation='relu',
	padding='same'))

model.add(Dropout(0.25))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=256,
	kernel_size=(3,3),
	activation='relu',
	padding='same'))

model.add(Dropout(0.25))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Flatten())
model.add(Dropout(0.25))

model.add(Dense(2500,activation='relu'))
model.add(Dropout(0.25))

model.add(Dense(1250,activation='relu'))
model.add(Dropout(0.25))

model.add(Dense(625, activation='relu'))
model.add(Dropout(0.25))

model.add(Dense(10, activation='softmax'))

print(model.summary())

import matplotlib.pyplot as plt
def show_train_history(train_acc,test_acc):
	plt.plot(train_history.history[train_acc])
	plt.plot(train_history.history[test_acc])
	plt.title('Train History')
	plt.ylabel('Accuracy')
	plt.xlabel('Epoch')
	plt.legend(['train','test'], loc='upper left')
	plt.show()

model.compile(loss='categorical_crossentropy', optimizer='adam',
		metrics=['accuracy'])
try:
	model.load_weights('cifar10CNNModel_2.h5')
#	train_history=model.fit(X_img_train_normalize, y_label_train_OneHot,
#		validation_split=0.2,
#		epochs=10,
#		batch_size=100,
#		verbose=1)
#	show_train_history('accuracy', 'val_accuracy')
#	show_train_history('loss','val_loss')
except:
	train_history=model.fit(X_img_train_normalize, y_label_train_OneHot,
		validation_split=0.2,
		epochs=10,
		batch_size=100,
		verbose=1)
	show_train_history('accuracy', 'val_accuracy')
	show_train_history('loss','val_loss')

scores=model.evaluate(X_img_test_normalize, y_label_test_OneHot, verbose=1)
print("\naccuracy: ",scores[1])

model.save_weights("cifar10CNNModel_2.h5")
print('save model to disk')

label_dict={0:"airplane", 
		1:"automobile",
		2:"bird",
		3:"cat",
		4:"deer",
		5:"dog",
		6:"frog",
		7:"horse",
		8:"ship",
		9:"truck"}

import matplotlib.pyplot as plt

def plot_images_labels_prediction(images, labels, prediction, idx, num=10):
	fig=plt.gcf()
	fig.set_size_inches(12,14)
	if num>25: num=25
	for i in range(0,num):
		ax=plt.subplot(5,5,1+i)
		ax.imshow(images[idx], cmap='binary')

		title=str(i)+',' + label_dict[labels[i][0]]
		if len(prediction)>0:
			title+='=>'+label_dict[prediction[i]]
		ax.set_title(title, fontsize=10)
		ax.set_xticks([])
		ax.set_yticks([])
		idx+=1
	plt.show()

prediction=model.predict_classes(X_img_test_normalize)
prediction[:10]

plot_images_labels_prediction(X_img_test_normalize, y_label_test, prediction, 0, 10)

def show_Predicted_Probability(X_img, Predicted_Probability, i):
	plt.figure(figsize=(2,2))
	plt.imshow(np.reshape(X_img_test[i],(32,32,3)))
	plt.show()
	print('\n-----------------------')
	for j in range(10):
		print(label_dict[j]+'Probability:%1.9f'%(Predicted_Probability[i][j]))

Predicted_Probability=model.predict(X_img_test_normalize)

show_Predicted_Probability(X_img_test, Predicted_Probability,0)
show_Predicted_Probability(X_img_test, Predicted_Probability,3)

import pandas as pd

print(label_dict)
confumatrix=pd.crosstab(y_label_test.reshape(-1), prediction, rownames=['label'], colnames=['predict'])
print(confumatrix)


