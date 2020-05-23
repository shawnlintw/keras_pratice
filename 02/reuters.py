# Reference : https://www.pyexercise.com/2019/01/reuters.html
import numpy as np
from keras.datasets import reuters

(train_data, train_labels), (test_data, test_labels)=reuters.load_data(num_words=10000)

#print(len(train_data))

def vector_sequences(sequences, dimension=10000):
	results = np.zeros((len(sequences),dimension))
	for i, sequence in enumerate(sequences):
		results[i,sequence]=1.
	return results

def to_one_hot(labels, dimension=46):
	results= np.zeros((len(labels),dimension))
	for i, label in enumerate(labels):
		results[i,label]=1.
	return results


x_train = vector_sequences(train_data)
x_test  = vector_sequences(test_data)
one_hot_train_labels = to_one_hot(train_labels)
one_hot_test_labels  = to_one_hot(test_labels)

from keras import models, layers

model=models.Sequential()
model.add(layers.Dense(64,activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(46,activation='softmax'))

model.compile(optimizer='rmsprop',
		loss='categorical_crossentropy',
		metrics=['accuracy'])

x_val= x_train[:1000]
partial_x_train=x_train[1000:]

y_val=one_hot_train_labels[:1000]
partial_y_train=one_hot_train_labels[1000:]

history=model.fit(partial_x_train,
		partial_y_train,
		epochs=20,
		batch_size=512,
		validation_data=(x_val,y_val))


import matplotlib.pyplot as plt
print(one_hot_test_labels[0])
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs= range(1,len(loss)+1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.clf()

accuracy = history.history['accuracy']
val_accuracy=history.history['val_accuracy']

plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

print(min(val_loss))
print(max(val_accuracy))

