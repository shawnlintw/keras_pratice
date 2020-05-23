#Reference : https://www.pyexercise.com/2019/01/bostonhousing.html
from keras.datasets import boston_housing
import numpy as np
(train_data,train_tragets), (test_data, test_tragets)=boston_housing.load_data()

import matplotlib.pyplot as plt

mean =train_data.mean(axis=0)
train_data -=mean
std = train_data.std(axis=0)
train_data /=std

test_data -= mean
test_data /= std


#x_axis = range(1, len(train_data[0])+1)
#y_axis = train_data[0]

#plt.plot(x_axis, y_axis, 'bo', label='normalize')
#plt.title('normalize data')
#plt.xlabel('X')
#plt.ylabel('Y')
#plt.legend()

#plt.show()

from keras import models, layers

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1))

model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

val_data= train_data[:100]
partial_train_data = train_data[100:]

val_tragets = train_tragets[:100]
partial_train_tragets= train_tragets[100:]

history = model.fit(partial_train_data,
		partial_train_tragets,
		validation_data= (val_data,val_tragets),
		epochs=200,
		batch_size=1,
		verbose=2)

loss= history.history['loss'][10:]
val_loss= history.history['val_loss'][10:]
epochs = range(1,len(loss)+1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label ='Vaildation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

mae_data= history.history['mae'][10:]
val_mae_data= history.history['val_mae'][10:]

plt.plot(epochs, mae_data, 'bo', label='mean_absolute_error')
plt.plot(epochs, val_mae_data, 'b', label ='Val_mean_absolute_error')
plt.title('Training and validation mean_absolute_error')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()
plt.show()


