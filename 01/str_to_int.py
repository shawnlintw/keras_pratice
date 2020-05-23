import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

training_data = np.array([["This is ths 1st sample."],["And here's the 2nd sample."]])
vectorizer=TextVectorization(output_mode="int")
vectorizer.adapt(training_data)
integer_data= vectorizer(training_data)
print(integer_data)
