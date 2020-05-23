import numpy as np
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.layers.experimental.preprocessing import Normalization

training_data = np.random.randint(0,256, size=(64,200,200,3)).astype("float32")

normalizer = Normalization(axis=-1)
normalizer.adapt(training_data)

normalized_data = normalizer(training_data)

print("var: %.4f" % np.var(normalized_data))
print("mean: %.4f" % np.mean(normalized_data))

