import numpy as np
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.layers.experimental.preprocessing import CenterCrop, Rescaling


training_data=np.random.randint(0,256, size=(64,200,200, 3)).astype("float32")

cropper=CenterCrop(height=150, width=150)
scaler = Rescaling(scale=1.0/255)

output_data = scaler(cropper(training_data))
print("shape:", output_data.shape)
print("min:", np.min(output_data))
print("max:", np.max(output_data))

