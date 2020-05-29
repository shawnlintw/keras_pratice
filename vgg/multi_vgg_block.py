from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D
from keras.utils import plot_model

# function for creating a vgg block
def vgg_block(layer_in, n_filters, n_conv):
	# add convolutional layers
	for _ in range(n_conv):
		layer_in = Conv2D(n_filters, (3,3), padding='same', activation='relu')(layer_in)
	# add max pooling layer
	layer_in = MaxPooling2D((2,2), strides=(2,2))(layer_in)
	return layer_in

# define model input 
visible = Input(shape=(256,256,3))
# add vgg module
layer = vgg_block(visible,64,2)
# add vgg module
layer = vgg_block(layer,128,2)
# add vgg module
layer = vgg_block(layer,256,4)
# create model
model = Model(inputs=visible, outputs=layer)
# summarize model
model.summary()
# plot model architecture
plot_model(model, show_shapes=True, to_file="multiple_vgg_blocks.png")
