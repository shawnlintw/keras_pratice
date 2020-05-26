# Reference : https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-to-classify-photos-of-dogs-and-cats/
from os import listdir
from numpy import asarray
from numpy import save

from keras.preprocessing.image import load_img, img_to_array

folder = 'dogs-vs-cats/train/'
photos, labels =list(), list()

for file in listdir(folder):
	# determine class
	output=0.0
	if file.startswith('cat'):
		output=1.0
	photo = load_img(folder+file, target_size=(200,200))
	# convert image to numpy array
	photo = img_to_array(photo)
	# store
	photos.append(photo)
	labels.append(output)

# convert to a numpy arrays
photos = asarray(photos)
labels = asarray(labels)
print(photos.shape, labels.shape)
# save the reshape photos
save('dogs-vs-cats_photos.npy',photos)
save('dogs-vs-cats_labels.npy',labels)

