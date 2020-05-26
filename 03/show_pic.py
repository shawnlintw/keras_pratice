
import matplotlib.pyplot as plt
from matplotlib.image import imread

folder='dogs-vs-cats/train/'

for i in range(9):
	plt.subplot(330+1+i)
	filename = folder +'dog.'+ str(i)+'.jpg'
	image =imread(filename)
	plt.imshow(image)
plt.show()
