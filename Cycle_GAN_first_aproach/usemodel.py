# Use the saved cyclegan models for image translation
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.models import load_model
from matplotlib import pyplot
from numpy.random import randint
from sklearn.utils import resample
from os import listdir
import numpy as np
from keras_preprocessing.image import img_to_array
from keras_preprocessing.image import load_img
from matplotlib import pyplot as plt





def load_images(path, size=(256,256)):
	data_list = list()
	# enumerate filenames in directory, assume all are images
	for filename in listdir(path):
		# load and resize the image
		pixels = load_img(path + filename, target_size=size)
		# convert to numpy array
		pixels = img_to_array(pixels)
		# store
		data_list.append(pixels)
	return np.asarray(data_list)


# dataset path

# select a random sample of images from the dataset
def select_sample(dataset, n_samples):
	# choose random instances
	ix = randint(0, dataset.shape[0], n_samples)
	# retrieve selected images
	X = dataset[ix]
	return X

# plot the image, its translation, and the reconstruction
def show_plot(imagesX, imagesY1, imagesY2):
	images = np.vstack((imagesX, imagesY1, imagesY2))
	titles = ['Hologram', 'Generated', 'Real']
	# scale from [-1,1] to [0,1]
	images = (images + 1) / 2.0
	# plot images row by row
	for i in range(len(images)):
		# define subplot
		pyplot.subplot(1, len(images), 1 + i)
		# turn off axis
		pyplot.axis('off')
		# plot raw pixel data
		pyplot.imshow(images[i])
		# title
		pyplot.title(titles[i])
	pyplot.show()


path = 'datas/'

# load dataset A - holograms..
dataA_all = load_images(path + 'holp/')
print('Loaded dataA: ', dataA_all.shape)	

# load dataset B - Photos 
dataB_all = load_images(path + 'reco/')
print('Loaded dataB: ', dataB_all.shape)

# load dataset
A_data = resample(dataA_all, 
                 replace=False,     
                 n_samples=50,    
                 random_state=50) # reproducible results

B_data = resample(dataB_all, 
                 replace=False,     
                 n_samples=50,    
                 random_state=50) # reproducible results

A_data = (A_data - 127.5) / 127.5
B_data = (B_data - 127.5) / 127.5


# load the models
cust = {'InstanceNormalization': InstanceNormalization}
model_AtoB = load_model('g_model_AtoB_022000.h5', cust)
model_BtoA = load_model('g_model_BtoA_022000.h5', cust)

# plot A->B->A (Monet to photo to Monet)
#A_real = select_sample(A_data, 1)
#B_generated  = model_AtoB.predict(A_real)
#A_reconstructed = model_BtoA.predict(B_generated)
#show_plot(A_real, B_generated, A_reconstructed)
# plot B->A->B (Photo to Monet to Photo)
#B_real = select_sample(B_data, 1)
#A_generated  = model_BtoA.predict(B_real)
#B_reconstructed = model_AtoB.predict(A_generated)
#show_plot(B_real, A_generated, B_reconstructed)

##########################
#Load a single custom image
test_image = load_img('datas/holp/D1_645.tif',target_size=(256,256))
test_image = img_to_array(test_image)
test_image_input = np.array([test_image])  # Convert single image to a batch.
test_image_input = (test_image_input - 127.5) / 127.5

test_image2 = load_img('datas/reco/D1_645.tif',target_size=(256,256))
test_image2 = img_to_array(test_image2)
test_image_input2 = np.array([test_image2])  # Convert single image to a batch.
test_image_input2 = (test_image_input2 - 127.5) / 127.5


# plot A->B->A (Photo to Monet to Photo)
holo_generated  =model_AtoB.predict(test_image_input)
photo_reconstructed = model_BtoA.predict(holo_generated)
show_plot(test_image_input, holo_generated, test_image_input2)

 