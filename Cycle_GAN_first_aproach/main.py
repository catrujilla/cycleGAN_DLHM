from os import listdir
import numpy as np
from keras_preprocessing.image import img_to_array
from keras_preprocessing.image import load_img
from matplotlib import pyplot as plt
from generator import define_generator  
from discriminator import discriminator
from compose_model import define_composite_model
from train import train

#load images
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
path = 'datas/'

# load dataset A - holograms..
dataA_all = load_images(path + 'holp/')
print('Loaded dataA: ', dataA_all.shape)

from sklearn.utils import resample
#subset of images
dataA = resample(dataA_all, 
                 replace=False,     
                 n_samples=2000,    
                 random_state=37) 

# load dataset B - reconstructions
dataB_all = load_images(path + 'reco/')
print('Loaded dataB: ', dataB_all.shape)
#subset of images
dataB = resample(dataB_all, 
                 replace=False,     
                 n_samples=2000,    
                 random_state=37) 

# plot source images
n_samples = 3
for i in range(n_samples):
	plt.subplot(2, n_samples, 1 + i)
	plt.axis('off')
	plt.imshow(dataA[i].astype('uint8'))
# plot target image
for i in range(n_samples):
	plt.subplot(2, n_samples, 1 + n_samples + i)
	plt.axis('off')
	plt.imshow(dataB[i].astype('uint8'))
plt.show()



# create dataset
data = [dataA, dataB]

print('Loaded', data[0].shape, data[1].shape)

#normalize the dataset in a range of 1 to -1 for the tahn activation 
def preprocess_data(data):
	
	X1, X2 = data[0], data[1]
	# scale from [0,255] to [-1,1]
	X1 = (X1 - 127.5) / 127.5
	X2 = (X2 - 127.5) / 127.5
	return [X1, X2]

dataset = preprocess_data(data)


# define input shape based on the loaded dataset
image_shape = dataset[0].shape[1:]
# generator: A -> B
g_model_AtoB = define_generator(image_shape)
# generator: B -> A
g_model_BtoA = define_generator(image_shape)
# discriminator: A 
d_model_A = discriminator(image_shape)
# discriminator: B 
d_model_B = discriminator(image_shape)
# composite: A -> B -> [real/fake, A]
c_model_AtoB = define_composite_model(g_model_AtoB, d_model_B, g_model_BtoA, image_shape)
# composite: B -> A -> [real/fake, B]
c_model_BtoA = define_composite_model(g_model_BtoA, d_model_A, g_model_AtoB, image_shape)

from datetime import datetime 
s = datetime.now() 
# train models
train(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA, dataset, epochs=12)

t = datetime.now()
#Execution time of the model 
execution_time = s-t
print("Execution time is: ", execution_time)

############################################

