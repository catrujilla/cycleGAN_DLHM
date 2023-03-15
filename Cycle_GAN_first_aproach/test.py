import tensorflow as tf 

if tf.test.gpu_device_name(): 

 print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
else:

   print("Please install GPU version of TF")