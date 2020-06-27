from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from flask import render_template
from keras import optimizers

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from preprocessing_function import preprocessing
from keras.preprocessing.image import ImageDataGenerator

import numpy

import matplotlib.pyplot as plt

def save_grafik(history):
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('static/accuracy.png')
    plt.close()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('static/loss.png')
    plt.close()

#Initializing CNN
def dCNN(train_dataset, test_dataset):
	classifier = Sequential()
	
	'''
	#START OF ALEXNET MODEL

	
	#1-Convolution
	classifier.add(Conv2D(96, (11,11), strides=(4,4), input_shape = (256,256,3), activation = 'relu'))
	#2-Pooling
	classifier.add(MaxPooling2D(pool_size = (2,2)))
	#add second conv
	classifier.add(Conv2D(256, (5,5), strides=(1,1), activation = 'relu'))
	classifier.add(MaxPooling2D(pool_size = (2,2)))
	#add third conv
	classifier.add(Conv2D(384, (3,3), strides=(1,1), activation = 'relu'))
	
	#add 4th conv
	classifier.add(Conv2D(384, (3,3), strides=(1,1), activation = 'relu'))
	
	#add 5th conv
	classifier.add(Conv2D(256, (3,3), strides=(1,1), activation = 'relu'))
	classifier.add(MaxPooling2D(pool_size = (2,2)))
	#3-Flattening
	classifier.add(Flatten())
	#4-Full Connection
	classifier.add(Dense(activation = 'relu', units = 900))
	classifier.add(Dense(activation = 'relu', units = 90))
	classifier.add(Dense(activation = 'softmax', units = 3))

	#END OF ALEXNET MODEL
	'''

	#'''
	#START OF MY MODEL

	#1-Convolution
	classifier.add(Conv2D(32, (3,3), strides=(1,1), input_shape = (256,256,3), activation = 'relu'))
	#2-Pooling
	classifier.add(MaxPooling2D(pool_size = (2,2)))
	#add second conv
	classifier.add(Conv2D(32, (3,3), strides=(1,1), activation = 'relu'))
	classifier.add(MaxPooling2D(pool_size = (2,2)))
	#add third conv
	classifier.add(Conv2D(32, (3,3), strides=(1,1), activation = 'relu'))
	classifier.add(MaxPooling2D(pool_size = (2,2)))
	#add third conv
	classifier.add(Conv2D(64, (3,3), strides=(1,1), activation = 'relu'))
	classifier.add(MaxPooling2D(pool_size = (2,2)))
	#add third conv
	classifier.add(Conv2D(64, (3,3), strides=(1,1), activation = 'relu'))
	classifier.add(MaxPooling2D(pool_size = (2,2)))
	#3-Flattening
	classifier.add(Flatten())
	#4-Full Connection
	classifier.add(Dense(activation = 'relu', units = 900))
	classifier.add(Dense(activation = 'relu', units = 90))
	classifier.add(Dense(activation = 'softmax', units = 4))
	
	#END OF MY MODEL
	#'''



	#Compiling CNN
	rmsprop = optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=1e-08, decay=0.0)
	classifier.compile(optimizer = rmsprop, loss = 'categorical_crossentropy', metrics = ['accuracy'])

	
	train_datagen = ImageDataGenerator(rescale = 1./255,
	                                   shear_range = 0.2,
	                                   zoom_range = 0.2,
	                                   horizontal_flip = True,
	                                   preprocessing_function=preprocessing)

	test_datagen = ImageDataGenerator(rescale = 1./255,
	                                  shear_range = 0.2,
	                                  zoom_range = 0.2,
	                                  horizontal_flip = True,
	                                  preprocessing_function=preprocessing)

	training_set = train_datagen.flow_from_directory(train_dataset,
	                                                 target_size = (256, 256),
	                                                 batch_size = 32,
	                                                 class_mode = 'categorical')

	test_set = test_datagen.flow_from_directory(test_dataset,
	                                            target_size = (256, 256),
	                                            batch_size = 32,
	                                            class_mode = 'categorical')

	history_callback = classifier.fit_generator(training_set,
	                         steps_per_epoch = (1200/32),
	                         epochs = 100,
	                         validation_data = test_set,
	                         validation_steps = (300/32))

	classifier.save('output/classifier.h5')

	loss_history = history_callback.history["loss"]
	accuracy_history = history_callback.history["acc"]
	val_loss_history = history_callback.history["val_loss"]
	val_accuracy_history = history_callback.history["val_acc"]

	numpy_loss_history = numpy.array(loss_history)
	numpy.savetxt("output/loss_history.txt", numpy_loss_history, delimiter=",")
	#return numpy_loss_history
	numpy_accuracy_history = numpy.array(accuracy_history)
	numpy.savetxt("output/acc_history.txt", numpy_accuracy_history, delimiter=",")

	numpy_val_loss_history = numpy.array(val_loss_history)
	numpy.savetxt("output/val_loss_history.txt", numpy_val_loss_history, delimiter=",")
	#return numpy_loss_history
	numpy_val_accuracy_history = numpy.array(val_accuracy_history)
	numpy.savetxt("output/val_acc_history.txt", numpy_val_accuracy_history, delimiter=",")

	save_grafik(history=history_callback)

	return 'OK'

