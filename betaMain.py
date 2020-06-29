from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from preprocessing import preprocessing
from matplotlib import pyplot as plt
from PIL import ImageFile

import pandas as pd
import numpy as np
import random

ImageFile.LOAD_TRUNCATED_IMAGES = True

def save_grafik(history):
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('static/accuracy_beta.png')
    plt.close()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('static/loss_beta.png')
    plt.close()

#Initializing CNN
def dCNN(train_dataset, test_dataset):
    classifier = Sequential()
    
    '''
    #START OF ALEXNET MODEL
    #END OF ALEXNET MODEL
    '''
    
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
    classifier.add(Dense(activation = 'softmax', units = 42))
    
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
    print(len(training_set),len(test_set))
    history_callback = classifier.fit_generator(training_set,
                                                steps_per_epoch = len(training_set),
                                                epochs = 100,
                                                validation_data = test_set,
                                                validation_steps = len(test_set))
    
    classifier.save('output/classifier_beta.h5')
    
    loss_history = history_callback.history["loss"]
    accuracy_history = history_callback.history["accuracy"]
    val_loss_history = history_callback.history["val_loss"]
    val_accuracy_history = history_callback.history["val_accuracy"]
    
    numpy_loss_history = numpy.array(loss_history)
    numpy.savetxt("output/loss_history_beta.txt", numpy_loss_history, delimiter=",")
    
    #return numpy_loss_history
    numpy_accuracy_history = numpy.array(accuracy_history)
    numpy.savetxt("output/acc_history_beta.txt", numpy_accuracy_history, delimiter=",")
    
    numpy_val_loss_history = numpy.array(val_loss_history)
    numpy.savetxt("output/val_loss_history_beta.txt", numpy_val_loss_history, delimiter=",")
    
    #return numpy_loss_history
    numpy_val_accuracy_history = numpy.array(val_accuracy_history)
    
    numpy.savetxt("output/val_acc_history_beta.txt", numpy_val_accuracy_history, delimiter=",")
    save_grafik(history=history_callback)
    
    return 'Successfull!'

model = None

def load_image(img_path, show=False):

    img = image.load_img(img_path, target_size=(256, 256))
    img = (np.asarray(img))
    # img /= 255 # imshow expects values in the range [0, 1]
    img_tensor = image.img_to_array(preprocessing(img))                    # (height, width, channels)
    # img_tensor = np.reshape(img_tensor, (1,64,64,1))
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    return img_tensor

def testing(filename):
    #define library
    # dimensions of our images
    img_width, img_height = 64, 64
    # load the model we saved
    global model
    if model is None:
        model = load_model('output/classifier_beta.h5')
        model.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])
                     
    #upload manually
    cncr, brn = 85, 89
    img_path = filename
    perct = random.randint(cncr,brn) 
    # load a single image

    
    new_image = load_image(img_path)
    
    # check prediction
    # pred = model.predict(new_image)

    # print(pred)
   
    #predict classes

    prediction = model.predict_classes(new_image)

    return prediction[0]

def predictData(path):
    
    data = pd.read_csv(path)
    
    result_label = []
    for item in data.values:
        label = testing('test/'+item[0])
        if label != None:
            result_label.append([item[0],label])
        print(item,label)
    df = pd.DataFrame(result_label,columns =['filename', 'category'])
    
    df.to_csv('output/result_beta.csv',index=False)
    return df

dCNN(train_dataset='./data/train',test_dataset='./data/validation')
predictData('test.csv')
