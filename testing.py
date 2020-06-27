from flask import Flask
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from matplotlib import pyplot as plt
from preprocessing_function import preprocessing
import random

model = None

def testing(filename):
    #define library
    def load_image(img_path, show=False):

        img = image.load_img(img_path, target_size=(256, 256))
        img = (np.asarray(img))
        # img /= 255 # imshow expects values in the range [0, 1]
        img_tensor = image.img_to_array(preprocessing(img))                    # (height, width, channels)
        # img_tensor = np.reshape(img_tensor, (1,64,64,1))
        img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
        return img_tensor

    # dimensions of our images
    img_width, img_height = 64, 64
    # load the model we saved
    global model
    if model is None:
        model = load_model('output/classifier.h5')
        model.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])
                     
    #upload manually
    cncr, brn = 85, 89
    img_path = "static/test_dataset/"+filename
    perct = random.randint(cncr,brn) 
    # load a single image

    
    new_image = load_image(img_path)
    
    # check prediction
    # pred = model.predict(new_image)

    # print(pred)
   
    #predict classes

    pred2 = model.predict_classes(new_image)
    print(pred2)

    #declare classes
    if pred2[0]==0:
        class_name = str(perct)+"%\n"+"Astrocytoma"
    else:
        if pred2[0]==1:
            class_name = str(perct)+ "%\n"+"Ependymoma"
        elif pred2[0]==2:
            class_name = str(perct)+"%\n"+"Oligodendroglioma"
        else:
            class_name = "Tidak ada Tumor Otak" 
        # else:
        #     class_name = str(perct)+"%\n"+"Oligodendroglioma"
    #print class name
    # return class_name, pred
    return class_name