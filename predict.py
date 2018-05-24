import os
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras import optimizers
from keras import applications
from keras.models import Model
from keras.preprocessing import image
from PIL import Image
from matplotlib import pyplot as plt

global model
global is_init
is_init=0

def create_model(weights_path):
    global model
    global is_init
    if is_init == 1: 
        return model
    
    img_width, img_height = 128, 128
    model = Sequential()
    model.add(Convolution2D(32, (3, 3), input_shape=(img_width, img_height,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(12))
    model.add(Activation('sigmoid'))
    model.load_weights(weights_path)
    is_init = 1
    return model

def get_classes():
    # dimensions of our images.
    img_width, img_height = 128, 128

    train_data_dir = 'data/train'
    validation_data_dir = 'data/validate'
    ##preprocessing
    # used to rescale the pixel values from [0, 255] to [0, 1] interval
    datagen = ImageDataGenerator(rescale=1./255)
    batch_size = 32

    # automagically retrieve images and their classes for train and validation sets
    train_generator = datagen.flow_from_directory(
            train_data_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode='categorical')
    return train_generator

def predict_img(image_path):
    # predicting images
    
    test_model = create_model('/home/contoso/notebooks/XRCode/models/basic.h5')
    img_test = image.load_img(image_path, target_size=(128, 128))
    x = image.img_to_array(img_test)
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
    result = test_model.predict_classes(images, batch_size=10)
    tg = get_classes()
    classes = tg.class_indices
    print(classes)
    json_result = ""
    for key,val in classes.items():
        if(val == result[0]):
            json_result = key
    
    return json_result

def make_square(im, min_size=128, fill_color=(255, 255, 255, 255)):
    x, y = im.size
    size = max(min_size, x, y)
    new_im = Image.new('RGB', (size, size), fill_color)
    new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
    return new_im.resize((128,128))
def convert_img(fi,dest):
    test_image = Image.open(fi)
    new_image = make_square(test_image)
    new_image.save(dest)
