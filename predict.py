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

def create_model(weights_path):
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
    return model


def predict(image_path)
    test_model = create_model('/home/contoso/notebooks/XRCode/models/basic.h5')

    # predicting images
    img_test = image.load_img(image_path, target_size=(128, 128))
    x = image.img_to_array(img_test)
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
    result = test_model.predict_classes(images, batch_size=10)
    classes = train_generator.class_indices
    print(classes)
    for key,val in classes.items():
        if(val == result[0]):
            return key

def make_square(im, min_size=128, fill_color=(255, 255, 255, 255)):
    x, y = im.size
    size = max(min_size, x, y)
    new_im = Image.new('RGB', (size, size), fill_color)
    new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
    return new_im.resize((128,128))
def convert(file,dest):
    test_image = Image.open(file)
    new_image = make_square(test_image)
    new_image.save(dest)



