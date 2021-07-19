"""
AUTHOR : NEWJIHWAN
LAST MODIFIED : 2021-07-19
PROJECTNAME : PNEUMONIA CLASSIFICATION BASE ON X-RAY IMAGE
"""
import tensorflow
import os
import cv2
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.models import Model
from build_model import build_model

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'


path = '../pneumonia_dataset/preprocessing/kaggle/chest_xray/'
img = cv2.imread('../preprocessing/kaggle/chest_xray/train/NORMAL/IM-0115-0001.jpeg')

# Data Load
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        path+'train',
        target_size=(600, 600),
        batch_size=32,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        path+'test',
        target_size=(600, 600),
        batch_size=32,
        class_mode='binary')
md = build_model(2)
md.fit_generator(train_generator,
                    steps_per_epoch = 300,
                    epochs = 25,
                    validation_data = validation_generator,
                    validation_steps = 2000)

# cxr
# Preprocessing -> (./255, CLAHE, Data Augmentation, etc..)
# Data split(Train, Validation, Test)


# Learning


# Test setopip


# Accuracy Calculatation


####################################################################################
