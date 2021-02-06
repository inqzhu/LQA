#! /usr/bin/env python
# coding=utf-8

"""
获取数据集
cifar10
"""

import tensorflow as tf
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

def get_dataflow(batch_size):
    (X0,Y0),(X1,Y1) = tf.keras.datasets.cifar10.load_data()

    N0=X0.shape[0];N1=X1.shape[0]
    X0 = X0.reshape(N0,32,32,3)/255
    X1 = X1.reshape(N1,32,32,3)/255

    ## 数据标准化
    mean = np.mean(X0,axis=(0,1,2,3))
    std = np.std(X0,axis=(0,1,2,3))
    X0 = (X0-mean)/(std+1e-7)
    X1 = (X1-mean)/(std+1e-7)

    X0 = X0.astype(np.float32)
    X1 = X1.astype(np.float32)

    datagen = ImageDataGenerator(
       featurewise_center=False,  # set input mean to 0 over the dataset
       samplewise_center=False,  # set each sample mean to 0
       featurewise_std_normalization=False,  # divide inputs by std of the dataset
       samplewise_std_normalization=False,  # divide each input by its std
       zca_whitening=False,  # apply ZCA whitening
       rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
       width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
       height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
       horizontal_flip=True,  # randomly flip images
       vertical_flip=False)  # randomly flip images
    datagen.fit(X0)
    
    #batch_size = 64
    flows = datagen.flow(X0, Y0, batch_size = batch_size)
    
    return X0, X1, Y0, Y1, flows
