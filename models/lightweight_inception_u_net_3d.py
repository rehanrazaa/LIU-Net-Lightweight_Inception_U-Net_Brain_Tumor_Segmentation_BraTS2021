# -*- coding: utf-8 -*-
"""Lightweight_Inception_U_Net-3D.ipynb

**Light-weight Inception U-Net : LIU-Net for Brain Tumor Segmentation**
"""

#importing required libraries
import numpy as np
import nibabel as nib
import os
import glob
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from tifffile import imsave

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D, concatenate
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam

# Defining model
def conv3d_inception_block(x, filters):
    tower_1 = Conv3D(filters, (1, 1, 1), padding='same', activation='relu')(x)
    tower_2 = Conv3D(filters, (3, 3, 3), padding='same', activation='relu')(x)
    tower_3 = Conv3D(filters, (5, 5, 5), padding='same', activation='relu')(x)
    return concatenate([tower_1, tower_2, tower_3], axis=-1)

def unet_3d_inception_lighter(input_shape):
    inputs = Input(input_shape)

    conv1 = conv3d_inception_block(inputs, 4)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

    conv2 = conv3d_inception_block(pool1, 8)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

    conv3 = conv3d_inception_block(pool2, 16)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

    conv4 = conv3d_inception_block(pool3, 32)
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conv4)

    conv5 = conv3d_inception_block(pool4, 64)

    up6 = UpSampling3D(size=(2, 2, 2))(conv5)
    merge6 = concatenate([conv4, up6], axis=-1)
    conv6 = conv3d_inception_block(merge6, 32)

    up7 = UpSampling3D(size=(2, 2, 2))(conv6)
    merge7 = concatenate([conv3, up7], axis=-1)
    conv7 = conv3d_inception_block(merge7, 16)

    up8 = UpSampling3D(size=(2, 2, 2))(conv7)
    merge8 = concatenate([conv2, up8], axis=-1)
    conv8 = conv3d_inception_block(merge8, 8)

    up9 = UpSampling3D(size=(2, 2, 2))(conv8)
    merge9 = concatenate([conv1, up9], axis=-1)
    conv9 = conv3d_inception_block(merge9, 4)

    outputs = Conv3D(4, 1, activation='softmax')(conv9)
    model = Model(inputs=inputs, outputs=outputs)

    return model

# Define input shape (adjust according to your dataset)
input_shape = (128, 128, 128, 4)

# Create the model
model = unet_3d_inception_lighter(input_shape)

# Compile the model
model.compile(loss=dice_coef_loss, optimizer=Adam(learning_rate=0.0001), metrics=['accuracy', dice_coef, brats_wt, brats_tc, brats_et])
model.summary()

print(model.input_shape)
print(model.output_shape)
