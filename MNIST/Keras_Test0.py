#!/usr/bin/env Python
# -*- coding: utf-8 -*-

# Keras, Test: 0

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, \
                         Conv2D, MaxPooling2D
from keras import backend as K
BATCH_SIZE = 128
NUM_CLASSES = 10
EPOCHS = 12
IMG_ROWS, IMG_COLS = 28, 28


( X_train, y_train ), ( X_test, y_test ) = mnist.load_data()


# 数据维度调整
if K.backend() == "tensorflow":
    X_train = X_train.reshape( X_train.shape[0], IMG_ROWS, IMG_COLS, 1 )
    X_test = X_test.reshape( X_test.shape[0], IMG_ROWS, IMG_COLS, 1 )
    INPUT_SHAPE = ( IMG_ROWS, IMG_COLS, 1 )
elif K.backend() == "theano":
    X_train = X_train.reshape( X_train.shape[0], 1, IMG_ROWS, IMG_COLS )
    X_test = X_test.reshape( X_test.shape[0], 1, IMG_ROWS, IMG_COLS )
    INPUT_SHAPE = ( 1, IMG_ROWS, IMG_COLS )
else:
    print("")


X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
X_train /= 255
X_test /= 255
# 打印数据形态
print( "Training-X shape:", X_train.shape )
print( X_train.shape[0], "Training samples" )
print( X_test.shape[0], "Testing samples" )


# 将标签数据变换为one-hot型数据
y_train = keras.utils.to_categorical( y = y_train, num_classes = NUM_CLASSES )
y_test = keras.utils.to_categorical( y = y_test, num_classes = NUM_CLASSES )


# 创建一个顺序模型, 将所有涉及的计算顺序加载入这个模型
model = Sequential()
model.add(layer = Conv2D( filters = 32, kernel_size = ( 3, 3 ), activation = "relu", input_shape = INPUT_SHAPE ))
model.add(layer = Conv2D( filters = 64, kernel_size = ( 3, 3 ), activation = "relu" ))
model.add(layer = MaxPooling2D(pool_size = ( 2, 2 )))
model.add(layer = Dropout(rate = 0.25))
model.add(layer = Flatten())
model.add(layer = Dense( units = 128, activation = "relu" ))
model.add(layer = Dropout(rate = 0.25))
model.add(layer = Dense( units = NUM_CLASSES, activation = "softmax" ))


# 编译模型
model.compile(
    loss = keras.losses.categorical_crossentropy,
    optimizer = keras.optimizers.Adadelta(),
    metrics = ["Accuracy"]
    )


# 训练模型
model.fit(
    X_train, y_train,
    batch_size = BATCH_SIZE,
    epochs = EPOCHS,
    verbose = 1,
    validation_data = ( X_test, y_test )
    )


# 模型评估
score = model.evaluate( x = X_test, y = y_test, verbose = 0 )
print( "Loss: %s" % score[0] )
print( "Accuracy: %s" % score[1] )


# -*- END -*- #



