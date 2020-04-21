from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, Conv2DTranspose
from keras import backend as K
import numpy as np

x_train = np.random.randint(low=5,size=(5,10,10))
y_train = np.random.randint(low=1,size=(5))

x_train = x_train.astype('float32')
x_train /= 255

model = Sequential()

#conv_1_1
n_filters = 64
model.add(ZeroPadding2D(100,name='padding_1_1'))
model.add(Conv2D(n_filters, kernel_size=(3,3),
                 activation='relu',
                 strides=(1,1),
                 name='conv_1_1',
                 padding="valid",
                 input_shape=(227,227)))
#conv_1_2
#model.add(ZeroPadding2D(1,name='padding_1_2'))
model.add(Conv2D(n_filters, kernel_size=(3,3),
                 activation='relu',
                 name='conv_1_2',
                 padding="same",
                 strides=(1,1)))
#pool_1
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),name='pool_1'))

#conv_2_1
n_filters = 128
#model.add(ZeroPadding2D(1,name='padding_2_1'))
model.add(Conv2D(n_filters, kernel_size=(3,3),
                 activation='relu',
                 name='conv_2_1',
                 padding="same",
                 strides=(1,1)))
#conv_2_2
#model.add(ZeroPadding2D(1,name='padding_2_2'))
model.add(Conv2D(n_filters, kernel_size=(3,3),
                 activation='relu',
                 name='conv_2_2',
                 padding="same",
                 strides=(1,1)))
#pool_2
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),name='pool_2'))


#conv_3_1
n_filters = 256
#model.add(ZeroPadding2D(1))
model.add(Conv2D(n_filters, kernel_size=(3,3),
                 activation='relu',
                 padding="same",
                 strides=(1,1)))
#conv_3_2
#model.add(ZeroPadding2D(1))
model.add(Conv2D(n_filters, kernel_size=(3,3),
                 activation='relu',
                 padding="same",
                 strides=(1,1)))
#conv_3_3
#model.add(ZeroPadding2D(1))
model.add(Conv2D(n_filters, kernel_size=(3,3),
                 activation='relu',
                 padding="same",
                 strides=(1,1)))
#pool_3
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))


#conv_4_1
n_filters = 512
#model.add(ZeroPadding2D(1))
model.add(Conv2D(n_filters, kernel_size=(3,3),
                 activation='relu',
                 padding="same",
                 strides=(1,1)))
#conv_4_2
#model.add(ZeroPadding2D(1))
model.add(Conv2D(n_filters, kernel_size=(3,3),
                 activation='relu',
                 padding="same",
                 strides=(1,1)))
#conv_4_3
#model.add(ZeroPadding2D(1))
model.add(Conv2D(n_filters, kernel_size=(3,3),
                 activation='relu',
                 padding="same",
                 strides=(1,1)))
#pool_4
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))


#conv_5_1
n_filters = 512
#model.add(ZeroPadding2D(1))
model.add(Conv2D(n_filters, kernel_size=(3,3),
                 activation='relu',
                 padding="same",
                 strides=(1,1)))
#conv_5_2
#model.add(ZeroPadding2D(1))
model.add(Conv2D(n_filters, kernel_size=(3,3),
                 activation='relu',
                 padding="same",
                 strides=(1,1)))
#conv_5_3
#model.add(ZeroPadding2D(1))
model.add(Conv2D(n_filters, kernel_size=(3,3),
                 activation='relu',
                 padding="same",
                 strides=(1,1)))
#pool_5
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))


n_filters = 4096
#fc6
model.add(Conv2D(n_filters, kernel_size=(7,7),
                 activation='relu',
                 padding="valid",
                 strides=(1,1)))


#drop6
model.add(Dropout(0.5))

#fc7
n_filters = 4096
model.add(Conv2D(n_filters, kernel_size=(1,1),
                 activation='relu',
                 padding="valid",
                 strides=(1,1)))


#drop7
model.add(Dropout(0.5))

n_filters = 60

#n.score_fr
model.add(Conv2D(n_filters, kernel_size=(1,1),
                 padding="valid",
                 strides=(1,1)))

n_filters=60
#n.upscore
model.add(Conv2DTranspose(n_filters,kernel_size=(64,64),strides=(32,32),use_bias=False))

#n.loss
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.build((5,227,227,3))
print(model.summary())

