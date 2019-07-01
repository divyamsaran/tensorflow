import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras import backend as K

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

sess = tf.InteractiveSession()

image_rows = 28
image_cols = 28

# reshape images to 28 X 28 X 1 
train_images = mnist.train.images.reshape(mnist.train.images.shape[0],image_rows, image_cols, 1)
test_images =  mnist.test.images.reshape(mnist.test.images.shape[0], image_rows, image_cols, 1)

# Layer values
num_filters_1 = 32                
num_filters_2 = 64                
max_pool_size = (2, 2)          
conv_kernel_size = (3, 3)       
imag_shape = (28,28,1)
num_classes = 10
drop_prob = 0.2                 

# Define the model type
model = Sequential()

# Convolution -> RELU -> MAX Pool
# border_mode=valid doesn't allow partial overlap between input and filter
model.add(Convolution2D(num_filters_1, conv_kernel_size[0], conv_kernel_size[1], border_mode='valid',
                        input_shape=imag_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=max_pool_size))

# Convolution -> RELU -> MAX Pool
model.add(Convolution2D(num_filters_2, conv_kernel_size[0], conv_kernel_size[1]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=max_pool_size))

# FC
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))

# dropout
model.add(Dropout(drop_prob))

# Output layer
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# Cross entropy loss, adam optimizer, accuracy metric
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Train settings
batch_size = 128
num_epoch = 10

# fit the training data to the model.  Nicely displays the time, loss, and validation accuracy on the test data
model.fit(train_images, mnist.train.labels, batch_size=batch_size, nb_epoch=num_epoch,
          verbose=1, validation_data=(test_images, mnist.test.labels))


