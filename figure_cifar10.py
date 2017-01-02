# Plot ad hoc CIFAR10 instances
import sys
import numpy as np
from matplotlib import pyplot
from scipy.misc import toimage
from sets import Set
# Simple CNN model for CIFAR-10
import numpy
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

from numpy import linalg as LA
from matplotlib import pyplot
from scipy.ndimage.filters import gaussian_filter
from scipy.misc import toimage
from numpy import percentile
from image_misc import saveimagesc, saveimagescc

from scipy.misc import imsave
import time


K.set_image_dim_ordering('th')
# load data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
# create a grid of 3x3 images
hash_table = {"airplane":0, "automobile":1, "bird":2, "cat":3, "deer":4, "dog":5, "frog":6, "horse":7, "ship":8, "truck":9}

# consider problem with 2, 4, 6, 8, 10 classes
def limited_classes(classes, type, X, y):
    if classes == 2: nameList = Set(["cat", "truck"])
    elif classes == 4: nameList = Set(["cat", "truck", "bird", "airplane"])
    else: sys.exit("invalid number of classes")
    tag = [hash_table[name] for name in hash_table if name in nameList]

    # as to 2 classes, tag = [9, 3]
    if type == "train": ifSet = [True if y_train[num] in tag else False for num in range(len(y_train))]
    if type == "test": ifSet = [True if y_test[num] in tag else False for num in range(len(y_test))]
    ifSet = np.array(ifSet)
    # set categorical label  9->0, 3->1
    projection = {}
    for i in range(len(tag)): projection[tag[i]] = i
    if type == "train":
        print(hash_table)
        print("Projection", projection)

    X, y = X[ifSet], y[ifSet]
    y = y.T.tolist()[0]
    y = np.array([projection[y[i]] for i in range(len(y))])
    return X, y

target_classes = 4
X_train, y_train = limited_classes(target_classes, "train", X_train, y_train)
X_test, y_test = limited_classes(target_classes, "test", X_test, y_test)



# normalize inputs from 0-255 to 0.0-1.0
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

# # for test case, reduce samples
# X_train = X_train[:500]
# y_train = y_train[:500]
# X_test = X_test[:300]
# y_test = y_test[:300]

# # test if our samples are rightly collected
# for i in range(100):
#     pyplot.imshow(toimage(X_train[i]))
#     pyplot.show()
#     print(y_train[i])



# Create a small model
# model = Sequential()
# model.add(Convolution2D(32, 3, 3, input_shape=(3, 32, 32), border_mode='same', activation='relu', W_constraint=maxnorm(3)))
# #model.add(Dropout(0.2))
# model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same', W_constraint=maxnorm(3)))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Flatten())
# model.add(Dense(512, activation='relu', W_constraint=maxnorm(3)))
# #model.add(Dropout(0.5))
# model.add(Dense(num_classes, activation='softmax'))

# # Create a large model
model = Sequential()
model.add(Convolution2D(32, 3, 3, input_shape=(3, 32, 32), activation='relu', border_mode='same'))
#model.add(Dropout(0.2))
model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))
#model.add(Dropout(0.2))
model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))
#model.add(Dropout(0.2))
model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
#model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu', W_constraint=maxnorm(3)))
#model.add(Dropout(0.2))
model.add(Dense(512, activation='relu', W_constraint=maxnorm(3)))
#model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))
# # Compile model
epochs = 1  # wei's change, original is 25
lrate = .01 # wei's change, original is .01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary())

# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=epochs, batch_size=500) ####  changed from 32 to 320
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=1)
print("Accuracy: %.2f%%" % (scores[1]*100))


## ********************************************************** Module 2 ********************************************************** 


# dimensions of the generated pictures for each filter.
img_width = 32
img_height = 32

# this is the placeholder for the input images
input_img = model.input

# get the symbolic outputs of each "key" layer (we gave them unique names).
layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])


# the name of the layer we want to visualize
# (see model definition at keras/applications/vgg16.py)
layer_name = 'dense_3'
max_iter = 200
params_decay = 0.01
params_blur_radius = .2
params_blur_every = 4

for filter_index in range(0, target_classes): # only keep 10
    # we only scan through the first 200 filters,
    # but there are actually 512 of them
    print('Processing filter %d' % filter_index)
    start_time = time.time()

    # we build a loss function that maximizes the activation
    # of the nth filter of the layer considered
    layer_output = layer_dict[layer_name].output
    # if K.image_dim_ordering() == 'th':
    #     loss = K.mean(layer_output[:, filter_index, :, :])
    # else:
    #     loss = K.mean(layer_output[:, :, :, filter_index])
    # layer_output = model.layers[-1].output
    loss = K.mean(layer_output[:, filter_index])

    # we compute the gradient of the input picture wrt this loss
    grads = K.gradients(loss, input_img)[0]

    # normalization trick: we normalize the gradient
    #grads = normalize(grads)

    # this function returns the loss and grads given the input picture
    iterate = K.function([input_img], [loss, grads])

    # step size for gradient ascent
    step = 1

    # we start from a gray image with some random noise
    input_img_data = np.random.normal(0, 10, (1, 3, img_width, img_height))

    best_prob = 0
    for i in range(max_iter):
        prob, grads_value = iterate([input_img_data])
        #grads_value = grads_value * 1000000 #/ LA.norm(grads_value)
        grads_value = grads_value / LA.norm(grads_value) * 15
        input_img_data += grads_value
        print("|-------------------------------------------------------------------------|")
        # L2 decay
        print("img:", input_img_data.min(), input_img_data.max(), LA.norm(input_img_data))
        print("grad:", grads_value.min(), grads_value.max(), LA.norm(grads_value))
        print("|_________________________________________________________________________|")

        # L2 decay
        input_img_data *= (1 - params_decay)  
        # gaussian filters
        if i % params_blur_every == 0:
            for channel in range(3):
                cimg = gaussian_filter(input_img_data[0, channel], params_blur_radius)
                input_img_data[0, channel] = cimg
        if prob > best_prob:
            best_prob = prob
            best_img = input_img_data
        out = model.predict(input_img_data)
        
        outIDX = np.argmax(out[0])
        print('loop: %d, current prob: %.10f, input idx: %d, pred idx: %d, best prob: %.10f' \
            %(i, prob, filter_index, outIDX, best_prob))
        if best_prob > 0.9999: break
    # decode the resulting input image
    img = best_img[0].transpose((1,2,0))
    saveimagescc('./images/best_pred_%d_between_4classes_radius.2.jpg' % filter_index, img, 0)

# # we will stich the best 64 filters on a 8 x 8 grid.
# n = target_classes # wei: i changed here from 8 to 2

# # the filters that have the highest loss are assumed to be better-looking.
# # we will only keep the top 64 filters.
# kept_filters.sort(key=lambda x: x[1], reverse=True)

# print(kept_filters)
# kept_filters = kept_filters[:n * n]

# # build a black picture with enough space for
# # our 8 x 8 filters of size 128 x 128, with a 5px margin in between
# margin = 5
# width = img_width
# height = n * img_height + (n - 1) * margin
# stitched_filters = np.zeros((width, height, 3))

# # fill the picture with our saved filters
# for i in range(1):
#     for j in range(n):
#         img, loss = kept_filters[i * n + j]
#         stitched_filters[(img_width + margin) * i: (img_width + margin) * i + img_width,
#                          (img_height + margin) * j: (img_height + margin) * j + img_height, :] = img

# # save the result to disk
# imsave('cifar_%dx%d.png' % (1, n), stitched_filters)
