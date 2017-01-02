'''Visualization of the filters of VGG16, via gradient ascent with regularization in input space.
This script can run on CPU in a few minutes (with the TensorFlow backend).
The original version is finished by keras, I added the regularization based on the work of yosinki
'''
from __future__ import print_function
from scipy.misc import imsave
import numpy as np
import time, sys
from keras.applications import vgg16
from keras import backend as K
#from keras.layers.core import K
from vgg16_model import VGG_16
import tensorflow as tf
from numpy import linalg as LA
import cv2
from matplotlib import pyplot
from scipy.ndimage.filters import gaussian_filter
from scipy.misc import toimage
from numpy import percentile
from image_misc import saveimagesc, saveimagescc

# https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html
# dimensions of the generated pictures for each filter.
K.set_learning_phase(0)

# build the VGG16 network with ImageNet weights, the following doesn't include fully connected layer
# model = vgg16.VGG16(weights='imagenet', include_top=False)
# the following contains fully connected layer 
model = VGG_16('./vgg16_weights.h5')
print('Model loaded.')

model.summary()

# this is the placeholder for the input images
input_img = model.input

# get the symbolic outputs of each "key" layer (we gave them unique names).
layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])

# the name of the layer we want to visualize
# (see model definition at keras/applications/vgg16.py)
layer_name = 'dense_3'
max_iter = 300
img_width = 224
img_height = 224
params_decay = 0.01
params_blur_radius = 1
params_blur_every = 4
# params_small_val_percentile = 0.5
# params_small_norm_percentile = .1
kept_filters = []
inputIDX = int(sys.argv[1])
for filter_index in range(inputIDX, inputIDX + 1, 30): # only keep 10
    # we only scan through the first 200 filters,
    # but there are actually 512 of them
    print('Processing filter %d' % filter_index)

    # we build a loss function that maximizes the activation
    # of the nth filter of the layer considered
    layer_output = layer_dict[layer_name].output
    # if K.image_dim_ordering() == 'th': loss = K.mean(layer_output[:, filter_index, :, :])
    # else: loss = K.mean(layer_output[:, :, :, filter_index])
    # layer_output = model.layers[-1].output
    loss = K.mean(layer_output[:, filter_index])

    # we compute the gradient of the input picture wrt this loss
    grads = K.gradients(loss, input_img)[0]

    # normalization trick: we normalize the gradient
    #grads = 3 * grads / (K.sqrt(K.mean(K.square(grads))) + 1e-5) # original suggected 

    # this function returns the loss and grads given the input picture
    iterate = K.function([input_img], [loss, grads])

    # we start from a zero-image with some random noise
    #input_img_data = np.random.random(0, 10, (1, 3, img_width, img_height))
    input_img_data = np.random.normal(0, 10, (1, 3, img_width, img_height))

    best_prob = 0
    for i in range(max_iter):
        prob, grads_value = iterate([input_img_data])
        #grads_value = grads_value * 1000000 #/ LA.norm(grads_value)
        grads_value = grads_value / LA.norm(grads_value) * 10
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
        if best_prob > 0.99: break
    # decode the resulting input image
    img = best_img[0].transpose((1,2,0))
    saveimagescc('./images/best_pred_%d.jpg' % filter_index, img, 0)
    
