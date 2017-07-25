#
# This file has been taken and modified from:
# https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
#
# COPYRIGHT
#
# All contributions by François Chollet:
# Copyright (c) 2015, François Chollet.
# All rights reserved.
#
# Each contributor holds copyright over their respective contributions.
# The project versioning (Git) records all such contribution source information.
#
# LICENSE
#
# The MIT License (MIT)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# SUBLICENSE
#
# The modifications to this file by Guillaume Chevalier, on behalf of Vooban Inc.,
# are also licenced (sublicense) with The MIT License (MIT).
#
# Copyright (c) 2017 Vooban Inc. See:
# https://github.com/Vooban/Hyperopt-Keras-CNN-CIFAR-100/blob/master/LICENSE
#

"""Visualization of the filters of the CNN, via gradient ascent in input space.

This script can run on CPU in a few minutes (with the TensorFlow backend).
All convolutional layers are processed, but only the top filters of that
layer are shown. The best neural network is loaded, but the weight file must
be set manually in case of a retrain.

Results are saved in the subfolder "layers/".
"""


from utils import load_best_hyperspace
from neural_net import WEIGHTS_DIR, build_model

from scipy.misc import imsave
import numpy as np
from keras import backend as K

import time
import os


# Dimensions of the generated pictures for each filter.
img_width = 32
img_height = 32
weight_file = "{}/f37d5.hdf5".format(WEIGHTS_DIR)
LAYERS_DIR = "layers"

# Load model in test phase mode: no dropout, and use fixed BN
K.set_learning_phase(0)
model = build_model(load_best_hyperspace())
model.load_weights(weight_file)

print('Model loaded.')
model.summary()


def normalize(x):
    """Utility function to normalize a tensor by its L2 norm."""
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)


def deprocess_image(x):
    """Utility function to convert a tensor into a valid image."""
    # Normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # Clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # Convert to RGB array
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x


# Placeholder for the input images
input_img = model.input

# Symbolic outputs of each "key" layer (we gave them unique names).
layer_dict = {layer.name: layer for layer in model.layers[1:]}

# The type of layers plotted can be changed by changing the "add" keyword.
# For example, we could plot every convolutional layer by replacing "add" by
# "conv".
layers_to_plot = [l.name for l in model.layers if "add" in l.name.lower()]
# We take add because it is a strategic bottleneck from the residual
# connections.

for layer_name in layers_to_plot:

    kept_filters = []
    layer_obj = layer_dict[layer_name]
    if K.image_data_format() == 'channels_first':
        nb_filters = layer_obj.output_shape[1]
    else:
        nb_filters = layer_obj.output_shape[-1]
    print("Processing layer '{}' with shape {}.".format(
        layer_name, layer_obj.output_shape))

    for filter_index in range(0, nb_filters):
        # We scan through all filters.
        print('Processing filter {}'.format(filter_index))
        start_time = time.time()

        # We build a loss function that maximizes the activation
        # of the `nth` filter of the current layer
        layer_output = layer_obj.output
        if K.image_data_format() == 'channels_first':
            loss = K.mean(layer_output[:, filter_index, :, :])
        else:
            loss = K.mean(layer_output[:, :, :, filter_index])

        # We compute the gradient of the input picture wrt this loss
        grads = K.gradients(loss, input_img)[0]

        # Normalization trick: we normalize the gradient
        grads = normalize(grads)

        # This function returns the loss and grads given the input picture
        iterate = K.function([input_img], [loss, grads])

        # Step size for gradient ascent
        step = 1.

        # We start from a gray image with some random noise
        if K.image_data_format() == 'channels_first':
            input_img_data = np.random.random((1, 3, img_width, img_height))
        else:
            input_img_data = np.random.random((1, img_width, img_height, 3))
        input_img_data = (input_img_data - 0.5) * 20 + 128

        # We run gradient ascent for 20 steps
        for i in range(20):
            loss_value, grads_value = iterate([input_img_data])
            input_img_data += grads_value * step

            print('Current loss value:', loss_value)
            if loss_value <= 0.:
                # Some filters get stuck to 0, we can skip them
                break

        # Decode the resulting input image
        if loss_value > 0:
            img = deprocess_image(input_img_data[0])
            kept_filters.append((img, loss_value))
        end_time = time.time()
        print('Filter {} processed in {}s'.format(
            filter_index, end_time - start_time))

    # We will stich only the best filters that fit on a perfect square grid
    # (excess is discarded). The file name will say how many filters were kept.
    # Some filters can be discarded due to a negative loss (diverged), too.
    n = int(float(len(kept_filters))**0.5)

    # The filters that have the highest loss are assumed to be better-looking.
    # We will only keep the top 64 filters.
    kept_filters.sort(key=lambda x: x[1], reverse=True)
    kept_filters = kept_filters[:n * n]

    # Build a black picture with enough space for our `n x n` filters
    # of size `img_width x img_height`, with a 5px margin in between
    margin = 5
    width = n * img_width + (n - 1) * margin
    height = n * img_height + (n - 1) * margin
    stitched_filters = np.zeros((width, height, 3))

    # Fill the picture with our saved filters
    for i in range(n):
        for j in range(n):
            img, loss = kept_filters[i * n + j]
            stitched_filters[(img_width + margin) * i: (img_width + margin) * i + img_width,
                             (img_height + margin) * j: (img_height + margin) * j + img_height, :] = img

    if not os.path.exists(LAYERS_DIR):
        os.makedirs(LAYERS_DIR)
    # Save the result to disk
    imsave(
        '{}/{}_best_filters_{}_({}x{})_out_of_{}.png'.format(
            LAYERS_DIR, layer_name, n**2, n, n, nb_filters
        ),
        stitched_filters
    )
