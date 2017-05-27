
"""Convolutional neural network built with Keras."""

import keras
from keras.datasets import cifar10
# from keras.datasets import cifar100
import keras.backend as K
import tensorflow as tf
from hyperopt import STATUS_OK


NB_CHANNELS = 3
IMAGE_BORDER_LENGTH = 32
NB_CLASSES = 10
# NB_CLASSES = 100

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# (x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')
x_train = x_train.astype('float32') / 255.0 - 0.5
x_test = x_test.astype('float32') / 255.0 - 0.5
y_train = keras.utils.to_categorical(y_train, NB_CLASSES)
y_test = keras.utils.to_categorical(y_test, NB_CLASSES)

BATCH_SIZE = 100
EPOCHS = 10


def cnn(hype_space):
    """Build a convolutional neural network and train it."""
    K.set_image_data_format('channels_last')
    model = build_model(hype_space)

    history = model.fit(
        x_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        shuffle=True,
        verbose=1,
        validation_data=(x_test, y_test)
    )

    score = model.evaluate(x_test, y_test, verbose=0)

    results = {
        'loss': score[0],
        'accuracy': score[1],
        'history': history.history,
        'status': STATUS_OK
    }
    print("RESULTS:")
    print(results)
    print("\n\n")
    return results


def build_model(hype_space):
    """Create model according to the hyperparameter space given."""
    print("Current space being optimized:")
    print(hype_space)

    input_layer = keras.layers.Input(
        (IMAGE_BORDER_LENGTH, IMAGE_BORDER_LENGTH, NB_CHANNELS))

    current_layer = random_image_mirror_left_right(input_layer)

    # Core loop that stacks multiple conv+pool layers, with maybe some
    # residual connections and other fluffs:
    n_filters = int(32 * hype_space['hidden_units_mult'])
    for i in range(hype_space['nb_conv_pool_layers']):
        print(i)
        print(n_filters)
        print(current_layer._keras_shape)

        current_layer = convolution(current_layer, n_filters, hype_space)
        print(current_layer._keras_shape)

        if hype_space['use_BN']:
            current_layer = bn(current_layer)
            print(current_layer._keras_shape)

        if hype_space['residual'] is not None:
            current_layer = bn(residual(current_layer, n_filters, hype_space))
            print(current_layer._keras_shape)

        if hype_space['use_allconv_pooling']:
            current_layer = convolution_pooling(
                current_layer, n_filters, hype_space)
            if hype_space['use_BN']:
                current_layer = bn(current_layer)
        else:
            print(current_layer._keras_shape)
            current_layer = keras.layers.pooling.MaxPooling2D(
                pool_size=(2, 2), dim_ordering="tf"
            )(current_layer)
        print(current_layer._keras_shape)

        current_layer = dropout(current_layer, hype_space)

        n_filters *= 2

    # Fully Connected (FC) part:
    current_layer = keras.layers.core.Flatten()(current_layer)
    print(current_layer._keras_shape)

    current_layer = keras.layers.core.Dense(
        units=int(700 * hype_space['fc_units_mult']),
        activation="relu"
    )(current_layer)
    print(current_layer._keras_shape)

    current_layer = dropout(current_layer, hype_space)

    current_layer = keras.layers.core.Dense(
        units=NB_CLASSES,
        activation="relu"
    )(current_layer)
    print(current_layer._keras_shape)

    # Finalize model:
    model = keras.models.Model(input=input_layer, output=current_layer)
    model.compile(
        optimizer=hype_space['optimizer'](
            lr=0.0015 * hype_space['lr_rate_mult']
        ),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def convolution(prev_layer, n_filters, hype_space):
    """Basic convolution layer, parametrized by the hype_space."""
    k = hype_space['conv_kernel_size']
    return keras.layers.convolutional.Conv2D(
        filters=n_filters, kernel_size=(k, k), strides=(1, 1),
        padding='same', activation='relu'
    )(prev_layer)


def residual(prev_layer, n_filters, hype_space):
    """Some sort of residual layer, parametrized by the hype_space."""
    current_layer = prev_layer
    for i in range(hype_space['residual']):
        layer_to_add = dropout(current_layer, hype_space)
        layer_to_add = convolution(layer_to_add, n_filters, hype_space)
        current_layer = keras.layers.add([
            current_layer,
            layer_to_add
        ])
    return current_layer


def convolution_pooling(prev_layer, n_filters, hype_space):
    """
    Pooling with a convolution of stride 2.

    See: https://arxiv.org/pdf/1412.6806.pdf
    """
    return keras.layers.convolutional.Conv2D(
        filters=n_filters, kernel_size=(3, 3), strides=(2, 2),
        padding='valid', activation='linear'
    )(prev_layer)


def bn(prev_layer):
    """Perform batch normalisation."""
    return keras.layers.normalization.BatchNormalization()(prev_layer)


def dropout(prev_layer, hype_space):
    """Add dropout after a layer."""
    return keras.layers.core.Dropout(
        rate=hype_space['dropout_drop_proba']
    )(prev_layer)


def random_image_mirror_left_right(input_layer):
    """
    Flip an image left-right like in a mirror, randomly.

    This acts as a data augmentation technique. See:
    https://stackoverflow.com/questions/39574999/tensorflow-tf-image-functions-on-an-image-batch
    """
    return keras.layers.core.Lambda(function=lambda batch_imgs: tf.map_fn(
        lambda img: tf.image.random_flip_left_right(img), batch_imgs
    )
    )(input_layer)
