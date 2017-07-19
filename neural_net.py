
"""Convolutional neural network built with Keras."""

from utils import print_json

import keras
from keras.datasets import cifar100  # from keras.datasets import cifar10
from keras.layers.core import K  # import keras.backend as K
from keras.optimizers import Adam, Nadam, RMSprop
import tensorflow as tf
from hyperopt import STATUS_OK, STATUS_FAIL

import uuid
import traceback
import os


__author__ = "Guillaume Chevalier"
__copyright__ = "Copyright 2017, Guillaume Chevalier"
__license__ = "MIT License"
__notice__ = (
    "Some further edits by Guillaume Chevalier are made on "
    "behalf of Vooban Inc. and belongs to Vooban Inc. ")
# See: https://github.com/Vooban/Hyperopt-Keras-CNN-CIFAR-100/blob/master/LICENSE"


TENSORBOARD_DIR = "TensorBoard/"
WEIGHTS_DIR = "weights/"


NB_CHANNELS = 3
IMAGE_BORDER_LENGTH = 32
# NB_CLASSES = 10
NB_CLASSES_FINE = 100
NB_CLASSES_COARSE = 20

# (x_train, y_train), (x_test, y_test) = cifar10.load_data()
(_, y_train_c), (_, y_test_coarse) = cifar100.load_data(label_mode='coarse')
(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')
x_train = x_train.astype('float32') / 255.0 - 0.5
x_test = x_test.astype('float32') / 255.0 - 0.5
y_train = keras.utils.to_categorical(y_train, NB_CLASSES_FINE)
y_test = keras.utils.to_categorical(y_test, NB_CLASSES_FINE)
y_train_c = keras.utils.to_categorical(y_train_c, NB_CLASSES_COARSE)
y_test_coarse = keras.utils.to_categorical(y_test_coarse, NB_CLASSES_COARSE)

# You may want to reduce this considerably if you don't have a killer GPU:
EPOCHS = 100
STARTING_L2_REG = 0.0007

OPTIMIZER_STR_TO_CLASS = {
    'Adam': Adam,
    'Nadam': Nadam,
    'RMSprop': RMSprop
}


def build_and_train(hype_space, save_best_weights=False, log_for_tensorboard=False):
    """Build the deep CNN model and train it."""
    K.set_learning_phase(1)
    K.set_image_data_format('channels_last')

    # if log_for_tensorboard:
    #     # We need a smaller batch size to not blow memory with tensorboard
    #     hype_space["lr_rate_mult"] = hype_space["lr_rate_mult"] / 10.0
    #     hype_space["batch_size"] = hype_space["batch_size"] / 10.0

    model = build_model(hype_space)

    # K.set_learning_phase(1)

    model_uuid = str(uuid.uuid4())[:5]

    callbacks = []

    # Weight saving callback:
    if save_best_weights:
        weights_save_path = os.path.join(
            WEIGHTS_DIR, '{}.hdf5'.format(model_uuid))
        print("Model's weights will be saved to: {}".format(weights_save_path))
        if not os.path.exists(WEIGHTS_DIR):
            os.makedirs(WEIGHTS_DIR)

        callbacks.append(keras.callbacks.ModelCheckpoint(
            weights_save_path,
            monitor='val_fine_outputs_acc',
            save_best_only=True, mode='max'))

    # TensorBoard logging callback:
    log_path = None
    if log_for_tensorboard:
        log_path = os.path.join(TENSORBOARD_DIR, model_uuid)
        print("Tensorboard log files will be saved to: {}".format(log_path))
        if not os.path.exists(log_path):
            os.makedirs(log_path)

        # Right now Keras's TensorBoard callback and TensorBoard itself are not
        # properly documented so we do not save embeddings (e.g.: for T-SNE).

        # embeddings_metadata = {
        #     # Dense layers only:
        #     l.name: "../10000_test_classes_labels_on_1_row_in_plain_text.tsv"
        #     for l in model.layers if 'dense' in l.name.lower()
        # }

        tb_callback = keras.callbacks.TensorBoard(
            log_dir=log_path,
            histogram_freq=2,
            # write_images=True, # Enabling this line would require more than 5 GB at each `histogram_freq` epoch.
            write_graph=True
            # embeddings_freq=3,
            # embeddings_layer_names=list(embeddings_metadata.keys()),
            # embeddings_metadata=embeddings_metadata
        )
        tb_callback.set_model(model)
        callbacks.append(tb_callback)

    # Train net:
    history = model.fit(
        [x_train],
        [y_train, y_train_c],
        batch_size=int(hype_space['batch_size']),
        epochs=EPOCHS,
        shuffle=True,
        verbose=1,
        callbacks=callbacks,
        validation_data=([x_test], [y_test, y_test_coarse])
    ).history

    # Test net:
    K.set_learning_phase(0)
    score = model.evaluate([x_test], [y_test, y_test_coarse], verbose=0)
    max_acc = max(history['val_fine_outputs_acc'])

    model_name = "model_{}_{}".format(str(max_acc), str(uuid.uuid4())[:5])
    print("Model name: {}".format(model_name))

    # Note: to restore the model, you'll need to have a keras callback to
    # save the best weights and not the final weights. Only the result is
    # saved.
    print(history.keys())
    print(history)
    print(score)
    result = {
        # We plug "-val_fine_outputs_acc" as a
        # minimizing metric named 'loss' by Hyperopt.
        'loss': -max_acc,
        'real_loss': score[0],
        # Fine stats:
        'fine_best_loss': min(history['val_fine_outputs_loss']),
        'fine_best_accuracy': max(history['val_fine_outputs_acc']),
        'fine_end_loss': score[1],
        'fine_end_accuracy': score[3],
        # Coarse stats:
        'coarse_best_loss': min(history['val_coarse_outputs_loss']),
        'coarse_best_accuracy': max(history['val_coarse_outputs_acc']),
        'coarse_end_loss': score[2],
        'coarse_end_accuracy': score[4],
        # Misc:
        'model_name': model_name,
        'space': hype_space,
        'history': history,
        'status': STATUS_OK
    }

    print("RESULT:")
    print_json(result)

    return model, model_name, result, log_path


def build_model(hype_space):
    """Create model according to the hyperparameter space given."""
    print("Hyperspace:")
    print(hype_space)

    input_layer = keras.layers.Input(
        (IMAGE_BORDER_LENGTH, IMAGE_BORDER_LENGTH, NB_CHANNELS))

    current_layer = random_image_mirror_left_right(input_layer)

    if hype_space['first_conv'] is not None:
        k = hype_space['first_conv']
        current_layer = keras.layers.convolutional.Conv2D(
            filters=16, kernel_size=(k, k), strides=(1, 1),
            padding='same', activation=hype_space['activation'],
            kernel_regularizer=keras.regularizers.l2(
                STARTING_L2_REG * hype_space['l2_weight_reg_mult'])
        )(current_layer)

    # Core loop that stacks multiple conv+pool layers, with maybe some
    # residual connections and other fluffs:
    n_filters = int(40 * hype_space['conv_hiddn_units_mult'])
    for i in range(hype_space['nb_conv_pool_layers']):
        print(i)
        print(n_filters)
        print(current_layer._keras_shape)

        current_layer = convolution(current_layer, n_filters, hype_space)
        if hype_space['use_BN']:
            current_layer = bn(current_layer)
        print(current_layer._keras_shape)

        deep_enough_for_res = hype_space['conv_pool_res_start_idx']
        if i >= deep_enough_for_res and hype_space['residual'] is not None:
            current_layer = residual(current_layer, n_filters, hype_space)
            print(current_layer._keras_shape)

        current_layer = auto_choose_pooling(
            current_layer, n_filters, hype_space)
        print(current_layer._keras_shape)

        current_layer = dropout(current_layer, hype_space)

        n_filters *= 2

    # Fully Connected (FC) part:
    current_layer = keras.layers.core.Flatten()(current_layer)
    print(current_layer._keras_shape)

    current_layer = keras.layers.core.Dense(
        units=int(1000 * hype_space['fc_units_1_mult']),
        activation=hype_space['activation'],
        kernel_regularizer=keras.regularizers.l2(
            STARTING_L2_REG * hype_space['l2_weight_reg_mult'])
    )(current_layer)
    print(current_layer._keras_shape)

    current_layer = dropout(
        current_layer, hype_space, for_convolution_else_fc=False)

    if hype_space['one_more_fc'] is not None:
        current_layer = keras.layers.core.Dense(
            units=int(750 * hype_space['one_more_fc']),
            activation=hype_space['activation'],
            kernel_regularizer=keras.regularizers.l2(
                STARTING_L2_REG * hype_space['l2_weight_reg_mult'])
        )(current_layer)
        print(current_layer._keras_shape)

        current_layer = dropout(
            current_layer, hype_space, for_convolution_else_fc=False)

    # Two heads as outputs:
    fine_outputs = keras.layers.core.Dense(
        units=NB_CLASSES_FINE,
        activation="sigmoid",
        kernel_regularizer=keras.regularizers.l2(
            STARTING_L2_REG * hype_space['l2_weight_reg_mult']),
        name='fine_outputs'
    )(current_layer)

    coarse_outputs = keras.layers.core.Dense(
        units=NB_CLASSES_COARSE,
        activation="sigmoid",
        kernel_regularizer=keras.regularizers.l2(
            STARTING_L2_REG * hype_space['l2_weight_reg_mult']),
        name='coarse_outputs'
    )(current_layer)

    # Finalize model:
    model = keras.models.Model(
        inputs=[input_layer],
        outputs=[fine_outputs, coarse_outputs]
    )
    model.compile(
        optimizer=OPTIMIZER_STR_TO_CLASS[hype_space['optimizer']](
            lr=0.001 * hype_space['lr_rate_mult']
        ),
        loss='categorical_crossentropy',
        loss_weights=[1.0, hype_space['coarse_labels_weight']],
        metrics=['accuracy']
    )
    return model


def random_image_mirror_left_right(input_layer):
    """
    Flip each image left-right like in a mirror, randomly, even at test-time.

    This acts as a data augmentation technique. See:
    https://stackoverflow.com/questions/39574999/tensorflow-tf-image-functions-on-an-image-batch
    """
    return keras.layers.core.Lambda(function=lambda batch_imgs: tf.map_fn(
        lambda img: tf.image.random_flip_left_right(img), batch_imgs
    )
    )(input_layer)


def bn(prev_layer):
    """Perform batch normalisation."""
    return keras.layers.normalization.BatchNormalization()(prev_layer)


def dropout(prev_layer, hype_space, for_convolution_else_fc=True):
    """Add dropout after a layer."""
    if for_convolution_else_fc:
        return keras.layers.core.Dropout(
            rate=hype_space['conv_dropout_drop_proba']
        )(prev_layer)
    else:
        return keras.layers.core.Dropout(
            rate=hype_space['fc_dropout_drop_proba']
        )(prev_layer)


def convolution(prev_layer, n_filters, hype_space, force_ksize=None):
    """Basic convolution layer, parametrized by the hype_space."""
    if force_ksize is not None:
        k = force_ksize
    else:
        k = int(round(hype_space['conv_kernel_size']))
    return keras.layers.convolutional.Conv2D(
        filters=n_filters, kernel_size=(k, k), strides=(1, 1),
        padding='same', activation=hype_space['activation'],
        kernel_regularizer=keras.regularizers.l2(
            STARTING_L2_REG * hype_space['l2_weight_reg_mult'])
    )(prev_layer)


def residual(prev_layer, n_filters, hype_space):
    """Some sort of residual layer, parametrized by the hype_space."""
    current_layer = prev_layer
    for i in range(int(round(hype_space['residual']))):
        lin_current_layer = keras.layers.convolutional.Conv2D(
            filters=n_filters, kernel_size=(1, 1), strides=(1, 1),
            padding='same', activation='linear',
            kernel_regularizer=keras.regularizers.l2(
                STARTING_L2_REG * hype_space['l2_weight_reg_mult'])
        )(current_layer)

        layer_to_add = dropout(current_layer, hype_space)
        layer_to_add = convolution(
            layer_to_add, n_filters, hype_space,
            force_ksize=int(round(hype_space['res_conv_kernel_size'])))

        current_layer = keras.layers.add([
            lin_current_layer,
            layer_to_add
        ])
        if hype_space['use_BN']:
            current_layer = bn(current_layer)
    if not hype_space['use_BN']:
        current_layer = bn(current_layer)

    return bn(current_layer)


def auto_choose_pooling(prev_layer, n_filters, hype_space):
    """Deal with pooling in convolution steps."""
    if hype_space['pooling_type'] == 'all_conv':
        current_layer = convolution_pooling(
            prev_layer, n_filters, hype_space)

    elif hype_space['pooling_type'] == 'inception':
        current_layer = inception_reduction(prev_layer, n_filters, hype_space)

    elif hype_space['pooling_type'] == 'avg':
        current_layer = keras.layers.pooling.AveragePooling2D(
            pool_size=(2, 2)
        )(prev_layer)

    else:  # 'max'
        current_layer = keras.layers.pooling.MaxPooling2D(
            pool_size=(2, 2)
        )(prev_layer)

    return current_layer


def convolution_pooling(prev_layer, n_filters, hype_space):
    """
    Pooling with a convolution of stride 2.

    See: https://arxiv.org/pdf/1412.6806.pdf
    """
    current_layer = keras.layers.convolutional.Conv2D(
        filters=n_filters, kernel_size=(3, 3), strides=(2, 2),
        padding='same', activation='linear',
        kernel_regularizer=keras.regularizers.l2(
            STARTING_L2_REG * hype_space['l2_weight_reg_mult'])
    )(prev_layer)

    if hype_space['use_BN']:
        current_layer = bn(current_layer)

    return current_layer


def inception_reduction(prev_layer, n_filters, hype_space):
    """
    Reduction block, vaguely inspired from inception.

    See: https://arxiv.org/pdf/1602.07261.pdf
    """
    n_filters_a = int(n_filters * 0.33 + 1)
    n_filters = int(n_filters * 0.4 + 1)

    conv1 = convolution(prev_layer, n_filters_a, hype_space, force_ksize=3)
    conv1 = convolution_pooling(prev_layer, n_filters, hype_space)

    conv2 = convolution(prev_layer, n_filters_a, hype_space, 1)
    conv2 = convolution(conv2, n_filters, hype_space, 3)
    conv2 = convolution_pooling(conv2, n_filters, hype_space)

    conv3 = convolution(prev_layer, n_filters, hype_space, force_ksize=1)
    conv3 = keras.layers.pooling.MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2), padding='same'
    )(conv3)

    current_layer = keras.layers.concatenate([conv1, conv2, conv3], axis=-1)

    return current_layer
