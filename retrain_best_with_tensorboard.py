
"""Retrain on the best training parameters with TensorBoard monitoring."""


from neural_net import build_and_train, TENSORBOARD_DIR
from utils import print_json, load_best_hyperspace

from keras.layers.core import K

import os


__author__ = "Vooban Inc."
__copyright__ = "Copyright 2017, Vooban Inc."
__license__ = "MIT License"
# See: https://github.com/Vooban/Hyperopt-Keras-CNN-CIFAR-100/blob/master/LICENSE"

if __name__ == "__main__":
    """Retrain best model with TensorBoard. Also save best weights."""
    space_best_model = load_best_hyperspace()
    print("Hyperspace:")
    print_json(space_best_model)

    model, model_name, results, log_path = build_and_train(
        space_best_model,
        save_best_weights=True,
        log_for_tensorboard=True
    )

    print("Model Name:", model_name)

    print(
        "Note: results 'json' file not saved to 'results/' since it is now "
        "available in TensorBoard. See above console output for json-styled "
        "results."
    )

    print("Model summary:")
    model.summary()

    print("TensorBoard logs directory:", log_path)

    print(
        "You may now want to run this command to launch TensorBoard:\n"
        "tensorboard --logdir={}".format(TENSORBOARD_DIR))

    K.clear_session()
    del model
