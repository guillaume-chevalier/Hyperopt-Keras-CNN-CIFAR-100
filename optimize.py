
"""Auto-optimizing a neural network with Hyperopt (TPE algorithm)."""

from neural_net import build_and_optimize_cnn, build_model

from keras.utils import plot_model
import keras.backend as K
from hyperopt import hp, tpe, fmin, Trials

import pickle
import traceback


__author__ = "Guillaume Chevalier"
__copyright__ = "Copyright 2017, Guillaume Chevalier"
__license__ = "Apache License 2.0"


space = {
    # This loguniform scale will multiply the learning rate, so as to make
    # it vary exponentially, in a multiplicative fashion rather than in
    # a linear fashion, to handle his exponentialy varying nature:
    'lr_rate_mult': hp.loguniform('lr_rate_mult', -0.5, 0.5),
    # L2 weight decay:
    'l2_weight_reg_mult': hp.loguniform('l2_weight_reg_mult', -1.3, 1.3),
    # Batch size fed for each gradient update
    'batch_size': hp.quniform('batch_size', 100, 450, 5),
    # Choice of optimizer:
    'optimizer': hp.choice('optimizer', ['Adam', 'Nadam', 'RMSprop']),
    # Coarse labels importance for weights updates:
    'coarse_labels_weight': hp.uniform('coarse_labels_weight', 0.1, 0.7),
    # Uniform distribution in finding appropriate dropout values, conv layers
    'conv_dropout_drop_proba': hp.uniform('conv_dropout_proba', 0.0, 0.35),
    # Uniform distribution in finding appropriate dropout values, FC layers
    'fc_dropout_drop_proba': hp.uniform('fc_dropout_proba', 0.0, 0.6),
    # Use batch normalisation at more places?
    'use_BN': hp.choice('use_BN', [False, True]),

    # Use a first convolution which is special?
    'first_conv': hp.choice(
        'first_conv', [None, hp.choice('first_conv_size', [3, 4])]
    ),
    # Use residual connections? If so, how many more to stack?
    'residual': hp.choice(
        'residual', [None, hp.quniform(
            'residual_units', 1 - 0.499, 4 + 0.499, 1)]
    ),
    # Let's multiply the "default" number of hidden units:
    'conv_hiddn_units_mult': hp.loguniform('conv_hiddn_units_mult', -0.6, 0.6),
    # Number of conv+pool layers stacked:
    'nb_conv_pool_layers': hp.choice('nb_conv_pool_layers', [2, 3]),
    # Starting conv+pool layer for residual connections:
    'conv_pool_res_start_idx': hp.quniform('conv_pool_res_start_idx', 0, 2, 1),
    # The type of pooling used at each subsampling step:
    'pooling_type': hp.choice('pooling_type', [
        'max',  # Max pooling
        'avg',  # Average pooling
        'all_conv',  # All-convolutionnal: https://arxiv.org/pdf/1412.6806.pdf
        'inception'  # Inspired from: https://arxiv.org/pdf/1602.07261.pdf
    ]),
    # The kernel_size for convolutions:
    'conv_kernel_size': hp.quniform('conv_kernel_size', 2, 4, 1),
    # The kernel_size for residual convolutions:
    'res_conv_kernel_size': hp.quniform('res_conv_kernel_size', 2, 4, 1),

    # Amount of fully-connected units after convolution feature map
    'fc_units_1_mult': hp.loguniform('fc_units_1_mult', -0.6, 0.6),
    # Use one more FC layer at output
    'one_more_fc': hp.choice(
        'one_more_fc', [None, hp.loguniform('fc_units_2_mult', -0.6, 0.6)]
    ),
    # Activations that are used everywhere
    'activation': hp.choice('activation', ['relu', 'elu'])
}


def plot_base_and_best_models():
    """Plot a demo model."""
    space_base_demo_to_plot = {
        'lr_rate_mult': 1.0,
        'l2_weight_reg_mult': 1.0,
        'batch_size': 300,
        'optimizer': 'Nadam',
        'coarse_labels_weight': 0.2,
        'conv_dropout_drop_proba': 0.175,
        'fc_dropout_drop_proba': 0.3,
        'use_BN': True,

        'first_conv': 4,
        'residual': 4,
        'conv_hiddn_units_mult': 1.0,
        'nb_conv_pool_layers': 3,
        'conv_pool_res_start_idx': 0.0,
        'pooling_type': 'inception',
        'conv_kernel_size': 3.0,
        'res_conv_kernel_size': 3.0,

        'fc_units_1_mult': 1.0,
        'one_more_fc': 1.0,
        'activation': 'elu'
    }
    space_best_model = {
        "activation": "elu",
        "batch_size": 320.0,
        "coarse_labels_weight": 0.3067103474295116,
        "conv_dropout_drop_proba": 0.25923531175521264,
        "conv_hiddn_units_mult": 1.5958302613876916,
        "conv_kernel_size": 3.0,
        "conv_pool_res_start_idx": 0.0,
        "fc_dropout_drop_proba": 0.4322253354921089,
        "fc_units_1_mult": 1.3083964454436132,
        "first_conv": 3,
        "l2_weight_reg_mult": 0.41206755600055983,
        "lr_rate_mult": 0.6549347353077412,
        "nb_conv_pool_layers": 3,
        "one_more_fc": None,
        "optimizer": "Nadam",
        "pooling_type": "avg",
        "res_conv_kernel_size": 2.0,
        "residual": 3.0,
        "use_BN": True
    }

    model = build_model(space_base_demo_to_plot)
    plot_model(model, to_file='model_demo.png', show_shapes=True)
    print("Saved base model visualization to model_demo.png.")
    K.clear_session()
    del model

    model = build_model(space_best_model)
    plot_model(model, to_file='model_best.png', show_shapes=True)
    print("Saved best model visualization to model_best.png.")
    K.clear_session()
    del model


def run_a_trial():
    """Run one TPE meta optimisation step and save its results."""
    max_evals = nb_evals = 1

    print("Attempt to resume a past training if it exists:")

    try:
        # https://github.com/hyperopt/hyperopt/issues/267
        trials = pickle.load(open("results.pkl", "rb"))
        print("Found saved Trials! Loading...")
        max_evals = len(trials.trials) + nb_evals
        print("Rerunning from {} trials to add another one.".format(
            len(trials.trials)))
    except:
        trials = Trials()
        print("Starting from scratch: new trials.")

    best = fmin(
        build_and_optimize_cnn,
        space,
        algo=tpe.suggest,
        trials=trials,
        max_evals=max_evals
    )
    pickle.dump(trials, open("results.pkl", "wb"))

    print("\nOPTIMIZATION STEP COMPLETE.\n")
    print("Best results yet (note that this is NOT calculated on the 'loss' "
          "metric despite the key is 'loss' - we rather take the negative "
          "best accuracy throughout learning as a metric to minimize):")
    print(best)


if __name__ == "__main__":
    """Plot the model and run the optimisation forever (and saves results)."""

    print("Plot a demo model that would represent "
          "a quite normal model (or a bit more huge), "
          "and then the best model.")

    plot_base_and_best_models()

    print("Here we train many models, one after the other. "
          "Note that hyperopt has support for cloud "
          "distributed training using MongoDB.")

    print("\nYour results will be saved in the folder named 'results/'. "
          "You can sort that alphabetically and take the greatest one. "
          "As you run the optimization, results are consinuously saved into a "
          "'results.pkl' file, too. Re-running optimize.py will resume "
          "the meta-optimization.\n")

    while True:
        try:
            run_a_trial()
        except Exception as err:
            err_str = str(err)
            print(err_str)
            traceback_str = str(traceback.format_exc())
            print(traceback_str)
