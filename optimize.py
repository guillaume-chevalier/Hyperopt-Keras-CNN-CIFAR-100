
"""Auto-optimizing a neural network with Hyperopt (TPE algorithm)."""


from neural_net import build_and_optimize_cnn, build_model

from keras.utils import plot_model
import keras.backend as K
from hyperopt import hp, tpe, fmin, Trials

import pickle


__author__ = "Guillaume Chevalier"
__copyright__ = "Copyright 2017, Guillaume Chevalier"
__license__ = "Apache License 2.0"


space = {
    # This loguniform scale will multiply the learning rate, so as to make
    # it vary exponentially, in a multiplicative fashion rather than in
    # a linear fashion, to handle his exponentialy varying nature:
    'lr_rate_mult': hp.loguniform('lr_rate_mult', -0.5, 0.5),
    # Choice of optimizer:
    'optimizer': hp.choice('optimizer', ['Adam', 'Nadam', 'RMSprop']),
    # Uniform distribution in finding appropriate dropout values
    'dropout_drop_proba': hp.uniform('dropout_drop_proba', 0.0, 0.25),
    # Use batch normalisation at more places?
    'use_BN': hp.choice('use_BN', [False, True]),

    # Use residual connections? If so, how many more to stack?
    'residual': hp.choice(
        'residual', [None, hp.choice(
            'residual_units', [1, 2, 3]
        )]
    ),
    # Let's multiply the "default" number of hidden units:
    'hidden_units_mult': hp.loguniform('hidden_units_mult', -0.5, 0.5),
    # Number of conv+pool layers stacked:
    'nb_conv_pool_layers': hp.choice('nb_conv_pool_layers', [2, 3]),
    # Use all-conlolutional pooling (stride 2), or else max pooling:
    'use_allconv_pooling': hp.choice('use_allconv_pooling', [False, True]),
    # The kernel_size for convolutions:
    'conv_kernel_size': hp.choice('conv_kernel_size', [2, 3]),

    # Amount of fully-connected units after convolution feature map
    'fc_units_mult': hp.loguniform('fc_units_1_mult', -0.5, 0.5),
}


def plot_average_model():
    """Plot a demo model."""
    space_normal_demo_to_plot = {
        'lr_rate_mult': 1.0,
        'optimizer': 'Adam',
        'dropout_drop_proba': 0.0,
        'use_BN': True,
        'residual': 3,
        'hidden_units_mult': 1.0,
        'nb_conv_pool_layers': 3,
        'use_allconv_pooling': True,
        'conv_kernel_size': 3,
        'fc_units_mult': 1.0,
    }

    model = build_model(space_normal_demo_to_plot)
    plot_model(model, to_file='model_demo.png', show_shapes=True)
    print("Saved model visualization to model_demo.png.")

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
    print("Best results yet:")
    print(best)


if __name__ == "__main__":
    """Plot the model and run the optimisation forever (and saves results)."""

    print("Plot a demo model that would represent "
          "a quite normal model (or a bit more huge).")

    plot_average_model()

    print("Here we train many models, one after the other. "
          "Note that hyperopt has support for cloud "
          "distributed training using MongoDB.")

    print("\nYour results will be saved in the folder named 'results/'. "
          "You can sort that alphabetically and take the greatest one. "
          "As you run the optimization, results are consinuously saved into a "
          "'results.pkl' file, too. Re-running optimize.py will resume "
          "the meta-optimization.\n")

    while True:
        run_a_trial()
