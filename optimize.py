
"""Auto-optimizing a neural network with Hyperopt (tpe algorithm)."""

from neural_net import cnn

from hyperopt import hp, tpe, fmin
from keras.optimizers import Adam, Nadam, RMSprop


if __name__ == "__main__":
    """Run the optimisation. Might take a few days."""

    space = {
        # This loguniform scale will multiply the learning rate, so as to make
        # it vary exponentially, in a multiplicative fashion rather than in
        # a linear fashion, to handle his exponentialy varying nature:
        'lr_rate_mult': hp.loguniform('lr_rate_mult', -0.5, 0.5),
        # Choice of optimizer:
        'optimizer': hp.choice('optimizer', [Adam, Nadam, RMSprop]),
        # Uniform distribution in finding appropriate dropout values
        'dropout_drop_proba': hp.uniform('dropout_drop_proba', 0.0, 0.3),
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

    # TODO: verify "use_residual"

    # Note: here we train many models, one after the other, but hyperopt has
    # support for cloud distributed training using MongoDB:
    best = fmin(
        cnn,
        space,
        algo=tpe.suggest,
        max_evals=50
    )

    print(best)
