# Hyperopt for solving CIFAR-100 with a convolutional neural network (CNN) built with Keras and TensorFlow, GPU backend

Auto (meta) optimizing a neural net (and its architecture) on the CIFAR-100 dataset (fine labels). This code could be easily transferred to another vision dataset or to any other machine learning task. 

To run the search, do: `python3 optimize.py`. You might want to look at `requirements.py` and install some of them manually to acquire GPU acceleration (e.g.: installing TensorFlow and Keras especially by yourself). 

Optimization results will continuously be saved in the `results/` folder (sort files to take best result as human-readable text). 
Also, the results are pickled to `results.pkl` to be able to resume the TPE meta-optimization process later simply by running the program again with `python3 optimize.py`. 

If you want to learn more about Hyperopt, you'll probably want to watch that [video](https://www.youtube.com/watch?v=tteE_Vtmrv4) made by the creator of Hyperopt. Also, if you want to run the model on the CIFAR-10 dataset, you must edit the file `neural_net.py`. 


## The Deep Convolutional Neural Network Model

Here is a basic overview of the model. I implemented it in such a way that Hyperopt will try to change the shape of the layers and remove or replace some of them according to some pre-parametrized ideas that I have got. Therefore, not only the learning rate is changed with hyperopt, but a lot more parameters. 

```python

space = {
    # This loguniform scale will multiply the learning rate, so as to make
    # it vary exponentially, in a multiplicative fashion rather than in
    # a linear fashion, to handle his exponentialy varying nature:
    'lr_rate_mult': hp.loguniform('lr_rate_mult', -0.5, 0.5),
    # L2 weight decay:
    'l2_weight_reg_mult': hp.loguniform('l2_weight_reg_mult', -1.3, 1.3),
    # Batch size fed for each gradient update
    'batch_size': hp.quniform('batch_size', 100, 700, 5),
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

# Here is one possible outcome for this stochastic space, let's plot that:
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


```

<p align="center">
  <img src="model_demo.png">
</p>
