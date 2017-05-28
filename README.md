# Hyperopt for solving CIFAR-100 with a convolutional neural network (CNN) built with Keras and TensorFlow, GPU backend

Auto (meta) optimizing a neural net (and its architecture) on the CIFAR-100 dataset (fine labels). This code could be easily transferred to another vision dataset or to any other machine learning task. 

To run the search, do: `python3 optimize.py`. You might want to look at `requirements.py` and install some of them manually to acquire GPU acceleration (e.g.: installing TensorFlow and Keras especially by yourself). 

Optimization results will continuously be saved in the `results/` folder (sort files to take best result as human-readable text). 
Also, the results are pickled to `results.pkl` to be able to resume the TPE meta-optimization process later simply by running the program again with `python3 optimize.py`. 

If you want to learn more about Hyperopt, you'll probably want to watch that [video](https://www.youtube.com/watch?v=tteE_Vtmrv4) made by the creator of Hyperopt. Also, if you want to run the model on the CIFAR-10 dataset, you must edit the file `neurak_net.py`. 


## The Deep Convolutional Neural Network Model

Here is a basic overview of the model. I implemented it in such a way that Hyperopt will try to change the shape of the layers and remove or replace some of them according to some pre-parametrized ideas that I have got. Therefore, not only the learning rate is changed with hyperopt, but a lot more parameters. 

```python

    space = {
        'lr_rate_mult': 1.0,
        'l2_weight_reg_mult': 1.0,
        'optimizer': 'Adam',
        'dropout_drop_proba': 0.0,
        'use_BN': True,
        'use_first_4x4': True,
        'residual': 3,
        'hidden_units_mult': 1.0,
        'nb_conv_pool_layers': 3,
        'use_allconv_pooling': False,
        'conv_kernel_size': 3,
        'fc_units_mult': 1.0,
    }

```

Note: each hyperparameter is described with a comment in `optimize.py`.

<p align="center">
  <img src="model_demo.png">
</p>
