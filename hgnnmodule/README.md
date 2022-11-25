# HGNN Module 
This file explains the usage of the heterogeneous graph convolution module.

## Prerequisites
Prerequisites for using the module is that the datasetmodule is set up correctly and that an EHR graph
has been created from the MIMIC-IV OMOP DCM mapped dataset as described in the README located
in the `datasetmodule/README.md` file. Follow all steps of that readme before continuing with this.

### Setup Neptune
This step should have been already completed by following the readme from the datasetmodule.
Insert the neptune project id into the configuration file at `hgnnmodule/config.ini` (remove .example
extension or create a new file).

## Running experiments
The latest graph created by the datasetmodule will be automatically used for experiments run from 
the HGNN Module.  
The script `hgnn_main.py` takes two optional arguments as an input:
- --use_hpo: When this argument is present, a 100 iteration parameter optimization task will be run
targeted the task of diagnosis prediction against the specified by the labels from the graph. (
notice, this can take many hours to run).
- -g: -g -1 means use cpu for training. If omitted we always use gpu.

If the argument `--use_hpo` is omitted, a single hyperparameter configuration is use as specified
by the configuration file `hgnnmodule/config.ini`. The editable variables are:
- use_logging: Whether to log info to Neptune or not
- experiment_name: Experimental name will be logged to neptune
- optimizer: Choose between Adam or standard SGD
- learning_rate: The learning rate of the optimizer
- weight_decay: The weight decay of the optimizer
- dropout: Node dropout
- h_dim: Dimension of hidden gcn layers
- n_layers: Number of hidden gcn layers
- max_epoch: Max number of epochs
- patience: Used for early stopping experiment. If no improvement with respect to the loss on the evaluation data
has been seen within patience iterations, then the experiment terminates.
- fanout: Sampling strategy variable from the GraphSage paper
- mini_batch_flag: Whether to use batch training or not
- batch_size: The size of mini batches

A single hyperparameter optimization could be started as:
`python hgnn_main.py -g`  

A hyperparameter optimization experiment can be started as:  
`python hgnn_main.py -g --use_hpo`  
The hyperparameter values to tune over can be edited from the file `hgnnmodule/utils/hpo`
