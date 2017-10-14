# On the challenges of learning with inference networks on sparse, high-dimensional data

## Overview
This contains code to learn DLGMs/VAEs on sparse-non negative 
data (though it may be easily modified for other data types) while optimizing the variational 
parameters predicted by the inference network during learning. It implements the models and techniques detailed in the paper: 
```
On the challenges in learning with inference networks on sparse, high-dimensional data 
Rahul G. Krishnan, Dawen Liang, Matthew Hoffman
```

## Contact
* For questions, email: [Rahul G. Krishnan](mailto:rahulgk@mit.edu)

## Requirements
* python2.7
* numpy, scipy, nltk, gensim
* [theano](http://deeplearning.net/software/theano/)
* [theanomodels](https://github.com/clinicalml/theanomodels)

## Setup 

The repository is arranged as follows:
* [`ipynb`](./ipynb) - Code to visualize plots/simulations/samples from the generative model
* [`expt`](./expt)   - Folders for experiments
* [`optvaedatasets`](./optvaedatasets) - Setup for datasets 
* [`optvaemodels`](./optvaemodels)     - Code for the model 
* [`optvaeutils`](./optvaeutils)       - Utility functions 

## Dataset Format
To run the model on your own data, you will have 
to specify a dataset as follows. See [newsgroups.py](./optvaedatasets/newsgroups.py) for an example of
how to setup the dataset from scratch. 
```
    dset = {}
    dset['vocabulary']= vocab # array of len V containing vocabulary
    dset['train']     = train_matrix #scipy sparse tensor of size Ntrain x dim_features
    dset['valid']     = valid_matrix #scipy sparse tensor of size Nvalid x dim_features
    dset['test']      = test_matrix #scipy sparse tensor of size Ntest x dim_features 
    dset['dim_observations'] = dset['train'].shape[1]
    dset['data_type'] = 'bow'
```
