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
The data
