# Inference \& Introspection in Deep Generative Models of Sparse Data

### Contact: Rahul G. Krishnan (rahul@cs.nyu.edu)

## Requirements

* python2.7
* [theano](http://deeplearning.net/software/theano/)
* [theanomodels](https://github.com/clinicalml/theanomodels)

## Overview
* This contains code to learn DLGMs/VAEs on sparse-non negative data (though it may be easily modified for other data types) while optimizing the variational 
parameters predicted by the inference network during learning
* The repository is arranged as follows:
	* [`ipynb`](./ipynb) - Code to visualize plots/simulations/samples from the generative model
	* [`expt`](./expt)  - Folders for experiments
	* [`optvaedatasets`](./optvaedatasets) - Setup for datasets 
	* [`optvaemodels`](./optvaemodels)   - Code for the model 
	* [`optvaeutils`](./optvaeutils)    - Utility functions 
