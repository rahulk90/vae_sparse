# Pre-Optimizing VAEs

Rahul G. Krishnan (rahul@cs.nyu.edu)

Requirements
------------

python2.7

[theano](http://deeplearning.net/software/theano/)

[theanomodels](https://github.com/clinicalml/theanomodels)


Repository Structure
--------------------

* ipynb - Code to visualize plots/simulations/samples from the generative model
* expt  - Folders for experiments
* optvaedatasets - Setup for datasets 
* optvaemodels   - Code for the model 
* optvaeutils    - Utility functions 

Implementation Details
----------------------

* optvaeutils/optimizer.py contains an implementation of the ADAM optimizer
  - Optimizer modified to incorporate a different learning rate for optimizing the parameters of the generative model (vs the inference network)
* optvaemodels/vae.py: An implementation of the DLGM/VAE
* optvaedatasets/load.py contains a wrapper to load different datasets. Individual datasets are created/loaded with the other files in optvaedatasets. See optvaedatasets/README.md for more details. 
