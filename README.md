# Pre-Optimizing VAEs

Rahul G. Krishnan (rahul@cs.nyu.edu)

Requirements
------------

python2.7

theano

theanomodels

Repository Structure
--------------------

* ipynb - Code to visualize plots/simulations/samples from the generative model
* expt  - Folders for experiments
* optvaedatasets - Setup for datasets (some datasets are private and can only be accessed on internal machines)
* optvaemodels   - Implementation of VAE with different schemes for learning
* optvaeutils    - Utility functions for project

Implementation Details
----------------------

* optvaeutils/optimizer.py contains an implementation of the ADAM optimizer
  - This has been modified to incorporate a different learning rate for optimizing the parameters of the generative model
* optvaemodels/vae.py contains the implementation of the VAE class
* optvaemodels/vae_learning.py contains the functions for learning
* optvaemodels/vae_evaluate.py contains the functions for evaluation
* optvaedatasets/load.py contains a wrapper to load different datasets. Individual datasets are created/loaded with the other files in optvaedatasets. See optvaedatasets/README.md for more details. 
