## Model Setup
Contact: Rahul G. Krishnan (rahul@cs.nyu.edu)

## Setup
* Run `python polysemous_words.py` to setup a folder to evaluate conditional Jacobians

## Code for learning and evaluation:
Each file contains code for learning or evaluating the DLGMs

* [`vae.py`](vae.py)
	* Main file containing an implementation of a DLGM with the option to optimize the variational parameters 
* [`vae_evaluate.py`](vae_evaluate.py)
	* Evaluating learned DLGMs
* [`vae_learn.py`](vae_learn.py)
	* Learning DLGMs
* [`polysemous_words.py`](polysemous_words.py)
	* Evaluating polysemous words
* [`evaluate_vecs.py`](evaluate_vecs.py)
	* Code used in ipython notebook to evaluate Jacobian vectors
