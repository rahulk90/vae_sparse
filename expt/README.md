## Experimental Setup

## Notation
Herein, the notation `finopt` referes to optimizing the variational parameters at train time 
and `none` refers to regular training of the models

## Workflow
* Call `python train.py` with a variety of options `-ds 200` (stochastic dimensions), `-otype finopt` (optimize variational parameters) and `-ns 200` (with 200 updates using ADAM) to train models
* Checkpoints and model parameters will be saved to folders with format `chkpt-<datasetname>` where the save frequency may be specified using the flag as `-sfreq 10`

## Code for setting up experiments
* [`setupExperiments.py`](setupExperiments.py) 
	* Use this with different options (see inside file) for setting up different experiments on different datasets 
	* The pre-specified options are settings used in the paper
* [`evaluate_timing.py`](evaluate_timing.py) 
    * Get the average per batch runtimes (saved in checkpoint files) for different datasets and optimization scheme
* [`evaluate_jac.py`](evaluate_jac.py) 
    * Evaluate the Jacobian matrix for different models 
* [`evaluate_finopt_table.py`](evaluate_finopt_table.py) 
    * Print the table comparing models trained with and without optimizing psi
* [`train.py`](train.py)
	* Main training script for DLGMs which accepts a variety of arguments specified in [`parse_args.py`](../optvaeutils/parse_args.py)
