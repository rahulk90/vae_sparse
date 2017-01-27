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
	* The pre-specified options here detail settings used in the paper
* [`getTestResults.py`](getTestResults.py) 
	* Use this to obtain all the saved test time accuracies from the checkpoint files
* [`compileTiming.py`](compileTiming.py) 
	* Run `python compileTiming.py` to get the average per iteration runtimes (saved in checkpoint files) for different datasets and optimization schemes
* [`evaluateConditionalWikipedia.py`](evaluateConditionalWikipedia.py), [`evaluateWikipedia.py`](evaluateWikipedia.py)
	* Setup and save Jacobian vectors
	* Evaluate the properties of the conditional Jacobian 
	* Evaluate on word similarity task
* [`train.py`](train.py)
	* Main training script for DLGMs which accepts a variety of arguments specified in [`parse_args.py`](../optvaeutils/parse_args.py)
* [`getTable_finopt.py`](getTable_finopt.py)
    * Set the `DIR` variable to point to the folder where the checkpoints are created (such as `chkpt-wikicorp-finopt`) 
    * This file compiles the table comparing the results on held out data from training models with and without finopt
* [`getTable_parallel_qvary.py`](getTable_parallel_qvary.py)
    * This experiment should be run when there are no other folders with `chkpt-rcv2...` present in the `expt` directory
    * This file is relevant to compiling results as a function of doing a small grid search over parameters of the inference network 
    * It uses [`template_parallel_qvary.py`](template_parallel_qvary.py) to create files named `tmplt_#.py` where # is a number
    * Each file estimates the train/test perplexity using the best model and writes the results of evaluation as text to files having format `tmplt_#.outf`
