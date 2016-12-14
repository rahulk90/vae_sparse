## Experimental Setup

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
