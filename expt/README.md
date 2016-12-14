## Experimental Setup
Contact: Rahul G. Krishnan (rahul@cs.nyu.edu)


## Code for setting up experiments
* [`setupExperiments.py`](setupExperiments.py) 
	* Use this with different options (see inside file) for setting up different experiments on different datasets 
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
