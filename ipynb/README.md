This folder contains ipython notebooks used for evaluation. For each
notebook, its purpose is listed below: 

* [`VisualizeTrainingCurves.ipynb`](./bow_text/VisualizeTrainingCurves.ipynb)
    * This ipython notebook plots the training curves 
* [`WikicorpVsFeatures.ipynb`](./bow_text/WikicorpVsFeatures.ipynb)
    * This ipython notebook builds the plot that compares the difference between the held-out perplexity obtained with and without finopt as a function of the number of features in the dataset 
* [`WordEmbeddings.ipynb`](./bow_text/WordEmbeddings.ipynb)
    * This reads the saved Jacobians created by the files `../expt/evaluateWikipedia.py` and `../expt/evaluateConditionalWikipedia` and analyses them (nearest neighbors etc.) 
    * This also creates plots of the log-singular values of the Jacobian matrix
* [`GridSearch_Q.ipynb`](./bow_text/GridSearch_Q.ipynb)
    * This reads the *.outf files saved in `../expt` from running the template python evaluation scripts 
    created by `../expt/getTable_parallel_qvary.py` and creates the relevant latex table