## Datasets

## Setup instructions:
For each dataset below, follow the provided instructions to setup each one. 
To be used to train models, the data must be converted into a dictionary that (at the very minimum) has the following keys:
```python
type(dataset)    #dict
dataset['train'] #CSR Matrix Ntrain x Nfeatures
dataset['valid'] #CSR Matrix Nvalid x Nfeatures
dataset['test']  #CSR Matrix Ntest  x Nfeatures
dataset['dim_observations']  #Scalar Nfeatures
dataset['data_type']  #Typically 'bow'
```

### Overview

### RCV2/20newsgroup
* Used in Miao et. al 
```bash
#Naviate to <path>/inference_introspection/optvaedatasets/rcv2_miao/
python download.py
```
* Preprocess the data (requires [Torch](http://torch.ch/))
```bash
th preprocessing.lua #(This step requires Torch to be installed)
#Navigate to <path>/inference_introspection/optvaedatasets/
python rcv2.py
```

```bash
#Download and setup 20newsgroups dataset:
python newsgroups.py 
```

### WordSim353/SCWS
* Datasets that contain pairs of words along with human annotated metric of similarity 
```bash
#Download and setup dataset:
python evaluate_wvecs.py 
```

### Wikicorp 
* Based on the Wikipedia datadump of Huang et. al
* Requires WordSim/SCWS in the previous step to be have been setup
* Download and setup dataset from raw Wikipedia text
```bash
#In <path>/inference_introspection/optvaedatasets/
python wikicorp.py #Downloads the dataset into wikicorp/ 
#Navigate to <path>/inference_introspection/optvaedatasets/wikicorp/
```
* Install [inflect](https://pypi.python.org/pypi/inflect), [gensim](https://radimrehurek.com/gensim/) and [nltk](http://www.nltk.org/)
```bash
#Parses the text and converts it into a BOW representation
python tokenizer.py 
```
* Limits the vocabulary (change parameters to get different variants of this dataset)
```bash
ipython trust *.ipynb
ipython notebook ProcessWikicorp.ipynb
```
* Final check to ensure the data can be loaded from python
```bash
python wikicorp.py 
```

### Synthetic
* Synthetic datasets obtained from scikit-learn's page on [Dimensionality Reduction](http://scikit-learn.org/stable/auto_examples/manifold/plot_compare_methods.html) 
```bash
#Setup by running
python synthetic.py
```

### Sentiment Analysis
* There are three datasets to setup using `python sentiment.py`
* The script will have to be re-run several times interspersed with preprocessing steps
* Two files must be run from the `sentiment` directory : `preprocess_imdb.py` and `sentiment_trees.py`. You will be prompted about when these must be run
* When all the instructions have been followed, you should be able to load the four datasets successfully and will halt at the command `import ipdb;ipdb.set_trace()` when running the file `sentiment.py`
