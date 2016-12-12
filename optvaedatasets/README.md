## Datasets
Contact: Rahul G. Krishnan (rahul@cs.nyu.edu)

## Setup instructions:
For each dataset below, follow the provided instructions to setup each of the datasets

### Overview

### RCV2/20newsgroup
* Used in Miao et. al 
```bash
#Go to <path>/inference_introspection/optvaedatasets/rcv2_miao/
python download.py
th preprocessing.lua #(This step requires Torch to be installed)
#Go to <path>/inference_introspection/optvaedatasets/
python rcv2.py
```

```bash
#Download and setup 20newsgroups dataset:
python newsgroups.py 
```

```bash
#Download and setup RCV2 dataset:
python rcv2.py 
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
```bash
#Download and setup dataset:
python wikicorp.py #Downloads the dataset 
#Go to <path>/inference_introspection/optvaedatasets/wikicorp/
#Parses the text and converts it into a BOW representation
python tokenizer.py #(This step requires gensim)
ipython trust *.ipynb
#Uses the BOW representation and limits the vocabulary (change parameters to get different variants of this dataset)
ipython notebook ProcessWikicorp.ipynb
#Final check to ensure the data can be loaded from python
python wikicorp.py 
```
