## Datasets
Contact: Rahul G. Krishnan (rahul@cs.nyu.edu)

## Setup instructions:
For each dataset below, follow the provided instructions to setup the datasets

### Overview

### RCV2/20newsgroup
* Used in Miao et. al 
```bash
#Go to inference_introspection/optvaedatasets/rcv2_miao/
python download.py
th preprocessing.lua #(This step requires Torch)
```

```bash
#Download and setup 20newsgroups dataset:
python newsgroups.py 
```

```bash
#Download and setup RCV2 dataset:
python rcv2.py 
```

### Wikicorp 
* Based on the Wikipedia datadump of Huang et. al

### WordSim353/SCWS
* Datasets that contain pairs of words along with human annotated metric of similarity 
```bash
#Download and setup dataset:
python evaluate_wvecs.py 
```
