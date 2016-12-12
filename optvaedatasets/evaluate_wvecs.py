from utils.misc import savePickle,downloadData
import h5py,time,tarfile,os,zipfile
import numpy as np
from collections import Counter,OrderedDict
import tarfile
"""
Evaluating Vectors in their contexts
------------------------------------
Relevant Publications:
@inproceedings{HuangEtAl2012,
author = {Eric H. Huang and Richard Socher and Christopher D. Manning and Andrew Y. Ng},
title = {{Improving Word Representations via Global Context and Multiple Word Prototypes}},
booktitle = {Annual Meeting of the Association for Computational Linguistics (ACL)},
year = 2012
}

Lev Finkelstein, Evgeniy Gabrilovich, Yossi Matias, 
Ehud Rivlin, Zach Solan, 
Gadi Wolfman, and Eytan Ruppin, "Placing Search 
in Context: The Concept Revisited", ACM Transactions on Information Systems, 20(1):116-131, January 2002 [Abstract / PDF]
"""

def _getData(DIR,locations):
    if not np.all([os.path.exists(DIR+'/'+f) for f in locations]):
        downloadData(DIR, locations)
    for f in locations:
        if 'zip' in f:
            with zipfile.ZipFile(DIR+'/'+f,"r") as zf:
                zf.extractall(DIR)
        elif 'tgz' in f:
            with tarfile.open(DIR+'/'+f,'r') as tf:
                tf.extractall(DIR)
        else:
            print 'Ignoring ',f


def _loadWordSim353():
    DIR = os.path.dirname(os.path.realpath(__file__)).split('inference_introspection')[0]+'inference_introspection/optvaedatasets/evaluate'
    if not os.path.exists(DIR):
        os.mkdir(DIR)
    locations = {}
    locations['wordsim353.zip'] = 'http://www.cs.technion.ac.il/~gabr/resources/data/wordsim353/wordsim353.zip'
    if not os.path.exists(DIR+'/wordsim353.zip'):
        _getData(DIR,locations)

    def fillData(idx, dataset, alllines):
        for line in alllines[1:]:
            ldata    = line.strip().split('\t')
            dataset[idx] = {}
            dataset[idx]['w1'] = ldata[0]
            dataset[idx]['w2'] = ldata[1]
            dataset[idx]['avgrat']   = ldata[2]
            dataset[idx]['indrat']   = np.array([float(k.strip()) for k in ldata[3:]])
            idx +=1 
        return idx
    dataset      = OrderedDict()
    with open(DIR+'/set1.tab') as f:
        alllines1 = f.readlines()
    with open(DIR+'/set2.tab') as f:
        alllines2 = f.readlines()
    dataset      = OrderedDict()
    nextIdx      = fillData(0, dataset, alllines1)
    _            = fillData(nextIdx, dataset, alllines2)
    return dataset

def _loadSCWS():
    DIR = os.path.dirname(os.path.realpath(__file__)).split('inference_introspection')[0]+'inference_introspection/optvaedatasets/evaluate'
    if not os.path.exists(DIR):
        os.mkdir(DIR)
    locations = {}
    locations['SCWS.zip']       = 'http://www-nlp.stanford.edu/~ehhuang/SCWS.zip'
    if not os.path.exists(DIR+'/SCWS.zip'):
        _getData(DIR,locations)
    dataset = OrderedDict() 
    with open(DIR+'/SCWS/ratings.txt') as f:
        alllines = f.readlines()
    for line in alllines:
        ldata = line.split('\t')
        idx   = int(ldata[0])
        dataset[idx] = {}
        dataset[idx]['w1']= ldata[1] 
        dataset[idx]['p1']= ldata[2] 
        dataset[idx]['w2']= ldata[3] 
        dataset[idx]['p2']= ldata[4] 
        dataset[idx]['ctex1']= ldata[5]
        dataset[idx]['ctex2']= ldata[6]
        dataset[idx]['avgrat']= float(ldata[7])
        dataset[idx]['indrat']= np.array([float(k.strip()) for k in ldata[8:]])
        assert len(dataset[idx]['indrat'])==10,'Expecting 10 ratings'
    return dataset

if __name__=='__main__':
    dataset = _loadSCWS()
    dataset  = _loadWordSim353()
    import ipdb;ipdb.set_trace()
