import h5py,time,tarfile
import numpy as np
from collections import Counter,OrderedDict
import os
from tqdm import tqdm
from scipy.sparse import coo_matrix,csr_matrix,csc_matrix,hstack
from utils.sparse_utils import saveSparseHDF5, loadSparseHDF5
from utils.misc import savePickle,downloadData
from scipy.io import loadmat

#Dataset Files
import newsgroups
import rcv2
import evaluate_wvecs
import wikicorp
import synthetic
import sentiment

def loadDataset(dsetname):
    if dsetname in ['20newsgroups']: #Code provided by Miao et. al 
        return newsgroups._load20news_miao()
    elif dsetname in ['rcv2']:
        return rcv2._loadrcv2_miao()
    elif dsetname in ['wordsim353']:
        return evaluate_wvecs._loadWordSim353()
    elif dsetname in ['scws']:
        return evaluate_wvecs._loadSCWS()
    elif dsetname in ['wikicorp']:
        return wikicorp._loadWikicorp()
    elif dsetname in ['wikicorp_large']:
        return wikicorp._loadWikicorpLarge()
    elif dsetname in ['synthetic_ball','synthetic_s']:
        return synthetic._loadSynthetic(dsetname)
    elif dsetname in ['wikicorp_1000','wikicorp_5000','wikicorp_10000']:
        return wikicorp._loadWikicorpSubset(int(dsetname.split('_')[1]))
    elif dsetname in ['imdb']:
        return sentiment._loadIMDB()
    elif dsetname in ['rotten_tomatoes']:
        return sentiment._loadRT()
    elif dsetname in ['sst_fine','sst_binary']:
        return sentiment._loadStanford(dsetname)
    elif dsetname in ['largewikivocab']:
        #Keep the vocabulary for wikipedia to include subset of words for imdb, rotten tomatoes, sst, wordsim and scws
        return largewikivocab() 
    else:
        assert False,'Invalid dataset name: '+str(dsetname)

def largewikivocab():
    pass

if __name__=='__main__':
    dset  = loadDataset('scws')
    dset  = loadDataset('wordsim353')
    dset  = loadDataset('wikicorp_large')
    import ipdb;ipdb.set_trace()
