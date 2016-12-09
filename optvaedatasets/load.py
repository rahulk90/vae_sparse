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

def loadDataset(dsetname):
    if dsetname in ['20newsgroups_miao']: #Code provided by Miao et. al 
        return newsgroups._load20news_miao()
    elif dsetname in ['rcv2_miao']:
        return rcv2._loadrcv2_miao()
    elif dsetname in ['wordsim353']:
        return evaluate_wvecs._loadWordSim353()
    elif dsetname in ['scws']:
        return evaluate_wvecs._loadSCWS()
    elif dsetname in ['wikicorp']:
        return wikicorp._loadWikicorp()
    elif dsetname in ['synthetic_ball','synthetic_s']:
        return synthetic._loadSynthetic(dsetname)
    else:
        assert False,'Invalid dataset name'

if __name__=='__main__':
    dset  = loadDataset('swcs')
    import ipdb;ipdb.set_trace()
