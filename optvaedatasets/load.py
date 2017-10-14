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
import wikicorp

def loadDataset(dsetname):
    if dsetname in ['20newsgroups']: #Code provided by Miao et. al 
        return newsgroups._load20news_miao()
    elif dsetname in ['rcv2']:
        return rcv2._loadrcv2_miao()
    elif dsetname in ['wikicorp']:
        return wikicorp._loadWikicorp()
    elif dsetname in ['wikicorp_1000','wikicorp_5000','wikicorp_10000']:
        return wikicorp._loadWikicorpSubset(int(dsetname.split('_')[1]))
    else:
        assert False,'Invalid dataset name: '+str(dsetname)

if __name__=='__main__':
    #dset  = loadDataset('wikicorp')
    dset  = loadDataset('rcv2')
    import ipdb;ipdb.set_trace()
