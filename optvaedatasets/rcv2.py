import h5py,time,tarfile
import numpy as np
from collections import Counter,OrderedDict
from nltk.corpus import stopwords
import os
from utils.sparse_utils import saveSparseHDF5, loadSparseHDF5, readSparseFile
from tqdm import tqdm
from scipy.sparse import coo_matrix,csr_matrix,csc_matrix,hstack
from utils.misc import savePickle
from scipy.io import loadmat

def _loadrcv2_miao():
    """ Dataset created using procedure detailed in Miao.et.al """
    DIR  = os.path.dirname(os.path.realpath(__file__)).split('vae_sparse')[0]+'vae_sparse/optvaedatasets'
    DIR += '/rcv2_miao'
    assert os.path.exists(DIR+'/preprocess.lua'),'See instructions to create dataset.'
    h5file = DIR+'/rcv2.h5'
    if not os.path.exists(h5file):
        flen     = len(open(DIR+'/vocab.new').readlines())
        print 'DIM: ',flen
        np.random.seed(1)
        TRAIN_VALID_MAT = readSparseFile(DIR+'/train.feat', flen, zeroIndexed=False)
        idx = np.random.permutation(TRAIN_VALID_MAT.shape[0])
        VALIDMAT = TRAIN_VALID_MAT[idx[:5000]]
        TRAINMAT = TRAIN_VALID_MAT[idx[5000:]]
        TESTMAT  = readSparseFile(DIR+'/test.feat', flen, zeroIndexed=False) 
        saveSparseHDF5(TRAINMAT,'train', h5file)
        saveSparseHDF5(VALIDMAT,'valid', h5file)
        saveSparseHDF5(TESTMAT, 'test' , h5file)
    dset = {}
    dset['vocabulary']= [k.strip().split(' ')[0] for k in open(DIR+'/vocab.new').readlines()]
    dset['train']     = loadSparseHDF5('train',h5file)
    dset['valid']     = loadSparseHDF5('valid',h5file)
    dset['test']      = loadSparseHDF5('test',h5file)
    dset['dim_observations'] = dset['train'].shape[1]
    dset['data_type'] = 'bow'
    return dset

if __name__=='__main__':
    dset = _loadrcv2_miao()
    import ipdb;ipdb.set_trace()
