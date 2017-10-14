from utils.misc import savePickle,downloadData
import h5py,time,tarfile,os,zipfile
import bz2
from utils.misc import readPickle
from utils.sparse_utils import loadSparseHDF5
import numpy as np
from collections import Counter,OrderedDict

def _getData(DIR,locations):
    if not np.all([os.path.exists(DIR+'/'+f) for f in locations]):
        downloadData(DIR, locations)
    for f in locations: 
        os.system('bunzip '+DIR+'/'+f)

def _loadWikicorp():
    DIR = os.path.dirname(os.path.realpath(__file__)).split('vae_sparse')[0]+'vae_sparse/optvaedatasets/wikicorp'
    if not os.path.exists(DIR):
        os.mkdir(DIR)
    locations = {}
    locations['WestburyLab.wikicorp.201004.txt.bz2'] = 'http://nlp.stanford.edu/data/WestburyLab.wikicorp.201004.txt.bz2'
    if not os.path.exists(DIR+'/WestburyLab.wikicorp.201004.txt'):
        _getData(DIR,locations)
    if not os.path.exists(DIR+'/data-learning.h5') or not os.path.exists(DIR+'/misc-learning.pkl'):
        raise ValueError,'Run ProcessWikicorp(-learning/-large).ipynb to setup data.h5'
    else:
        dataset = {}
        dataset['data_type'] = 'bow'
        dataset['train']     = loadSparseHDF5('train',DIR+'/data-learning.h5')
        dataset['valid']     = loadSparseHDF5('valid',DIR+'/data-learning.h5')
        dataset['test']      = loadSparseHDF5('test',DIR+'/data-learning.h5')
        dataset['dim_observations'] = dataset['train'].shape[1]
        objs = readPickle(DIR+'/misc-learning.pkl',nobjects=3)
        dataset['mapIdx']              = objs[0]
        dataset['vocabulary']          = objs[1]
        dataset['vocabulary_singular'] = objs[2]
        return dataset

def _loadWikicorpSubset(kval):
    assert kval in [1000, 5000, 10000, 15000],'Bad value: '+str(kval) 
    DIR = os.path.dirname(os.path.realpath(__file__)).split('vae_sparse')[0]+'vae_sparse/optvaedatasets/wikicorp'
    assert type(kval) is int,'Expecting kval as int'
    h5file = DIR+'/data-learning.h5'
    pklfile= DIR+'/misc-learning.pkl'
    assert os.path.exists(h5file) and os.path.exists(pklfile),'Please run _loadWikicorp to generate data.h5'
    #Load Wikicorp raw data
    train= loadSparseHDF5('train',h5file).tocsc()
    valid= loadSparseHDF5('valid',h5file).tocsc()
    test = loadSparseHDF5('test',h5file).tocsc()
    objs = readPickle(DIR+'/misc-learning.pkl',nobjects=3)
    vocabulary          = objs[1]
    
    sumfeats = np.array(train.sum(0)).squeeze()
    idx_sort = np.argsort(sumfeats)
    idx_to_keep = idx_sort[-kval:]
    dset = {}
    dset['vocabulary'] = [vocabulary[idx] for idx in idx_to_keep.squeeze().tolist()]
    train_tmp          = train[:,idx_to_keep].tocsr()
    valid_tmp          = valid[:,idx_to_keep].tocsr()
    test_tmp           = test[:,idx_to_keep].tocsr()
    #Use documents w/ atleast five words in it
    train_cts_idx = np.where(np.array(train_tmp.sum(1)).squeeze()>5)[0]
    valid_cts_idx = np.where(np.array(valid_tmp.sum(1)).squeeze()>5)[0]
    test_cts_idx  = np.where(np.array(test_tmp.sum(1)).squeeze()>5)[0]
    dset['train']      = train_tmp[train_cts_idx] 
    dset['valid']      = valid_tmp[valid_cts_idx]
    dset['test']       = test_tmp[test_cts_idx]
    dset['dim_observations'] = dset['train'].shape[1]
    dset['data_type']  = 'bow'
    return dset

if __name__=='__main__':
    dataset = _loadWikicorp()
    dataset = _loadWikicorpSubset(5000)
    import ipdb;ipdb.set_trace()
