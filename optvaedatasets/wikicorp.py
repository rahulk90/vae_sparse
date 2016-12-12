from utils.misc import savePickle,downloadData
import h5py,time,tarfile,os,zipfile
import bz2
from utils.misc import readPickle
from utils.sparse_utils import loadSparseHDF5
import numpy as np
from collections import Counter,OrderedDict
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
"""
def _getData(DIR,locations):
    if not np.all([os.path.exists(DIR+'/'+f) for f in locations]):
        downloadData(DIR, locations)
    for f in locations: 
        os.system('bunzip '+DIR+'/'+f)

def _loadWikicorp():
    DIR = os.path.dirname(os.path.realpath(__file__)).split('opt-vae')[0]+'opt-vae/optvaedatasets/wikicorp'
    if not os.path.exists(DIR):
        os.mkdir(DIR)
    locations = {}
    locations['WestburyLab.wikicorp.201004.txt.bz2'] = 'http://nlp.stanford.edu/data/WestburyLab.wikicorp.201004.txt.bz2'
    if not os.path.exists(DIR+'/WestburyLab.wikicorp.201004.txt'):
        _getData(DIR,locations)
    if not os.path.exists(DIR+'/data.h5') or not os.path.exists(DIR+'/misc.pkl'):
        raise ValueError,'Run ProcessWikicorp.ipynb to setup data.h5'
    else:
        dataset = {}
        dataset['data_type'] = 'bow'
        dataset['train']     = loadSparseHDF5('train',DIR+'/data.h5')
        dataset['valid']     = loadSparseHDF5('valid',DIR+'/data.h5')
        dataset['test']      = loadSparseHDF5('test',DIR+'/data.h5')
        dataset['dim_observations'] = dataset['train'].shape[1]
        objs = readPickle(DIR+'/misc.pkl',nobjects=3)
        """
        For evaluating on WS and SCWS
        """
        dataset['mapIdx']              = objs[0]
        dataset['vocabulary']          = objs[1]
        dataset['vocabulary_singular'] = objs[2]
        return dataset



if __name__=='__main__':
    dataset = _loadWikicorp()
    import ipdb;ipdb.set_trace()
