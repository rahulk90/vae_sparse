from utils.misc import savePickle,downloadData,extractData
import h5py,time,tarfile,os,zipfile
from utils.misc import readPickle
from utils.sparse_utils import loadSparseHDF5
import numpy as np
from collections import Counter,OrderedDict

"""
Setup Stanford Sentiment Analysis Dataset
"""
def _setupStanford(DIR):
    results = {}
    results['trees.zip'] = 'http://nlp.stanford.edu/sentiment/trainDevTestTrees_PTB.zip'
    downloadData(DIR, results)
    extractData(DIR, results)

def _loadStanford(dset):
    DIR = os.path.dirname(os.path.realpath(__file__)).split('inference_introspection')[0]+'inference_introspection/optvaedatasets/sentiment/stanford'
    if not os.path.exists(DIR):
        os.system('mkdir -p '+DIR)
    if not os.path.exists(DIR+'/trees.zip'):
        _setupStanford(DIR)
    dataset = {}
    if dset == 'sst_fine':
        pass
    elif dset=='sst_binary':
        pass
    else:
        raise ValueError('Invalid choice of dataset specified')
    return dataset

"""
Setup IMDB Movie Rating Dataset
"""
def _setupIMDB(DIR):
    locations = {}
    locations['aclImdb.tar.gz'] = 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
    downloadData(DIR, locations)
    extractData(DIR, locations)

def _loadIMDB():
    DIR = os.path.dirname(os.path.realpath(__file__)).split('inference_introspection')[0]+'inference_introspection/optvaedatasets/sentiment/imdb'
    if not os.path.exists(DIR):
        os.system('mkdir -p '+DIR)
    if not os.path.exists(DIR+'/aclImdb.tar.gz'):
        _setupIMDB(DIR)

"""
Setup Rotten Tomatoes dataset
"""
def _setupRT(DIR):
    if not (os.path.exists(DIR+'/train.tsv.zip') and os.path.exists(DIR+'/test.tsv.zip')):
        raise ValueError('Download and place train.tsv.zip and test.tsv.zip in sentiment/rotten_tomatoes/ from Kaggle(https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews/data?test.tsv.zip)')
    locations = {}
    locations['train.tsv.zip'] = 'None'
    locations['test.tsv.zip'] = 'None'
    extractData(DIR, locations)

def _loadRT():
    DIR = os.path.dirname(os.path.realpath(__file__)).split('inference_introspection')[0]+'inference_introspection/optvaedatasets/sentiment/rotten_tomatoes'
    if not os.path.exists(DIR):
        os.system('mkdir -p '+DIR)
    if not (os.path.exists(DIR+'/train.tst.zip') and os.path.exists(DIR+'/test.tst.zip')):
        _setupRT(DIR)

if __name__=='__main__':
    dataset = _loadStanford('sst_fine')
    dataset = _loadStanford('sst_binary')
    dataset = _loadIMDB()
    dataset = _loadRT()
