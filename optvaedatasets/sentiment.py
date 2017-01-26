from utils.misc import savePickle,downloadData,extractData
import os,re
from utils.misc import readPickle, savePickle, saveHDF5, loadHDF5
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

def setupDataset(dataset):
    """
    Input format: list of lists. each element in the list is a list of indices corresponding to works in 
        the vocabulary
    Output format: matrix of indices, masks denoting which indices are valid for each datapoint
    """
    MAX     = np.max([len(elem) for elem in dataset])
    indices = np.zeros((len(dataset), MAX))
    mask    = np.zeros((len(dataset), MAX))
    for idx, elem in enumerate(dataset): 
        indices[idx,:len(elem)] = np.sort(np.array(elem)) 
        mask[idx,:len(elem)]    = 1. 
    return indices, mask

"""
Setup Stanford Sentiment Analysis Dataset
"""
def _setupStanford(DIR):
    results = {}
    results['trees.zip'] = 'http://nlp.stanford.edu/sentiment/trainDevTestTrees_PTB.zip'
    downloadData(DIR, results)
    extractData(DIR, results)

def _processStanford(DIR, dset):
    if not os.path.exists(DIR+'/train-rootfine'):
        raise ValueError('Run the code sentiment/sentiment_trees.py first')
    elif os.path.exists(DIR+'/'+dset+'.pkl'):
        vocabdict = readPickle(DIR+'/'+dset+'.pkl')[0]
        dataset   = loadHDF5(DIR+'/'+dset+'.h5')
        for w in vocabdict:
            dataset[w] = vocabdict[w]
        return dataset
    else:
        assert dset in ['sst_fine','sst_binary'],'Invalid value for dset'
        train= readPickle(DIR+'/train-rootfine')[0]
        valid= readPickle(DIR+'/dev-rootfine')[0]
        test = readPickle(DIR+'/test-rootfine')[0]
        word2idx= readPickle(DIR+'/trees/wordMapAll.bin')[0]
        def split(dataset, binary=False):
            data, labels = [], []
            for elem in dataset:
                if binary and elem[1]==2:
                    pass
                else:
                    data.append(elem[0])
                    if binary:
                        labels.append((elem[1]>2)*1)
                    else:
                        labels.append(elem[1])
                assert len(elem)==2,'Bad element'
            return data, np.array(labels)
        dataset_fine = {}
        dataset_fine['train_x'], dataset_fine['train_y'] = split(train) 
        dataset_fine['valid_x'], dataset_fine['valid_y'] = split(valid) 
        dataset_fine['test_x'],  dataset_fine['test_y']  = split(test) 
        dataset_fine['train_x'], dataset_fine['mask_train']   = setupDataset(dataset_fine['train_x'])
        dataset_fine['valid_x'], dataset_fine['mask_valid']   = setupDataset(dataset_fine['valid_x'])
        dataset_fine['test_x'],  dataset_fine['mask_test']    = setupDataset(dataset_fine['test_x'])
        print 'sst_fine train:' , dataset_fine['train_x'].shape, dataset_fine['train_y'].shape, dataset_fine['mask_train'].shape
        print 'sst_fine valid:' , dataset_fine['valid_x'].shape, dataset_fine['valid_y'].shape, dataset_fine['mask_valid'].shape
        print 'sst_fine test :' , dataset_fine['test_x'].shape,  dataset_fine['test_y'].shape,  dataset_fine['mask_test'].shape
        saveHDF5(DIR+'/sst_fine.h5', dataset_fine)
        dataset_fine = {}
        dataset_fine['word2idx'] = word2idx
        idx2word = {}
        for w in word2idx:
            idx2word[word2idx[w]] = w
        dataset_fine['idx2word'] = idx2word
        savePickle([dataset_fine],DIR+'/sst_fine.pkl')

        dataset_binary = {} 
        dataset_binary['train_x'], dataset_binary['train_y'] = split(train, binary=True) 
        dataset_binary['valid_x'], dataset_binary['valid_y'] = split(valid, binary=True) 
        dataset_binary['test_x'],  dataset_binary['test_y']  = split(test,  binary=True) 
        dataset_binary['train_x'], dataset_binary['mask_train']   = setupDataset(dataset_binary['train_x'])
        dataset_binary['valid_x'], dataset_binary['mask_valid']   = setupDataset(dataset_binary['valid_x'])
        dataset_binary['test_x'],  dataset_binary['mask_test']    = setupDataset(dataset_binary['test_x'])
        print 'sst_bin train:' , dataset_binary['train_x'].shape, dataset_binary['train_y'].shape, dataset_binary['mask_train'].shape
        print 'sst_bin valid:' , dataset_binary['valid_x'].shape, dataset_binary['valid_y'].shape, dataset_binary['mask_valid'].shape
        print 'sst_bin test :' , dataset_binary['test_x'].shape,  dataset_binary['test_y'].shape,  dataset_binary['mask_test'].shape
        saveHDF5(DIR+'/sst_binary.h5', dataset_binary)
        dataset_binary  = {}
        dataset_binary['word2idx'] = word2idx
        dataset_binary['idx2word'] = idx2word
        savePickle([dataset_binary],DIR+'/sst_binary.pkl')
        if dset =='sst_fine':
            return dataset_fine
        else:
            return dataset_binary
    return dataset
def _loadStanford(dset):
    DIR = os.path.dirname(os.path.realpath(__file__)).split('inference_introspection')[0]+'inference_introspection/optvaedatasets/sentiment/stanford'
    if not os.path.exists(DIR):
        os.system('mkdir -p '+DIR)
    if not os.path.exists(DIR+'/trees.zip'):
        _setupStanford(DIR)
    return _processStanford(DIR,dset)

"""
Setup IMDB Movie Rating Dataset
"""
def _setupIMDB(DIR):
    locations = {}
    locations['aclImdb.tar.gz'] = 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
    downloadData(DIR, locations)
    extractData(DIR, locations)

def _processIMDB(DIR):
    if not os.path.exists(DIR+'/aclImdb/imdb_splits'):
        raise ValueError('Run code in sentiment/preprocess_imdb.py')
    elif os.path.exists(DIR+'/imdb.pkl'):
        print 'Loading...'
        dataset   = loadHDF5(DIR+'/imdb.h5')
        vocabdict = readPickle(DIR+'/imdb.pkl')[0]
        for k in vocabdict:
            dataset[k] = vocabdict[k]
        return dataset
    else:
        saved     = readPickle(DIR+'/aclImdb/imdb_splits')[0]
        train_valid, test_tup, _, word2idx= saved[0], saved[1], saved[2], saved[3]
        np.random.seed(0)
        idxlst    = np.random.permutation(len(train_valid))
        Ntrain    = int(0.9*len(train_valid))
        train_idx = idxlst[:Ntrain]
        valid_idx = idxlst[Ntrain:]
        train_tup = [train_valid[idx] for idx in train_idx.tolist()] 
        valid_tup = [train_valid[idx] for idx in valid_idx.tolist()] 
        def split(dataset_tup):
            data, labels = [],[]
            for (x,y) in dataset_tup:
                data.append(x)
                labels.append(y)
            return data, np.array(labels)
        idx2word  = {}
        for w in word2idx:
            idx2word[word2idx[w]] = w
        dataset = {}
        dataset['train_x'], dataset['train_y'] = split(train_tup) 
        dataset['valid_x'], dataset['valid_y'] = split(valid_tup) 
        dataset['test_x'], dataset['test_y']   = split(test_tup) 
        dataset['train_x'], dataset['mask_train'] = setupDataset(dataset['train_x'])
        dataset['valid_x'], dataset['mask_valid'] = setupDataset(dataset['valid_x'])
        dataset['test_x'], dataset['mask_test']   = setupDataset(dataset['test_x'])
        print 'IMDB train:' , dataset['train_x'].shape, dataset['train_y'].shape, dataset['mask_train'].shape
        print 'IMDB valid:' , dataset['valid_x'].shape, dataset['valid_y'].shape, dataset['mask_valid'].shape
        print 'IMDB test :' , dataset['test_x'].shape,  dataset['test_y'].shape,  dataset['mask_test'].shape
        saveHDF5( DIR+'/imdb.h5', dataset)
        dataset = {}
        dataset['word2idx']=word2idx
        dataset['idx2word']=idx2word
        savePickle([dataset],DIR+'/imdb.pkl')
        print 'Saved....'
        return dataset
def _loadIMDB():
    DIR = os.path.dirname(os.path.realpath(__file__)).split('inference_introspection')[0]+'inference_introspection/optvaedatasets/sentiment/imdb'
    if not os.path.exists(DIR):
        os.system('mkdir -p '+DIR)
    if not (os.path.exists(DIR+'/aclImdb.tar.gz')):
        _setupIMDB(DIR)
    return _processIMDB(DIR)

"""
Setup Rotten Tomatoes dataset
"""
def _setupRT(DIR):
    locations = {}
    locations['rt-polaritydata.tar.gz'] = 'http://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz'
    downloadData(DIR, locations)
    extractData(DIR, locations)
    with open(DIR+'/rt-polaritydata/rt-polarity.pos') as f:
        pos = f.read()
    with open(DIR+'/rt-polaritydata/rt-polarity.neg') as f:
        neg = f.read()
    def procDoc(document):
        ws  = re.sub('[^\w\s]','', document.strip())
        ws_lower = ws.lower()
        wsd = re.sub(r'\d', '', ws_lower)
        return wsd
    positive = procDoc(pos).split('\n')
    negative = procDoc(neg).split('\n')
    ctvec    = CountVectorizer(stop_words='english',analyzer='word',strip_accents='ascii')
    ctvec.fit(positive)
    ctvec.fit(negative)
    positive_vecs = ctvec.transform(positive)
    negative_vecs = ctvec.transform(negative)
    
    data  =[]
    labels=[]
    for pvec in positive_vecs:
        pos_list  = pvec.toarray().squeeze()
        idxlist   = []
        non_zero_idx = np.where(pos_list>0.)[0]
        for idx in non_zero_idx.tolist(): 
            count   = pos_list[idx]
            idxlist += [idx]*count
        data.append(idxlist)
        labels.append(1)
    for nvec in negative_vecs:
        neg_list  = nvec.toarray().squeeze().tolist()
        idxlist   = []
        non_zero_idx = np.where(neg_list>0.)[0]
        for idx in non_zero_idx.tolist(): 
            count    = neg_list[idx]
            idxlist += [idx]*count
        data.append(idxlist)
        labels.append(0)
    labels = np.array(labels)
    np.random.seed(0)
    idxlist= np.random.permutation(labels.shape[0])
    Ntrain = int(0.7*idxlist.shape[0]) 
    Ntest  = int(0.2*idxlist.shape[0]) 
    Nvalid = idxlist.shape[0]-Ntrain-Ntest
    train_idx = idxlist[:Ntrain]
    test_idx  = idxlist[Ntrain:(Ntrain+Ntest)]
    valid_idx = idxlist[(Ntrain+Ntest):]
    vocab         = ctvec.vocabulary_
    word2idx      = vocab
    idx2word      = {}
    for w in vocab:
        idx2word[word2idx[w]] = w
    dataset = {}
    dataset['train_x'], dataset['mask_train'] = setupDataset([data[idx] for idx in train_idx.tolist()])
    dataset['valid_x'], dataset['mask_valid'] = setupDataset([data[idx] for idx in valid_idx.tolist()])
    dataset['test_x'], dataset['mask_test']   = setupDataset([data[idx] for idx in test_idx.tolist()])
    dataset['train_y']    = labels[train_idx] 
    dataset['valid_y']    = labels[valid_idx] 
    dataset['test_y']     = labels[test_idx] 
    print 'RT train:' , dataset['train_x'].shape, dataset['train_y'].shape, dataset['mask_train'].shape
    print 'RT valid:' , dataset['valid_x'].shape, dataset['valid_y'].shape, dataset['mask_valid'].shape
    print 'RT test :' , dataset['test_x'].shape,  dataset['test_y'].shape,  dataset['mask_test'].shape
    saveHDF5( DIR+'/rt.h5', dataset)
    dataset = {}
    dataset['idx2word'] = idx2word
    dataset['word2idx'] = word2idx
    savePickle([dataset],DIR+'/rt.pkl')

def _loadRT():
    DIR = os.path.dirname(os.path.realpath(__file__)).split('inference_introspection')[0]+'inference_introspection/optvaedatasets/sentiment/rotten_tomatoes'
    if not os.path.exists(DIR):
        os.system('mkdir -p '+DIR)
    if not (os.path.exists(DIR+'/rt.pkl')):
        _setupRT(DIR)
    vocabdict = readPickle(DIR+'/rt.pkl')[0]
    dataset   = loadHDF5(DIR+'/rt.h5')
    for k in vocabdict:
        dataset[k] = vocabdict[k]
    return dataset

if __name__=='__main__':
    sst_fine= _loadStanford('sst_fine')
    sst_bin = _loadStanford('sst_binary')
    imdb    = _loadIMDB()
    rt      = _loadRT()
    import ipdb;ipdb.set_trace()
