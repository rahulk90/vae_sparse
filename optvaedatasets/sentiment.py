from utils.misc import savePickle,downloadData,extractData
import os,re
from utils.misc import readPickle, savePickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

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
        return readPickle(DIR+'/'+dset+'.pkl')[0]
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
                    labels.append(elem[1])
                assert len(elem)==2,'Bad element'
            return data, np.array(labels)
        dataset_fine = {}
        dataset_fine['train_x'], dataset_fine['train_y'] = split(train) 
        dataset_fine['valid_x'], dataset_fine['valid_y'] = split(valid) 
        dataset_fine['test_x'],  dataset_fine['test_y']  = split(test) 
        dataset_fine['word2idx'] = word2idx
        idx2word = {}
        for w in word2idx:
            idx2word[word2idx[w]] = w
        dataset_fine['idx2word'] = idx2word

        dataset_binary = {} 
        dataset_binary['train_x'], dataset_binary['train_y'] = split(train, binary=True) 
        dataset_binary['valid_x'], dataset_binary['valid_y'] = split(valid, binary=True) 
        dataset_binary['test_x'],  dataset_binary['test_y']  = split(test,  binary=True) 
        dataset_binary['word2idx'] = word2idx
        dataset_binary['idx2word'] = idx2word
        savePickle([dataset_fine],DIR+'/sst_fine.pkl')
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
        raise ValueError('Run preprocess_imdb.py')
    elif os.path.exists(DIR+'/imdb.pkl'):
        print 'Loading...'
        return readPickle(DIR+'/imdb.pkl')[0]
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
        dataset['word2idx']=word2idx
        dataset['idx2word']=idx2word
        dataset['train_x'], dataset['train_y'] = split(train_tup) 
        dataset['valid_x'], dataset['valid_y'] = split(valid_tup) 
        dataset['test_x'], dataset['test_y']   = split(test_tup) 
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
        idx       = np.where(pvec.toarray().squeeze()>0)[0]
        data.append(idx)
        labels.append(1)
    for nvec in negative_vecs:
        idx       = np.where(nvec.toarray().squeeze()>0)[0]
        data.append(idx)
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
    dataset['train_x']    = [data[idx] for idx in train_idx.tolist()] 
    dataset['valid_x']    = [data[idx] for idx in valid_idx.tolist()] 
    dataset['test_x']     = [data[idx] for idx in test_idx.tolist()] 
    dataset['train_y']    = labels[train_idx] 
    dataset['valid_y']    = labels[valid_idx] 
    dataset['test_y']     = labels[test_idx] 
    dataset['idx2word'] = idx2word
    dataset['word2idx'] = word2idx
    savePickle([dataset],DIR+'/dataset.pkl')

def _loadRT():
    DIR = os.path.dirname(os.path.realpath(__file__)).split('inference_introspection')[0]+'inference_introspection/optvaedatasets/sentiment/rotten_tomatoes'
    if not os.path.exists(DIR):
        os.system('mkdir -p '+DIR)
    if not (os.path.exists(DIR+'/dataset.pkl')):
        _setupRT(DIR)
    return readPickle(DIR+'/dataset.pkl')[0]

if __name__=='__main__':
    dataset = _loadStanford('sst_fine')
    #dataset = _loadStanford('sst_binary')
    #dataset = _loadIMDB()
    #dataset = _loadRT()
    import ipdb;ipdb.set_trace()
