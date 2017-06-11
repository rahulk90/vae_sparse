import os,re,time
from utils.misc import downloadData,extractData
from utils.misc import readPickle, savePickle, saveHDF5
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import sys

"""
  cleanDataset
  * Use this function to clean up the vocabulary in accordance with what you would expect 
  from the wikipedia data setup
    * replace multiple instances of '-' with just one '-'
    * split up words separated by '\' into two and assign to the same training point 
    * do nothing to the labels 
    * setup new word2idx, idx2word
"""
def cleanDataset(dataset, min_df = 1):
    def recreateDoc(dataidx, idx2word):
        doc         = '' 
        for tnum, tlist in enumerate(dataidx): 
            if tnum>0:
                doc     = doc+'\n'+' '.join([idx2word[idx] for idx in tlist])
            else:
                doc     = ' '.join([idx2word[idx] for idx in tlist])
        return doc
    def arrayToIdxLists(idxarray):
        data  =[]
        for vec in idxarray:
            idxlist   = []
            non_zero_idx = np.where(vec>0.)[0]
            for idx in non_zero_idx.tolist(): 
                count   =  vec[idx]
                idxlist += [idx]*count
            data.append(idxlist)
        return data
    start = time.time()
    doclist_train   = procDoc(recreateDoc(dataset['train_x'], dataset['idx2word'])).split('\n') 
    doclist_valid   = procDoc(recreateDoc(dataset['valid_x'], dataset['idx2word'])).split('\n') 
    doclist_test    = procDoc(recreateDoc(dataset['test_x'], dataset['idx2word'])).split('\n')
    end   = (time.time()-start)/60.
    print 'Cleaning train/valid/test took:',end,' mins'
    ctvec           = CountVectorizer(stop_words='english',analyzer='word',strip_accents='ascii', min_df = min_df)
    ctvec.fit(doclist_train+doclist_valid+doclist_test)
    dataset_new         = {}
    dataset_new['train_x']= arrayToIdxLists(ctvec.transform(doclist_train).toarray())
    dataset_new['valid_x']= arrayToIdxLists(ctvec.transform(doclist_valid).toarray())
    dataset_new['test_x'] = arrayToIdxLists(ctvec.transform(doclist_test).toarray())
    dataset_new['train_y']= dataset['train_y']
    dataset_new['valid_y']= dataset['valid_y']
    dataset_new['test_y'] = dataset['test_y']
    vocab         = ctvec.vocabulary_
    print 'New Vocab:' , len(vocab)
    word2idx      = vocab
    idx2word      = {}
    for w in vocab:
        idx2word[word2idx[w]] = w
    dataset_new['word2idx']   = word2idx
    dataset_new['idx2word']   = idx2word
    return dataset_new

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
        dataset_fine['word2idx'] = word2idx
        idx2word = {}
        for w in word2idx:
            idx2word[word2idx[w]] = w
        dataset_fine['idx2word'] = idx2word
        dataset_fine = cleanDataset(dataset_fine)
        savePickle([dataset_fine],DIR+'/sst_fine.pkl')

        dataset_binary = {} 
        dataset_binary['train_x'], dataset_binary['train_y'] = split(train, binary=True) 
        dataset_binary['valid_x'], dataset_binary['valid_y'] = split(valid, binary=True) 
        dataset_binary['test_x'],  dataset_binary['test_y']  = split(test,  binary=True) 
        dataset_binary['word2idx'] = word2idx
        dataset_binary['idx2word'] = idx2word
        dataset_binary = cleanDataset(dataset_binary)
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
        dataset['train_x'], dataset['train_y'] = split(train_tup) 
        dataset['valid_x'], dataset['valid_y'] = split(valid_tup) 
        dataset['test_x'], dataset['test_y']   = split(test_tup) 
        dataset['word2idx']=word2idx
        dataset['idx2word']=idx2word
        dataset = cleanDataset(dataset, min_df = 10)
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
def procDoc(document):
    ws  = document.replace('-',' ')
    ws  = ws.replace('/',' ')
    ws  = ws.replace(' ',' ')
    ws  = re.sub('[^\w\s]','',ws.strip())
    ws_lower = ws.lower()
    wsd = re.sub(r'\d', '', ws_lower)
    return wsd

def _setupRT(DIR):
    locations = {}
    locations['rt-polaritydata.tar.gz'] = 'http://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz'
    downloadData(DIR, locations)
    extractData(DIR, locations)
    with open(DIR+'/rt-polaritydata/rt-polarity.pos') as f:
        pos = f.read()
    with open(DIR+'/rt-polaritydata/rt-polarity.neg') as f:
        neg = f.read()
    positive = procDoc(pos).split('\n')
    negative = procDoc(neg).split('\n')
    ctvec    = CountVectorizer(stop_words='english',analyzer='word',strip_accents='ascii')
    ctvec.fit(positive+negative)
    positive_vecs = ctvec.transform(positive).toarray()
    negative_vecs = ctvec.transform(negative).toarray()
    data  =[]
    labels=[]
    for pvec in positive_vecs:
        idxlist   = []
        non_zero_idx = np.where(pvec>0.)[0]
        for idx in non_zero_idx.tolist(): 
            count   = pvec[idx]
            idxlist += [idx]*count
        data.append(idxlist)
        labels.append(1)
    for nvec in negative_vecs:
        idxlist   = []
        non_zero_idx = np.where(nvec>0.)[0]
        for idx in non_zero_idx.tolist(): 
            count    = nvec[idx]
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
    dataset['train_x']  = [data[idx] for idx in train_idx.tolist()]
    dataset['valid_x']  = [data[idx] for idx in valid_idx.tolist()]
    dataset['test_x']   = [data[idx] for idx in test_idx.tolist()]
    dataset['train_y']  = labels[train_idx] 
    dataset['valid_y']  = labels[valid_idx] 
    dataset['test_y']   = labels[test_idx] 
    dataset['idx2word'] = idx2word
    dataset['word2idx'] = word2idx
    savePickle([dataset],DIR+'/rt.pkl')

def _loadRT():
    DIR = os.path.dirname(os.path.realpath(__file__)).split('inference_introspection')[0]+'inference_introspection/optvaedatasets/sentiment/rotten_tomatoes'
    if not os.path.exists(DIR):
        os.system('mkdir -p '+DIR)
    if not (os.path.exists(DIR+'/rt.pkl')):
        _setupRT(DIR)
    return readPickle(DIR+'/rt.pkl')[0]

def _setupGlove():
    DIR = os.path.dirname(os.path.realpath(__file__)).split('inference_introspection')[0]+'inference_introspection/optvaedatasets/sentiment/glove'
    if not os.path.exists(DIR):
        os.system('mkdir -p '+DIR)
    locations = {}
    locations['glove.42B.300d.zip'] = 'http://nlp.stanford.edu/data/glove.42B.300d.zip'
    downloadData(DIR, locations)
    extractData(DIR, locations)

def _setupGloveJacobian():
    DIR = os.path.dirname(os.path.realpath(__file__)).split('inference_introspection')[0]+'inference_introspection/optvaedatasets/sentiment/glove'
    if not os.path.exists(DIR+'/glove.42B.300d.txt'):
        _setupGlove()
    WIKIDIR = os.path.dirname(os.path.realpath(__file__)).split('inference_introspection')[0]+'inference_introspection/optvaedatasets/wikicorp'
    if not os.path.exists(DIR+'/glove.h5'):
        dataset = {}
        objs = readPickle(WIKIDIR+'/misc-large.pkl',nobjects=3)
        dataset['mapIdx']              = objs[0]
        dataset['vocabulary']          = objs[1]
        dataset['vocabulary_singular'] = objs[2]
        vocab    = dataset['vocabulary'].tolist()
        vocab2idx= {} 
        for idx, w in enumerate(vocab):
            vocab2idx[w] = idx
        newjacob = np.zeros((len(vocab), 300))
        print 'Loading txt'
        word2vec = {}
        with open(DIR+'/glove.42B.300d.txt') as f:
            alllines = f.readlines()
        print 'Building word->vec'
        VECLEN   = 300
        for idx in xrange(len(alllines)): 
            spt  = alllines[idx].strip().split(' ')
            w    = spt[0]
            vec  = spt[1:] 
            word2vec[w] = vec
            assert len(vec)==VECLEN,'expecting '+str(VECLEN)+', got:'+str(len(vec))

        print 'Checking vocab and building new matrix'
        newjacob    = np.zeros((len(vocab),300))-500
        print 'Not found (idx set to -500):'
        norms       = []
        for idx in xrange(len(vocab)):
            w       = vocab[idx]
            if w in word2vec:
                vec = np.array([float(k) for k in word2vec[w]]) 
                newjacob[idx,:] = vec
                norms.append(np.linalg.norm(vec))
            else:
                print w,',', 
        print '\n'
        results = {}
        results['ejacob']       = newjacob
        saveHDF5(DIR+'/glove-large.h5', results)
        print 'Sanity check (should be different): ',np.mean(norms[:10]), np.mean(norms[-10:])

if __name__=='__main__':
    #sst_fine= _loadStanford('sst_fine')
    sst_bin = _loadStanford('sst_binary')
    imdb    = _loadIMDB()
    rt      = _loadRT() #checked
    _setupGloveJacobian()
    import ipdb;ipdb.set_trace()
