"""
Code to evaluate vectors
"""
import numpy as np
from polysemous_words import polysemous_words
import operator
from sklearn.metrics.pairwise import pairwise_distances
from sklearn import cluster
import os,sys

""" 3 kinds of Jacobians"""
def conditionalJacobian(vae, z):
    return vae.jacobian_logprobs(z.ravel())

def expectedConditionalJacobian(vae, mu, logvar, nsamples = 20, individual = []):
    np.random.seed(1)
    jvec = 0.
    for k in range(nsamples):
        jvec_ind = conditionalJacobian(vae, 
                (mu.ravel() + np.exp(0.5*logvar).ravel()*np.random.randn(vae.params['dim_stochastic'],)).astype('float32'))
        individual.append(jvec_ind)
        jvec+=jvec_ind
    jvec /= nsamples
    return jvec
def expectedJacobian(vae, nsamples =1000):
    np.random.seed(1)
    jvec = 0.
    for k in range(nsamples):
        if k%100==0:
            print 'Estimated ',k,' times'
        jvec += conditionalJacobian(vae, np.random.randn(vae.params['dim_stochastic'],).astype('float32'))
    jvec /= nsamples
    return jvec
def modeJacobian(vae):
    return conditionalJacobian(vae, np.zeros((vae.params['dim_stochastic'],)).astype('float32'))

#Grad wrt Probs
def conditionalJacobianProbs(vae, z):
    return vae.jacobian_probs(z.ravel())
def expectedJacobianProbs(vae, nsamples =1000):
    np.random.seed(1)
    jvec = 0.
    for k in range(nsamples):
        if k%100==0:
            print 'Estimated ',k,' times'
        jvec += conditionalJacobianProbs(vae, np.random.randn(vae.params['dim_stochastic'],).astype('float32'))
    jvec /= nsamples
    return jvec
def modeJacobianProbs(vae):
    return conditionalJacobianProbs(vae, np.zeros((vae.params['dim_stochastic'],)).astype('float32'))

#Grad wrt Energy
def conditionalJacobianEnergy(vae, z):
    return vae.jacobian_energy(z.ravel())
def expectedJacobianEnergy(vae, nsamples =1000):
    np.random.seed(1)
    jvec = 0.
    for k in range(nsamples):
        if k%100==0:
            print 'Estimated ',k,' times'
        jvec += conditionalJacobianEnergy(vae, np.random.randn(vae.params['dim_stochastic'],).astype('float32'))
    jvec /= nsamples
    return jvec
def modeJacobianEnergy(vae):
    return conditionalJacobianEnergy(vae, np.zeros((vae.params['dim_stochastic'],)).astype('float32'))
"""
Functions of Works
"""
def wordIdx(words,vocab):
    if type(vocab) is not list:
        vocab = vocab.tolist()
    idxlist={}
    for w in words:
        idxlist[w] = None
        if w in vocab:
            idxlist[w] = vocab.index(w)
    return idxlist
def getWordSimilarity(jacob, wordlist, vocabulary,metric):
    assert metric in ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan','braycurtis', 'canberra', 
            'chebyshev', 'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis', 'matching'],'Bad Metric: '+str(metric) 
    """Return words that are near each of the words in wordlist"""
    dist_mat = pairwise_distances(jacob,metric=metric)
    word_idx = wordIdx(wordlist, vocabulary)
    results  = {}
    for w in wordlist:
        idx    = word_idx[w]
        if idx is not None:
            results[w] = [vocabulary[k] for k in np.argsort(dist_mat[idx])[:7]]
    return results

def clusterEmbeddings(jacob,  n_clusters, vocabulary):
    algorithm = cluster.SpectralClustering(n_clusters=n_clusters, eigen_solver='arpack',affinity="nearest_neighbors")
    algorithm.fit(jacob)
    if hasattr(algorithm, 'labels_'):
        y_pred = algorithm.labels_.astype(np.int)
    else:
        y_pred = algorithm.predict(X)
    word_clusters = {}
    for k in np.unique(y_pred):
        word_clusters[k] = ','.join([vocabulary[idx] for idx in np.where(y_pred==k)[0]])
    return word_clusters

def evaluateVectors(vae, topK=10):
    dim_obs, dim_stoc = vae.params['dim_observations'], vae.params['dim_stochastic']
    assert jacob.shape[0]==dim_obs and jacob.shape[1]==dim_stoc,'Shape mismatch'
    assert hasattr(vae,'reconstruct'),'Model needs to have a reconstruct function'
    assert hasattr(model,'jacobian_logprobs'),'Model needs to have a function to evaluate the jacobian'
    jacob_mode = modeJacobian(vae)
    stdev_jacob= np.std(jacob_mode,axis=0)
    assert stdev_jacob.shape[0]==dim_stoc,'Bad dimension'
    useful_dim = np.argsort(stdev_jacob)[-topK:].tolist()
    mode_z     = np.zeros((vae.params['dim_stochastic']))
    results    = {}
    for dim in useful_dims:
        results[dim] = {}
        for dim_val in np.linspace(-1,1,num=10).tolist():
            z         = np.copy(mode_z)
            z[dim]    = dim_val
            jacob_dim = conditionalJacobian(vae, z)
            recons_dim= vae.reconstruct(z)
            results[dim][dim_val] = (jacob_dim,recons_dim)
    return results

def evaluateWordVectors(jacob, dset, 
                        wordlist = ['weapon','medical','companies','define','israel','book'],
                        n_clusters= 8,
                        metric='cosine'): #was euclidian
    vocabulary      = dset['vocabulary']
    word_similarity = getWordSimilarity(jacob, wordlist, vocabulary, metric)
    word_clusters   = clusterEmbeddings(jacob, n_clusters, vocabulary)
    return word_similarity, word_clusters

def evaluateConditionalWordVectors(vae, dset, metric='euclidean'):
    dim_obs    = vae.params['dim_observations']
    dataset    = dset['test']
    rand_doc   = dataset[0].toarray()
    vocabulary = dset['vocabulary']
    if isinstance(vocabulary, np.ndarray):
        vocabulary = vocabulary.tolist()
    assert rand_doc.shape[0]==1 and rand_doc.shape[1]==dim_obs,'Shape mismatch'
    #Evaluate conditional word vectors 
    results    = {}
    for word in polysemous_words:
        results[word] = {}
        for context in polysemous_words[word]:
            results[word][context] = {}
            wlist         = polysemous_words[word][context]
            idx_to_add_to = []
            for w in wlist:
                if w in vocabulary:
                    idx_to_add_to.append(vocabulary.index(w))
            rdoc = np.copy(rand_doc)
            rdoc[0,idx_to_add_to] += 10
            _,mu,logvar   = vae.inference(rdoc.astype('float32'))
            jacob         = expectedConditionalJacobian(vae, mu, logvar)
            results[word][context]['random'] = getWordSimilarity(jacob, [word], vocabulary, metric).values()

            empty_doc     = np.zeros_like(rand_doc)
            empty_doc[0,idx_to_add_to] += 10
            _,mu,logvar   = vae.inference(empty_doc.astype('float32'))
            jacob         = expectedConditionalJacobian(vae, mu, logvar)
            results[word][context]['empty']  = getWordSimilarity(jacob, [word], vocabulary,metric).values()
    return results

def visualizePolysemousWords(vae, dset):
    dim_obs    = vae.params['dim_observations']
    dataset    = dset['test']
    rand_doc   = dataset[0].toarray()
    vocabulary = dset['vocabulary']
    if isinstance(vocabulary, np.ndarray):
        vocabulary = vocabulary.tolist()
    assert rand_doc.shape[0]==1 and rand_doc.shape[1]==dim_obs,'Shape mismatch'
    #Evaluate conditional word vectors 
    results    = {}
    for word in polysemous_words:
        results[word] = [] 
        if word not in vocabulary:
            continue
        print 'Processing:' ,word
        #Find the word in dataset
        idx = vocabulary.index(word) 
        docs=  dataset[np.where(np.array(dataset[:,idx].toarray()).squeeze()>1)[0]]
        word_jacob    = []
        print 'Word: ',word,' Docs: ',docs.shape
        #For each of the documents, estimate the expected jacobian and extract the word vector
        for didx,doc in enumerate(docs):
            if didx%10==0:
                print '(',didx,')',
            doc_np = np.array(doc.toarray())
            _,mu,logvar   = vae.inference(doc_np.astype('float32'))
            _    = expectedConditionalJacobian(vae, mu, logvar, individual = word_jacob, nsamples = 1)
        print '\n'
        #Select the word vector from the jacobian
        results[word] = np.concatenate([k[[idx],:] for k in word_jacob],axis=0)
    return results

def evaluateSWCS():
    dataset = loadDataset('scws')
    raise NotImplementedError,' Not implemented'

def evaluateWsim353(vae):
    dataset    = loadDataset('wordsim353')
    raise NotImplementedError,' Not implemented'

if __name__=='__main__':
    print 'done'
