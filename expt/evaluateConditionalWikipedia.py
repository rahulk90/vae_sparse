#Load Wikipedia Model
import sys,os
import time
from collections import OrderedDict
import re
from scipy.stats import spearmanr
sys.path.append('../')
import numpy as np
from theano import config
import itertools,time
from utils.misc import readPickle,savePickle
from utils.misc import getConfigFile, loadHDF5, saveHDF5, createIfAbsent
from utils.sparse_utils import loadSparseHDF5, saveSparseHDF5
from optvaeutils.viz import getName,stitchMNISTSamples
from sklearn.feature_extraction.text import TfidfTransformer
from scipy.spatial.distance import cdist
from optvaemodels.evaluate_vecs import expectedJacobian, expectedJacobianProbs,expectedJacobianEnergy
from optvaemodels.evaluate_vecs import conditionalJacobianProbs, conditionalJacobian, conditionalJacobianEnergy
from scipy.sparse import csr_matrix,csc_matrix,dok_matrix
from   optvaemodels.vae import VAE
import optvaemodels.vae_learn as VAE_learn
import optvaemodels.vae_evaluate as VAE_evaluate
import optvaemodels.vae_learn as VAE_learn
from   optvaedatasets.load import loadDataset 

if len(sys.argv)!=2:
    raise ValueError,'Bad input: python evaluateConditionalWikipedia.py <pl-2-none/pl-2-finopt>'
MODEL_TO_USE = sys.argv[-1].strip()
dataset = loadDataset('wikicorp')

additional_attrs        = {}
tfidf                   = TfidfTransformer(norm=None)
tfidf.fit(dataset['train'])
additional_attrs['idf'] = tfidf.idf_
    
models, epochval        = OrderedDict(), OrderedDict()

models['pl-2-none']     = './chkpt-wikicorp-none/VAE_lr-8_0e-04-ph-400-ds-100-pl-2-ql-2-nl-relu-bs-500-ep-250-plr-1_0e-02-ar-0-otype-none-ns-100-om-adam-qs-standard-etype-mlp-p_up-10-q_up-10-pr-normal-ll-mult-itype-tfidfl20_01_-uid'
epochval['pl-2-none']   = '30'

models['pl-2-finopt']   = './chkpt-wikicorp-finopt/VAE_lr-8_0e-04-ph-400-ds-100-pl-2-ql-2-nl-relu-bs-500-ep-250-plr-1_0e-02-ar-0-otype-finopt-ns-100-om-adam-qs-standard-etype-mlp-p_up-10-q_up-10-pr-normal-ll-mult-itype-tfidfl20_01_-uid'
epochval['pl-2-finopt'] = '30'

models['pl-0-none']   = './chkpt-wikicorp-none/VAE_lr-8_0e-04-ph-400-ds-100-pl-0-ql-2-nl-relu-bs-500-ep-100-plr-1_0e-02-ar-0-otype-none-ns-100-om-adam-qs-standard-etype-mlp-p_up-10-q_up-10-pr-normal-ll-mult-itype-tfidfl20_01_-uid'
epochval['pl-0-none'] = '30'

models['pl-0-finopt']   = './chkpt-wikicorp-finopt/VAE_lr-8_0e-04-ph-400-ds-100-pl-0-ql-2-nl-relu-bs-500-ep-100-plr-1_0e-02-ar-0-otype-finopt-ns-100-om-adam-qs-standard-etype-mlp-p_up-10-q_up-10-pr-normal-ll-mult-itype-tfidfl20_01_-uid'
epochval['pl-0-finopt'] = '20'
if MODEL_TO_USE not in models:
    raise ValueError, MODEL_TO_USE+' not found'

def setupDocs(polydocs, vocabulary, vocabulary_singular):
    V                = len(vocabulary)
    word2Idx         = dict([(w,i) for i,w in enumerate(vocabulary)])
    word2IdxSingular = dict([(w,i) for i,w in enumerate(vocabulary_singular)])
    result           = {}
    for word in polydocs:
        result[word] = {}
        for context in polydocs[word]:
            result[word][context] = {}
            context_doc = re.sub('\W+',' ',polydocs[word][context].lower())
            words_ctex = [w.strip() for w in context_doc.split()]
            result[word][context] = np.zeros((1,V)) 
            for w in words_ctex:
                if w in word2Idx:
                    result[word][context][0,word2Idx[w]]+=1
                elif w in word2IdxSingular:
                    result[word][context][0,word2IdxSingular[w]]+=1
            print word, context,result[word][context].min(), result[word][context].max(), result[word][context].sum()
    return result
polydocs = {}
polydocs['fire']  = {}
polydocs['fire']['burn']   = open('./docs_context/fire_burn.txt').read()
polydocs['fire']['layoff'] = open('./docs_context/fire_layoff.txt').read()
polydocs['bank']  = {}
polydocs['bank']['money']  = open('./docs_context/bank_money.txt').read() 
polydocs['bank']['river']  = open('./docs_context/bank_river.txt').read() 
polydocs['crane'] = {}
polydocs['crane']['bird'] = open('./docs_context/crane_bird.txt').read()
polydocs['crane']['construction'] = open('./docs_context/crane_construction.txt').read()
polydocs_x        = setupDocs(polydocs, dataset['vocabulary'], dataset['vocabulary_singular'])
def runInference(vae, X):
    if params['opt_type']=='none':
        _,mu,logvar,_   = vae.inference0(X=X.astype(config.floatX))
    elif params['opt_type']=='finopt':
        _,mu,logvar,_   = vae.inferencef(X=X.astype(config.floatX))
    else:
        raise ValueError,'Bad value'+params['opt_type']
    return mu,logvar
SAVEDIR = './evalWikicorp-conditional/'
def sample(mu,logvar):
    z  = (mu.ravel() + np.exp(0.5*logvar).ravel()*np.random.randn(vae.params['dim_stochastic'],)).astype('float32')
    return z
createIfAbsent(SAVEDIR)
for mname in [MODEL_TO_USE]:
    print 'Model: ',mname
    pfile = models[mname].split('uid')[0]+'uid-config.pkl'
    params= readPickle(pfile)[0]
    suffix= '-EP'+str(epochval[mname])+'-params.npz'
    rfile = models[mname]+suffix
    assert os.path.exists(rfile),'not found'
    params['EVALUATION'] = True
    vae   = VAE(params, paramFile=pfile, reloadFile=rfile, additional_attrs = additional_attrs)
    for word in polydocs_x:
        for context in polydocs_x[word]:
            input = polydocs_x[word][context].astype('float32')
            mu,logvar = runInference(vae, input) 

            np.random.seed(1)
            nsamples  = 50
            cjacob = 0.
            for ctr in range(nsamples):
                z = sample(mu,logvar)
                cjacob +=conditionalJacobian(vae, z)
            cjacob /= float(nsamples)

            np.random.seed(1)
            cjacob_probs  = 0.
            for ctr in range(nsamples):
                z = sample(mu,logvar)
                cjacob_probs   +=conditionalJacobianProbs(vae, z)
            cjacob_probs/=float(nsamples)

            np.random.seed(1)
            cjacob_energy  = 0.
            for k in range(nsamples):
                z = sample(mu,logvar)
                cjacob_energy  +=conditionalJacobianEnergy(vae, z)
            cjacob_energy/=float(nsamples)

            fname = SAVEDIR+word+'-'+context+'-jacob.h5'
            saveHDF5(fname,{'cjacob':cjacob,'cjacob_probs':cjacob_probs,'cjacob_energy':cjacob_energy})
            print 'saved...',word,context
    print 'Saved'
print 'Done'
import ipdb;ipdb.set_trace()