#Load Wikipedia Model
import sys,os
from collections import OrderedDict
from scipy.stats import spearmanr
sys.path.append('../')
import numpy as np
from theano import config
import itertools,time
from utils.misc import readPickle,savePickle
from utils.misc import getConfigFile, loadHDF5, saveHDF5, createIfAbsent
from utils.sparse_utils import loadSparseHDF5, saveSparseHDF5
from optvaeutils.viz import getName
from sklearn.feature_extraction.text import TfidfTransformer
from scipy.spatial.distance import cdist
from optvaemodels.evaluate_vecs import expectedJacobian, conditionalJacobian,expectedJacobianProbs,conditionalJacobianProbs,expectedJacobianEnergy
from scipy.sparse import csr_matrix,csc_matrix,dok_matrix
from   optvaemodels.vae import VAE
import optvaemodels.vae_learn as VAE_learn
import optvaemodels.vae_evaluate as VAE_evaluate
import optvaemodels.vae_learn as VAE_learn
from   optvaedatasets.load import loadDataset 

if len(sys.argv)!=2:
    raise ValueError,'Bad input: python evaluateWikipedia.py <pl-2-none/pl-2-finopt>'
MODEL_TO_USE = sys.argv[-1].strip()
dataset = loadDataset('wikicorp')
scws = loadDataset('scws')
wsim = loadDataset('wordsim353')

additional_attrs        = {}
tfidf                   = TfidfTransformer(norm=None)
tfidf.fit(dataset['train'])
additional_attrs['idf'] = tfidf.idf_
    
models, epochval        = OrderedDict(), OrderedDict()

models['pl-2-none']     = './results_dec19/chkpt-wikicorp-none/VAE_lr-8_0e-04-ph-400-ds-100-pl-2-ql-2-nl-relu-bs-500-ep-20-plr-1_0e-02-ar-0-otype-none-ns-100-om-adam-etype-mlp-ll-mult-itype-tfidfl20_01_-uid'
epochval['pl-2-none']   = '20'

models['pl-2-finopt']   = './results_dec19/chkpt-wikicorp-finopt/VAE_lr-8_0e-04-ph-400-ds-100-pl-2-ql-2-nl-relu-bs-500-ep-20-plr-1_0e-02-ar-0-otype-finopt-ns-100-om-adam-etype-mlp-ll-mult-itype-tfidfl20_01_-uid'
epochval['pl-2-finopt'] = '20'

models['pl-0-none']   = './results_dec19/chkpt-wikicorp-none/VAE_lr-8_0e-04-ph-400-ds-100-pl-0-ql-2-nl-relu-bs-500-ep-20-plr-1_0e-02-ar-0-otype-none-ns-100-om-adam-etype-mlp-ll-mult-itype-tfidfl20_01_-uid'
epochval['pl-0-none'] = '20'

models['pl-0-finopt']   = './results_dec19/chkpt-wikicorp-finopt/VAE_lr-8_0e-04-ph-400-ds-100-pl-0-ql-2-nl-relu-bs-500-ep-20-plr-1_0e-02-ar-0-otype-finopt-ns-100-om-adam-etype-mlp-ll-mult-itype-tfidfl20_01_-uid'
epochval['pl-0-finopt'] = '20'
if MODEL_TO_USE not in models:
    raise ValueError, MODEL_TO_USE+' not found'

def setupDatasets(dset, vocabulary, vocabulary_singular):
    V                = len(vocabulary)
    word2Idx         = dict([(w,i) for i,w in enumerate(vocabulary)])
    word2IdxSingular = dict([(w,i) for i,w in enumerate(vocabulary_singular)])
    for k in dset:
        dset[k]['w1_idx'] = None 
        dset[k]['w2_idx'] = None
        w1 = dset[k]['w1'].lower().replace('-','')
        w2 = dset[k]['w2'].lower().replace('-','')
        #Word1
        if w1 in word2Idx:
            dset[k]['w1_idx'] = word2Idx[w1]
        elif w1 in word2IdxSingular:
            dset[k]['w1_idx'] = word2IdxSingular[w1]
        else:
            print 'Cannot match:',w1
        #Word2
        if w2 in word2Idx:
            dset[k]['w2_idx'] = word2Idx[w2]
        elif w2 in word2IdxSingular:
            dset[k]['w2_idx'] = word2IdxSingular[w2]
        else:
            print 'Cannot match:',w2
        #For SCWS, create context documents
        if 'ctex1' in dset[k]:
            if 'ctex2' not in dset[k]:
                raise ValueError,'Expecting context2'
            dset[k]['ctex1_vec'] = np.zeros((1,V)) 
            dset[k]['ctex2_vec'] = np.zeros((1,V)) 
            words_ctex1 = [w.strip().lower().replace('-','') for w in dset[k]['ctex1'].split()]
            for word in words_ctex1:
                if word in word2Idx:
                    dset[k]['ctex1_vec'][0,word2Idx[word]]+=1
                elif word in word2IdxSingular:
                    dset[k]['ctex1_vec'][0,word2IdxSingular[word]]+=1
            words_ctex2 = [w.strip().lower().replace('-','') for w in dset[k]['ctex1'].split()]
            for word in words_ctex2:
                if word in word2Idx:
                    dset[k]['ctex2_vec'][0,word2Idx[word]]+=1
                elif word in word2IdxSingular:
                    dset[k]['ctex2_vec'][0,word2IdxSingular[word]]+=1

setupDatasets(wsim, dataset['vocabulary'], dataset['vocabulary_singular'])
setupDatasets(scws, dataset['vocabulary'], dataset['vocabulary_singular'])

def runInference(vae, X):
    if params['opt_type']=='none':
        _,mu,_,_   = vae.inference0(X=X.astype(config.floatX))
    elif params['opt_type']=='finopt':
        _,mu,_,_   = vae.inferencef(X=X.astype(config.floatX))
    else:
        raise ValueError,'Bad value'+params['opt_type']
    return mu
SAVEDIR = './evalWikicorp-exponly/'
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
    ejacob         = expectedJacobian(vae, nsamples=400)
    #Derivative wrt exp(logprobs)
    ejacob_probs   = expectedJacobianProbs(vae, nsamples=400)
    #Derivative wrt E
    ejacob_energy  = expectedJacobianEnergy(vae, nsamples=400)
    fname = SAVEDIR+mname+'-ejacob.h5'
    saveHDF5(fname,{'ejacob':ejacob,'ejacob_probs':ejacob_probs,'ejacob_energy':ejacob_energy})
    results = {}
    results['wsim'] = []
    results['wsim_probs'] = []
    results['wsim_energy'] = []
    for idx,item in enumerate(wsim):
        if wsim[item]['w1_idx'] is None or wsim[item]['w2_idx'] is None:
            pass
        else:
            v1 = ejacob[[wsim[item]['w1_idx']],:]
            v2 = ejacob[[wsim[item]['w2_idx']],:]
            results['wsim'].append((cdist(v1,v2,metric='cosine').ravel()[0], float(wsim[item]['avgrat']))) 
            v1 = ejacob_probs[[wsim[item]['w1_idx']],:]
            v2 = ejacob_probs[[wsim[item]['w2_idx']],:]
            results['wsim_probs'].append((cdist(v1,v2,metric='cosine').ravel()[0], float(wsim[item]['avgrat']))) 
            v1 = ejacob_energy[[wsim[item]['w1_idx']],:]
            v2 = ejacob_energy[[wsim[item]['w2_idx']],:]
            results['wsim_energy'].append((cdist(v1,v2,metric='cosine').ravel()[0], float(wsim[item]['avgrat']))) 
    fname = SAVEDIR+mname+'-wsim-results.h5'
    saveHDF5(fname,{'wsim':np.array(results['wsim']), 'wsim_probs':np.array(results['wsim_probs']), 
        'wsim_energy':np.array(results['wsim_energy'])})
    results['scws_ejacob'] = []
    results['scws_ejacob_probs'] = []
    results['scws_ejacob_energy'] = []
    for iidx, item in enumerate(scws):
        if scws[item]['w1_idx'] is None or scws[item]['w2_idx'] is None:
            pass
        else:
            v1 = ejacob[[scws[item]['w1_idx']],:]
            v2 = ejacob[[scws[item]['w2_idx']],:]
            results['scws_ejacob'].append((cdist(v1,v2,metric='cosine').ravel()[0], float(scws[item]['avgrat']))) 
            v1 = ejacob_probs[[scws[item]['w1_idx']],:]
            v2 = ejacob_probs[[scws[item]['w2_idx']],:]
            results['scws_ejacob_probs'].append((cdist(v1,v2,metric='cosine').ravel()[0], float(scws[item]['avgrat']))) 
            v1 = ejacob_energy[[scws[item]['w1_idx']],:]
            v2 = ejacob_energy[[scws[item]['w2_idx']],:]
            results['scws_ejacob_energy'].append((cdist(v1,v2,metric='cosine').ravel()[0], float(scws[item]['avgrat']))) 
        if iidx % 10 == 0: 
            print '(',iidx,')'
            fname = SAVEDIR+mname+'-scws-intermediate.h5'
            computeSpearman = lambda res: np.array(spearmanr(res[:,0].max()+1-res[:,0],res[:,1]))
            spman_wsim= computeSpearman(np.array(results['wsim']).astype(float)) 
            spman_wsim_probs= computeSpearman(np.array(results['wsim_probs']).astype(float)) 
            spman_wsim_energy= computeSpearman(np.array(results['wsim_energy']).astype(float)) 
            spmanscws_e= computeSpearman(np.array(results['scws_ejacob']).astype(float)) 
            spmanscws_e_probs= computeSpearman(np.array(results['scws_ejacob_probs']).astype(float)) 
            spmanscws_e_energy= computeSpearman(np.array(results['scws_ejacob_energy']).astype(float)) 
            print mname
            print 'WSIM: ',np.array(spman_wsim)
            print 'WSIM (Probs): ',np.array(spman_wsim_probs)
            print 'WSIM (Energy): ',np.array(spman_wsim_energy)
            print 'SCWS (E): ',np.array(spmanscws_e)
            print 'SCWS (E) Probs: ',np.array(spmanscws_e_probs)
            print 'SCWS (E) Energy: ',np.array(spmanscws_e_energy)
            if os.path.exists(fname):
                os.remove(fname)
            results_int = {}
            for k in results:
                results_int[k] = np.array(results[k])
            saveHDF5(fname,results_int)
    fname = SAVEDIR+mname+'-scws-intermediate.h5'
    spman_wsim= computeSpearman(np.array(results['wsim']).astype(float)) 
    spman_wsim_probs= computeSpearman(np.array(results['wsim_probs']).astype(float)) 
    spman_wsim_energy= computeSpearman(np.array(results['wsim_energy']).astype(float)) 
    spmanscws_e= computeSpearman(np.array(results['scws_ejacob']).astype(float)) 
    spmanscws_e_probs= computeSpearman(np.array(results['scws_ejacob_probs']).astype(float)) 
    spmanscws_e_energy= computeSpearman(np.array(results['scws_ejacob_energy']).astype(float)) 
    print mname
    print 'WSIM: ',np.array(spman_wsim)
    print 'WSIM (Probs): ',np.array(spman_wsim_probs)
    print 'WSIM (Energy): ',np.array(spman_wsim_energy)
    print 'SCWS (E): ',np.array(spmanscws_e)
    print 'SCWS (E) Probs: ',np.array(spmanscws_e_probs)
    print 'SCWS (E) Energy: ',np.array(spmanscws_e_energy)
    results_int = {}
    for k in results:
        results_int[k] = np.array(results[k])
    saveHDF5(fname,results_int)
    print 'Saved'
print 'Done'
import ipdb;ipdb.set_trace()
