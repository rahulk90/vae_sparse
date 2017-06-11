#Evaluate the Jacobian for wikipedia/rcv2 to analyze its log-singular values
import sys,os
from collections import OrderedDict
sys.path.append('../')
import numpy as np
from theano import config
import itertools,time
from utils.misc import readPickle,savePickle
from utils.misc import getConfigFile, loadHDF5, saveHDF5, createIfAbsent
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

    
models, epochval        = OrderedDict(), OrderedDict()

models['wikicorp-pl-2-none']     = './results_wiki/chkpt-wikicorp-none/VAE_lr-8_0e-04-ph-400-qh-400-ds-100-pl-2-ql-2-nl-relu-bs-500-ep-50-plr-1_0e-02-ar-0-otype-none-ns-200-om-adam-etype-mlp-ll-mult-itype-tfidfl20_01_-uid'
epochval['wikicorp-pl-2-none']   = '50'

models['wikicorp-pl-2-finopt']   = './results_wiki/chkpt-wikicorp-finopt/VAE_lr-8_0e-04-ph-400-qh-400-ds-100-pl-2-ql-2-nl-relu-bs-500-ep-50-plr-1_0e-02-ar-0-otype-finopt-ns-200-om-adam-etype-mlp-ll-mult-itype-tfidfl20_01_-uid'
epochval['wikicorp-pl-2-finopt'] = '50'

models['wikicorp-pl-0-none']   = './results_wiki/chkpt-wikicorp-none/VAE_lr-8_0e-04-ph-400-qh-400-ds-100-pl-0-ql-2-nl-relu-bs-500-ep-50-plr-1_0e-02-ar-0-otype-none-ns-200-om-adam-etype-mlp-ll-mult-itype-tfidfl20_01_-uid'
epochval['wikicorp-pl-0-none'] = '50'

models['wikicorp-pl-0-finopt']   = './results_wiki/chkpt-wikicorp-finopt/VAE_lr-8_0e-04-ph-400-qh-400-ds-100-pl-0-ql-2-nl-relu-bs-500-ep-50-plr-1_0e-02-ar-0-otype-finopt-ns-200-om-adam-etype-mlp-ll-mult-itype-tfidfl20_01_-uid'
epochval['wikicorp-pl-0-finopt'] = '50'

models['rcv2-pl-2-none']     = './results_dec19/chkpt-rcv2_miao-none/VAE_lr-8_0e-04-ph-400-ds-100-pl-2-ql-2-nl-relu-bs-500-ep-200-plr-1_0e-02-ar-0-otype-none-ns-100-om-adam-etype-mlp-ll-mult-itype-tfidfl20_01_-uid'
epochval['rcv2-pl-2-none']   = '200'

models['rcv2-pl-2-finopt']   = './results_dec19/chkpt-rcv2_miao-finopt/VAE_lr-8_0e-04-ph-400-ds-100-pl-2-ql-2-nl-relu-bs-500-ep-200-plr-1_0e-02-ar-0-otype-finopt-ns-100-om-adam-etype-mlp-ll-mult-itype-tfidfl20_01_-uid'
epochval['rcv2-pl-2-finopt'] = '200'

models['rcv2-pl-0-none']   = './results_dec19/chkpt-rcv2_miao-none/VAE_lr-8_0e-04-ph-400-ds-100-pl-0-ql-2-nl-relu-bs-500-ep-200-plr-1_0e-02-ar-0-otype-none-ns-100-om-adam-etype-mlp-ll-mult-itype-tfidfl20_01_-uid'
epochval['rcv2-pl-0-none'] = '200'

models['rcv2-pl-0-finopt']   = './results_dec19/chkpt-rcv2_miao-finopt/VAE_lr-8_0e-04-ph-400-ds-100-pl-0-ql-2-nl-relu-bs-500-ep-200-plr-1_0e-02-ar-0-otype-finopt-ns-100-om-adam-etype-mlp-ll-mult-itype-tfidfl20_01_-uid'
epochval['rcv2-pl-0-finopt'] = '200'

MODELS_TO_USE = models.keys()
print 'Evaluating on: ',MODELS_TO_USE

SAVEDIR = './evaluateJac/'
createIfAbsent(SAVEDIR)


DNAME = ''
dataset_wiki = loadDataset('wikicorp') 
dataset_rcv2 = loadDataset('rcv2_miao') 
additional_attrs_wiki = {}
additional_attrs_rcv2 = {} 
def getTF(dataset):
    tfidf = TfidfTransformer(norm=None)
    tfidf.fit(dataset['train'])
    return tfidf.idf_
additional_attrs_wiki['idf'] = getTF(dataset_wiki) 
additional_attrs_rcv2['idf'] = getTF(dataset_rcv2) 

for mname in MODELS_TO_USE:

    print 'Model: ',mname
    pfile = models[mname].split('uid')[0]+'uid-config.pkl'
    params= readPickle(pfile)[0]
    suffix= '-EP'+str(epochval[mname])+'-params.npz'
    rfile = models[mname]+suffix
    assert os.path.exists(rfile),'not found'
    params['EVALUATION'] = True
    if 'wikicorp' in mname: 
        vae   = VAE(params, paramFile=pfile, reloadFile=rfile, additional_attrs = additional_attrs_wiki)
    else:
        vae   = VAE(params, paramFile=pfile, reloadFile=rfile, additional_attrs = additional_attrs_rcv2)
    jacob= expectedJacobian(vae, nsamples=1)
    _,s,_  = np.linalg.svd(jacob)
    fname = SAVEDIR+mname+'-jacob.h5'
    saveHDF5(fname,{'jacob':jacob,'svals':s})
    print 'Saved: ',mname
print 'Done'
