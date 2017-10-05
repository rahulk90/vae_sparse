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
from   optvaemodels.vae import VAE
import optvaemodels.vae_learn as VAE_learn
import optvaemodels.vae_evaluate as VAE_evaluate
import optvaemodels.vae_learn as VAE_learn
from   optvaedatasets.load import loadDataset 

models, epochval        = OrderedDict(), OrderedDict()
models['wikicorp-pl-2-finopt']   = './chkpt-wikicorp-finopt/VAE_lr-8_0e-04-ph-400-qh-400-ds-100-pl-2-ql-2-nl-relu-bs-500-ep-52-plr-1_0e-02-ar-0-otype-finopt-ns-100-etype-mlp-ll-mult-itype-tfidfl20_01_-uid'
epochval['wikicorp-pl-2-finopt'] = '50'

MODELS_TO_USE = models.keys()
print 'Evaluating on: ',MODELS_TO_USE

SAVEDIR = './evaluateIF_params/'
createIfAbsent(SAVEDIR)

DNAME = ''
dataset_wiki = loadDataset('wikicorp') 
additional_attrs_wiki = {}
def getTF(dataset):
    tfidf = TfidfTransformer(norm=None)
    tfidf.fit(dataset['train'])
    return tfidf.idf_
additional_attrs_wiki['idf'] = getTF(dataset_wiki) 

for mname in MODELS_TO_USE:
    if 'wikicorp' not in mname:
        continue
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
    savef     = SAVEDIR+mname
    trainData  = dataset['train'];validData = dataset['valid']
    train_map_init_final = VAE_evaluate.getInitFinal(vae, trainData)
    saveHDF5(savef+'-if_train.h5',train_map_init_final)
    eval_map_init_final  = VAE_evaluate.getInitFinal(vae, validData)
    saveHDF5(savef+'-if_eval.h5', eval_map_init_final)
    print 'Saved: ',mname
print 'Done'
