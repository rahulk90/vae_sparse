from utils.misc import loadHDF5, getConfigFile, readPickle, savePickle
import os,time,sys,glob
sys.path.append('../')
import numpy as np
from optvaedatasets.load import loadDataset 
from optvaemodels.vae import VAE as Model
import optvaemodels.vae_evaluate as Evaluate
from sklearn.feature_extraction.text import TfidfTransformer
DIR     = './'
dataset= loadDataset('rcv2_miao')

additional_attrs        = {}
tfidf                   = TfidfTransformer(norm=None) 
tfidf.fit(dataset['train'])
additional_attrs['idf'] = tfidf.idf_ 

names = {}
for ql in ['1','3']:
    for qh in ['100','400']:
        names['ql-'+ql+'-qh-'+qh] = {}

from setupExperiments import chkpt

names['ql-1-qh-100']['pfile']    = chkpt['1-100-p_fixed']+'-config.pkl'
names['ql-1-qh-100']['rfile']    = './chkpt-rcv2_miao-q_only_random/VAE_lr-8_0e-04-ph-400-qh-100-ds-100-pl-2-ql-1-nl-relu-bs-500-ep-100-plr-1_0e-02-ar-0-otype-q_only_random-ns-100-om-adam-etype-mlp-ll-mult-itype-tfidf-idrop-1_0e-04l20_01_-uid'+'-EP100-params.npz' 

names['ql-1-qh-400']['pfile']    = chkpt['1-400-p_fixed']+'-config.pkl' 
names['ql-1-qh-400']['rfile']    = './chkpt-rcv2_miao-q_only_random/VAE_lr-8_0e-04-ph-400-qh-400-ds-100-pl-2-ql-1-nl-relu-bs-500-ep-100-plr-1_0e-02-ar-0-otype-q_only_random-ns-100-om-adam-etype-mlp-ll-mult-itype-tfidf-idrop-1_0e-04l20_01_-uid'+'-EP100-params.npz'

names['ql-3-qh-100']['pfile']    = chkpt['3-100-p_fixed']+'-config.pkl'
names['ql-3-qh-100']['rfile']    = './chkpt-rcv2_miao-q_only_random/VAE_lr-8_0e-04-ph-400-qh-100-ds-100-pl-2-ql-3-nl-relu-bs-500-ep-100-plr-1_0e-02-ar-0-otype-q_only_random-ns-100-om-adam-etype-mlp-ll-mult-itype-tfidf-idrop-1_0e-04l20_01_-uid'+'-EP100-params.npz'

names['ql-3-qh-400']['pfile']    = chkpt['3-400-p_fixed']+'-config.pkl'
names['ql-3-qh-400']['rfile']    = './chkpt-rcv2_miao-q_only_random/VAE_lr-8_0e-04-ph-400-qh-400-ds-100-pl-2-ql-3-nl-relu-bs-500-ep-100-plr-1_0e-02-ar-0-otype-q_only_random-ns-100-om-adam-etype-mlp-ll-mult-itype-tfidf-idrop-1_0e-04l20_01_-uid'+'-EP100-params.npz' 

outfname          = __file__.replace('py','outf')
outf              = open(outfname,'w')
result            = {}
for name in names:
    print 'Running',name
    pfile         = names[name]['pfile']
    rfile         = names[name]['rfile']
    params        = readPickle(pfile)[0]
    params['validate_only']  = True
    bestModel     = Model(params, paramFile = pfile, reloadFile = rfile, additional_attrs = additional_attrs)
    train_results = Evaluate.evaluateBound(bestModel, dataset['train'], batch_size = params['batch_size'])
    train_perp_0  = train_results['perp_0']
    train_perp_f  = train_results['perp_f']
    test_results  = Evaluate.evaluateBound(bestModel, dataset['test'],  batch_size = params['batch_size'])
    valid_perp_0  = test_results['perp_0']
    valid_perp_f  = test_results['perp_f']
    result[name]  = (train_perp_0, train_perp_f, valid_perp_0, valid_perp_f)
    outf.write(name+' train: '+str(train_perp_0)+','+str(train_perp_f)+'; valid:'+str(valid_perp_0)+','+str(valid_perp_f)+' \n')
outf.close()
import ipdb;ipdb.set_trace()
