from utils.misc import loadHDF5, getConfigFile, readPickle, savePickle
import os,time,sys,glob
sys.path.append('../')
import numpy as np
from optvaedatasets.load import loadDataset 
from optvaemodels.vae import VAE as Model
import optvaemodels.vae_evaluate as Evaluate
from sklearn.feature_extraction.text import TfidfTransformer
DIR     = './'
dataset= loadDataset('rcv2')

additional_attrs        = {}
tfidf                   = TfidfTransformer(norm=None) 
tfidf.fit(dataset['train'])
additional_attrs['idf'] = tfidf.idf_ 

names = {}
<INSERT HERE>

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
#import ipdb;ipdb.set_trace()
