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

result_fname = 'qvary.pkl'
if not os.path.exists(result_fname):
    print 'Creating ',result_fname
    result  = {}
    for f in glob.glob(DIR+'/chkpt-rcv2_miao-*/*evaluate.h5'):
        params  = readPickle(getConfigFile(f.replace('evaluate.h5','')))[0] 
        rfile   = f.replace('evaluate.h5','EP200-params.npz')
        pfile   = f.replace('evaluate.h5','config.pkl')
        params['validate_only']  = True
        if params['opt_type']=='finopt':
            name = 'pl-'+str(params['p_layers'])+'-qh-'+str(params['q_dim_hidden'])+'-ql-'+str(params['q_layers'])+'-M'+str(params['n_steps'])
        else:
            name = 'pl-'+str(params['p_layers'])+'-qh-'+str(params['q_dim_hidden'])+'-ql-'+str(params['q_layers'])+'-M1'
        print 'Evaluating model: ',name
        print name,pfile, rfile
        bestModel     = Model(params, paramFile = pfile, reloadFile = rfile, additional_attrs = additional_attrs)
        train_results = Evaluate.evaluateBound(bestModel, dataset['train'], batch_size = params['batch_size'])
        test_results  = Evaluate.evaluateBound(bestModel, dataset['test'],  batch_size = params['batch_size'])
        train_perp_f  = train_results['perp_f'] 
        valid_perp_f  = test_results['perp_f'] 
        result[name]  = (train_perp_f, valid_perp_f)
    savePickle([result],result_fname)
    print 'Saved',result_fname

result = readPickle(result_fname)[0]
for pl in ['2']:
    for ql in ['1','2','3']: 
        for qh in ['100','400']:
            for M in ['M1','M100']:
                name = 'pl-'+pl+'-qh-'+qh+'-ql-'+ql+'-'+M
                print name, result[name]
        print '\n'
