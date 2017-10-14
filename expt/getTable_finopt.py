"""
Read saved results to obtain tables
"""
import glob
from utils.misc import loadHDF5,getConfigFile, readPickle
DIR     = './'
datasets= ['20newsgroups','rcv2']
result_best  = {}
result_last  = {}
for dataset in datasets:
    print 'Dataset: ',dataset
    for f in glob.glob(DIR+'/chkpt-'+dataset+'-*/*evaluate.h5'):
        if 'mnist' in f or 'qvary' in f:
            continue
        dataset = f.split('chkpt-')[1].split('-')[0]
        opt_type= f.split('chkpt-')[1].split('-')[1].split('/')[0]
        params  = readPickle(getConfigFile(f.replace('evaluate.h5','')))[0] 
        dset    = loadHDF5(f)
        if params['opt_type']=='finopt':
            name = str(params['p_layers'])+'-M'+str(params['n_steps'])+'-'+params['input_type']
        else:
            name = str(params['p_layers'])+'-M1-'+params['input_type']
        result_best[params['dataset']+'-'+name] = (dset['perp_0_best'],dset['perp_f_best'])
        result_last[params['dataset']+'-'+name] = (dset['test_perp_0'],dset['test_perp_f'])
        print name, (dset['perp_0_best'],dset['perp_f_best'])
for dataset in datasets:
    for itype in ['normalize','tfidf']:
        for layer in ['0','2']: 
            for M in ['M1','M100']:
               name = dataset+'-'+layer+'-'+M+'-'+itype
               print name, result_best[name], result_last[name]
    print '\n'
