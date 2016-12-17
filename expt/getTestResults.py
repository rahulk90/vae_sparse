import glob
from utils.misc import loadHDF5,getConfigFile, readPickle
#DIR = './results-sept6/'
DIR = './'

dataset = '20newsgroups'
#dataset = 'rcv2'

for f in glob.glob(DIR+'/chkpt-'+dataset+'_*/*evaluate.h5'):
    if 'mnist' in f:
        continue
    dataset = f.split('chkpt-')[1].split('-')[0]
    opt_type= f.split('chkpt-')[1].split('-')[1].split('/')[0]
    params  = readPickle(getConfigFile(f.replace('evaluate.h5','')))[0] 
    dset    = loadHDF5(f)
    if params['opt_type']=='finopt':
        name = str(params['p_layers'])+'-M'+str(params['n_steps'])+'-'+params['input_type']
    else:
        name = str(params['p_layers'])+'-M1-'+params['input_type']
    print params['dataset'],name, dset['perp_0_eb'],dset['perp_f_eb']
