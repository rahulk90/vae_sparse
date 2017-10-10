import glob,os
import numpy as np
from utils.misc import loadHDF5,getConfigFile,readPickle

for f in glob.glob('./chkpt-*/*-EP50-stats.h5'):
    code = 'ds'+os.path.basename(f).split('-ql')[0].split('ds')[1]
    if 'finopt' in f:
        code = 'finopt-'+code
    else:
        code = 'none-'+code
    data = loadHDF5(f)
    params=readPickle(getConfigFile(f))[0]
    code = params['dataset']+'-'+code
    runtimes = [] 
    for edata in data['batch_time']:
        if int(edata[0])%params['savefreq']==0:
            continue
        else:
            runtimes.append(edata[1])
    print code, np.mean(runtimes)
