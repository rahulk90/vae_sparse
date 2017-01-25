import os,time,sys
sys.path.append('../')
import numpy as np
from datasets.load import loadDataset
from optvaedatasets.load import loadDataset as loadDataset_OVAE
from optvaeutils.parse_args_dan import params 
from utils.misc import removeIfExists,createIfAbsent,mapPrint,saveHDF5,displayTime,getLowestError
from sklearn.feature_extraction.text import TfidfTransformer

dataset = params['dataset']
params['savedir']+='-'+dataset+'-'+params['opt_type']
createIfAbsent(params['savedir'])
dataset = loadDataset_OVAE(dataset)

#Load datasets that you're using to 
dataset_wvecs = params['dataset_wvecs']
dataset_wvecs = loadDataset_OVAE(dataset_wvecs)

#Load Jacobian vectors
jacobian      = np.load(params['jacobian_location'])

assert jacobian.shape[0] == len(dataset_wvecs['vocabulary']),'shapes dont match up'
params['dim_observation'] = jacobian.shape[1]

#Store dataset parameters into params 
mapPrint('Options: ',params)

#Setup VAE DAN (or reload from existing savefile)
start_time = time.time()
from   optvaemodels.dan import DAN 
import optvaemodels.dan import Learn
import optvaemodels.dan import Evaluate

displayTime('import DAN',start_time, time.time())
vae    = None
#Remove from params
start_time = time.time()
removeIfExists('./NOSUCHFILE')
reloadFile = params.pop('reloadFile')
if os.path.exists(reloadFile):
    pfile=params.pop('paramFile')
    assert os.path.exists(pfile),pfile+' not found. Need paramfile'
    print 'Reloading trained model from : ',reloadFile
    print 'Assuming ',pfile,' corresponds to model'
    model = DAN(params, paramFile = pfile, reloadFile = reloadFile, additional_attrs = additional_attrs)
else:
    pfile= params['savedir']+'/'+params['unique_id']+'-config.pkl'
    print 'Training model from scratch. Parameters in: ',pfile
    model = DAN(params, paramFile = pfile, additional_attrs = additional_attrs)
displayTime('Building vae',start_time, time.time())

savef      = os.path.join(params['savedir'],params['unique_id']) 
start_time = time.time()
trainData  = dataset['train'];validData     = dataset['valid']
savedata   = Learn.learn( model, 
                                dataset     = trainData,
                                epoch_start = 0 , 
                                epoch_end   = params['epochs'], 
                                batch_size  = params['batch_size'],
                                savefreq    = params['savefreq'],
                                savefile    = savef,
                                dataset_eval= validData
                                )

displayTime('Running DAN',start_time, time.time())
savedata['test_klmat']  = test_results['klmat'] 
saveHDF5(savef+'-final.h5',savedata)

# Work w/ the best model thus far
epochMin, valMin, idxMin = getLowestError(savedata['valid_acc'])
reloadFile               = pfile.replace('-config.pkl','')+'-EP'+str(int(epochMin))+'-params.npz'
print 'Loading from : ',reloadFile
params['validate_only']  = True
bestDAN                  = DAN(params, paramFile = pfile, reloadFile = reloadFile, additional_attrs = additional_attrs)

test_results = Evaluate.evaluateAccuracy(bestDAN, dataset['test'], batch_size = params['batch_size'])
saveHDF5(savef+'-evaluate.h5',test_results)
import ipdb; ipdb.set_trace()
