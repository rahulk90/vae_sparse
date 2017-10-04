import os,time,sys
sys.path.append('../')
import numpy as np
from datasets.load import loadDataset
from optvaedatasets.load import loadDataset as loadDataset_OVAE
from optvaeutils.parse_args import params 
from utils.misc import removeIfExists,createIfAbsent,mapPrint,saveHDF5,displayTime,getLowestError
from sklearn.feature_extraction.text import TfidfTransformer

dataset = params['dataset']
params['savedir']+='-'+dataset+'-'+params['opt_type']
createIfAbsent(params['savedir'])
if 'mnist' in dataset:
    dataset = loadDataset(dataset)
    if 'binarized' not in dataset:
        dataset['train'] = (dataset['train']>0.5)*1.
        dataset['valid'] = (dataset['valid']>0.5)*1.
        dataset['test']  =  (dataset['test']>0.5)*1.
else:
    dataset = loadDataset_OVAE(dataset)
#Store dataset parameters into params 
for k in ['dim_observations','data_type']:
    params[k] = dataset[k]
if params['data_type']=='bow':
    params['max_word_count'] =dataset['train'].max()
mapPrint('Options: ',params)
#Setup VAE Model (or reload from existing savefile)
start_time = time.time()
from optvaemodels.vae import VAE as Model
import optvaemodels.vae_learn as Learn
import optvaemodels.vae_evaluate as Evaluate
import optvaemodels.evaluate_vecs as EVECS

additional_attrs = {}
if params['data_type']=='bow':
    additional_attrs        = {}
    tfidf                   = TfidfTransformer(norm=None) 
    tfidf.fit(dataset['train'])
    #Get normalized idf vectors
    additional_attrs['idf'] = tfidf.idf_ 

displayTime('import vae',start_time, time.time())
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
    model = Model(params, paramFile = pfile, reloadFile = reloadFile, additional_attrs = additional_attrs)
else:
    pfile= params['savedir']+'/'+params['unique_id']+'-config.pkl'
    print 'Training model from scratch. Parameters in: ',pfile
    model = Model(params, paramFile = pfile, additional_attrs = additional_attrs)
displayTime('Building vae',start_time, time.time())

savef      = os.path.join(params['savedir'],params['unique_id']) 
start_time = time.time()
trainData  = dataset['train'];validData = dataset['valid']
savedata   = Learn.learn( model, 
                                dataset     = trainData,
                                epoch_start = 0 , 
                                epoch_end   = params['epochs'], 
                                batch_size  = params['batch_size'],
                                savefreq    = params['savefreq'],
                                savefile    = savef,
                                dataset_eval= validData
                                )

displayTime('Running Model',start_time, time.time())
saveHDF5(savef+'-final.h5',savedata)

train_map_init_final = model.getInitFinal(trainData)
saveHDF5(savef+'-init_final_train.h5',train_map_init_final)

eval_map_init_final  = model.getInitFinal(validData)
saveHDF5(savef+'-init_final_eval.h5',eval_map_init_final)
print 'done'

"""
test_results = Evaluate.evaluateBound(model, dataset['test'], batch_size = params['batch_size'])
if model.params['data_type']=='bow':
    savedata['test_perp_0'] = test_results['perp_0'] 
    savedata['test_perp_f'] =test_results['perp_f'] 
    print 'Test Bound: ',savedata['test_perp_f']
    kname = 'valid_perp_f'
else:
    savedata['test_bound_0'] = test_results['elbo_0'] 
    savedata['test_bound_f'] =test_results['elbo_f'] 
    print 'Test Bound: ',savedata['test_bound_f']
    kname = 'valid_bound_f'


# Work w/ the best model thus far
epochMin, valMin, idxMin = getLowestError(savedata[kname])
reloadFile               = pfile.replace('-config.pkl','')+'-EP'+str(int(epochMin))+'-params.npz'
print 'Loading from : ',reloadFile
params['validate_only']  = True
bestModel                = Model(params, paramFile = pfile, reloadFile = reloadFile, additional_attrs = additional_attrs)

evaluate     = {}
test_results = Evaluate.evaluateBound(bestModel, dataset['test'], batch_size = params['batch_size'])
for k in test_results:
    evaluate[k+'_eb'] = test_results[k]

#Additions for synthetic datasets
if 'synthetic' in params['dataset']:
    evaluate['jacob_logprobs'] = EVECS.expectedJacobian(bestModel, nsamples = 200)
    evaluate['jacob_probs']    = EVECS.expectedJacobianProbs(bestModel, nsamples = 200)
    evaluate['jacob_energy']   = EVECS.expectedJacobianEnergy(bestModel, nsamples = 200)

saveHDF5(savef+'-evaluate.h5',evaluate)
if np.any([k in params['dataset'] for k in ['synthetic','newsgroups']]):
    pass
"""
