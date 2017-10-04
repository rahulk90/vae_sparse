import numpy as np
import time
from theano import config
import optvaemodels.vae_evaluate as VAE_evaluate
from utils.misc import saveHDF5,savePickle

def numActive(vec):
    assert vec.ndim==1 ,'Expecting single dimensional vector'
    return np.sum((vec>0.01)*1.).astype(int)

def arrToReadableString(nparr, divideBy=1):
    assert nparr.ndim==1,'expecting 1 dimensional array'
    if len(nparr)>20:
        small_list = nparr[:5].tolist()+nparr[-5:].tolist()
    else:
        small_list = nparr.tolist()
    ss =''
    for k in small_list:
        ss += ('%.1f'%(k/float(divideBy)))+', '
    return ss

def _optNone(vae, bnum, Nbatch, X, retVals = {}, calcELBOfinal = False, update_opt= None):
    """
    vae:    VAE object
    bnum:   Batch number
    Nbatch: # elements in batch
    X:      Current batch
    retVals:Map to return values from 
    calcELBOfinal: Whether or not calculate final evidence lower bound
    """
    start_time = time.time()
    elbo_0, pnorm, gnorm, optnorm, anneal, lr = vae.train(X=X)
    if np.isnan(elbo_0):
        print 'NAN warning'
        import ipdb;ipdb.set_trace()
    elbo_f = np.nan
    n_steps    = 0
    gmu, glcov, diff_elbo = np.nan, np.nan, np.nan
    if calcELBOfinal:
        _, elbo_f, n_steps, gmu, glcov,  diff_elbo = vae.final_elbo(X=X)
    freq  = 100
    time_taken = time.time()-start_time
    if bnum%freq==0:
        vae._p(('--Batch: %d, ELBO[0]: %.4f, ELBO[f]: %.4f, Anneal : %.4f, lr : %.4f, Time(s): %.2f--')%
                (bnum, elbo_0/Nbatch, elbo_f/Nbatch, anneal, lr, time_taken))
        vae._p(('--||w||: %.4f, ||dw|| : %.4f, ||w_opt||: %.4f --')%
                (pnorm, gnorm, optnorm))
        vae._p(('--n_steps: %d, g_mu_f:%.3f, g_lcov_f:%.3f,diff_elbo:%.3f, diff_ent:%.3f--')%
                (n_steps,gmu/Nbatch,glcov/Nbatch, diff_elbo/Nbatch, 0./Nbatch))
    retVals['elbo_0']     = elbo_0
    retVals['elbo_f']     = elbo_f
    retVals['gmu']        = gmu
    retVals['glcov']      = glcov
    retVals['diff_elbo']  = diff_elbo
    retVals['time_taken'] = time_taken
    return retVals

def _optFinopt(vae, bnum, Nbatch, X,  retVals = {}, calcELBOfinal=True, update_opt = None):
    start_time = time.time()
    D_b     = vae.update_q(X=X)
    results = vae.update_p(X=X)
    elbo_0, elbo_f, anneal, pnorm, gnorm, optnorm   = results[0], results[1], results[2], results[3], results[4],results[5]
    n_steps, gmu,glcov, diff_elbo = results[6], results[7], results[8], results[9]
    if np.isnan(elbo_0):
        print 'NAN warning'
        import ipdb;ipdb.set_trace()
    freq = 100
    time_taken = time.time()-start_time
    if bnum%freq==0:
        vae._p(('--Batch: %d, Bound (init): %.4f, Bound (final): %.4f, Time (sec): %.2f')%
                (bnum, elbo_0/Nbatch, elbo_f/Nbatch, time_taken))
        vae._p(('--||w||: %.4f, ||dw|| : %.4f, ||w_opt||: %.4f, anneal : %.4f----')%
                (pnorm, gnorm, optnorm, anneal))
        vae._p(('--D_b: %.4f, n_steps: %d, g_mu_f:%.3f, g_lcov_f:%.3f,diff_elbo:%.3f, diff_ent:%.3f--')%
                (D_b/Nbatch,n_steps,gmu/Nbatch,glcov/Nbatch,diff_elbo/Nbatch, 0./Nbatch))
    retVals['elbo_0'] = elbo_0
    retVals['elbo_f'] = elbo_f
    retVals['gmu']    = gmu
    retVals['glcov']  = glcov
    retVals['diff_elbo'] = diff_elbo
    retVals['time_taken']= time_taken
    return retVals

def _optMixed(vae, bnum, Nbatch, X, retVals = {}, calcELBOfinal = True, update_opt= False):
    start_time = time.time()
    if update_opt:
        D_b        = vae.update_q(X=X)
        results    = vae.update_p(X=X)
        elbo_0, elbo_f, anneal, pnorm, gnorm, optnorm   = results[0], results[1], results[2], results[3], results[4],results[5]
        n_steps, gmu,glcov, diff_elbo = results[6], results[7], results[8], results[9]
    else:
        elbo_0, pnorm, gnorm, optnorm, anneal, lr = vae.train(X=X)
        if np.isnan(elbo_0):
            print 'NAN warning'
            import ipdb;ipdb.set_trace()
        elbo_f = np.nan
        n_steps    = 0
        gmu, glcov, diff_elbo = np.nan, np.nan, np.nan
        _, elbo_f, n_steps, gmu, glcov,  diff_elbo = vae.final_elbo(X=X)
    if np.isnan(elbo_0):
        print 'NAN warning'
        import ipdb;ipdb.set_trace()
    time_taken = time.time()-start_time
    freq = 100
    if bnum%freq==0:
        vae._p(('--Batch: %d, Bound (init): %.4f, Bound (final): %.4f, Time (sec): %.2f')%
                (bnum, elbo_0/Nbatch, elbo_f/Nbatch, time_taken))
        vae._p(('--||w||: %.4f, ||dw|| : %.4f, ||w_opt||: %.4f, anneal : %.4f----')%
                (pnorm, gnorm, optnorm, anneal))
        vae._p(('--n_steps: %d, g_mu_f:%.3f, g_lcov_f:%.3f,diff_elbo:%.3f, diff_ent:%.3f--')%
                (n_steps,gmu/Nbatch,glcov/Nbatch,diff_elbo/Nbatch, 0./Nbatch))
    retVals['elbo_0'] = elbo_0
    retVals['elbo_f'] = elbo_f
    retVals['gmu']    = gmu
    retVals['glcov']  = glcov
    retVals['diff_elbo'] = diff_elbo
    retVals['time_taken']= time_taken
    return retVals

def learn(vae, dataset=None, epoch_start=0, epoch_end=1000, batch_size=200, shuffle=False, 
        savefile = None, savefreq = None, dataset_eval=None):
    assert dataset is not None,'Expecting 2D dataset matrix'
    assert np.prod(dataset.shape[1:])==vae.params['dim_observations'],'dim observations incorrect'
    N = dataset.shape[0]
    idxlist = range(N)
    trainbound_0, trainbound_f = [], []
    trainperp_0, trainperp_f, validperp_0, validperp_f, svallist = [],[],[],[],[]
    gmulist, glcovlist, diff_elbolist, batchtimelist = [],[],[],[]
    gdifflists = {}
    for name in vae.p_names:
        gdifflists[name] = []

    learnBatch = None
    print 'OPT TYPE: ',vae.params['opt_type']
    if vae.params['opt_type'] in ['none','q_only', 'q_only_random']:
        learnBatch = _optNone
    elif vae.params['opt_type'] in ['finopt']:
        learnBatch = _optFinopt
    elif vae.params['opt_type'] in ['finopt_none','none_finopt']:
        learnBatch = _optMixed
    else:
        raise ValueError('Invalid optimization type: '+str(vae.params['opt_type']))
    for epoch in range(epoch_start, epoch_end+1):
        np.random.shuffle(idxlist)
        start_time = time.time()
        bd_0, bd_f, gmu, glcov,diff_elbo, time_taken = 0, 0, 0, 0, 0, 0
        """ Evaluate more frequently in the initial few epochs """
        if epoch > 10:
            sfreq = savefreq 
            tfreq = savefreq
        else:
            sfreq = 3 
            tfreq = 3
        for bnum,st_idx in enumerate(range(0,N,batch_size)):
            end_idx = min(st_idx+batch_size, N)
            X       = dataset[idxlist[st_idx:end_idx]]
            if X.__class__.__name__=='csr_matrix' or X.__class__.__name__=='csc_matrix':
                X   = X.toarray()
            X       = X.astype(config.floatX)
            Nbatch  = X.shape[0]
            update_opt= None
            if vae.params['opt_type'] in ['finopt_none','none_finopt']:
                if vae.params['opt_type'] == 'finopt_none': #start w/ optimizing var. params, then stop
                    if epoch<10:#(epoch_end/2.):
                        update_opt = True
                    else:
                        update_opt = False
                else: #'none_finopt' - start w/out optimizing var. params, then optimize them
                    if epoch<10:#(epoch_end/2.):
                        update_opt = False
                    else:
                        update_opt = True
            retVal  = learnBatch(vae, bnum, Nbatch, X, calcELBOfinal=(epoch%tfreq==0),  update_opt=update_opt) 
            bd_0    += retVal['elbo_0']
            bd_f    += retVal['elbo_f']
            gmu     += retVal['gmu']
            glcov   += retVal['glcov']
            diff_elbo  += retVal['diff_elbo']
            time_taken += retVal['time_taken']
            break
        bd_0   /= float(N)
        bd_f   /= float(N)
        gmu    /= float(N)
        glcov  /= float(N)
        diff_elbo  /= float(N)
        time_taken /= float(bnum) 
        batchtimelist.append((epoch,time_taken))
        trainbound_0.append((epoch,bd_0))
        trainbound_f.append((epoch,bd_f))
        gmulist.append((epoch, gmu))
        glcovlist.append((epoch, glcov))
        diff_elbolist.append((epoch, diff_elbo))
        end_time   = time.time()
        print '\n'
        vae._p(('Ep(%d) ELBO[0]: %.4f, ELBO[f]: %.4f, gmu: %.4f, glcov: %.4f, [%.3f, %.3f] [%.4f seconds]')%(epoch, bd_0, bd_f, gmu, glcov, diff_elbo, 0., (end_time-start_time)))
        if savefreq is not None and epoch%sfreq==0:
            vae._p(('Saving at epoch %d'%epoch))
            vae._saveModel(fname=savefile+'-EP'+str(epoch))
            if dataset_eval is not None:
                if vae.params['data_type']=='bow':
                    k0='perp_0'
                    kf='perp_f'
                else:
                    k0='elbo_0'
                    kf='elbo_f'
                eval_retVal = VAE_evaluate.evaluateBound(vae, dataset_eval, batch_size=batch_size)
                validperp_0.append((epoch,eval_retVal[k0]))
                validperp_f.append((epoch,eval_retVal[kf]))
                train_retVal = VAE_evaluate.evaluateBound(vae, dataset, batch_size=batch_size)
                trainperp_0.append((epoch,train_retVal[k0]))
                trainperp_f.append((epoch,train_retVal[kf]))
                vae._p(('\nEp (%d): Valid[0]: %.4f, Valid[f]: %.4f')%(epoch, eval_retVal[k0], eval_retVal[kf]))
                vae._p(('\nEp (%d): Train[0]: %.4f, Train[f]: %.4f')%(epoch, train_retVal[k0], train_retVal[kf]))
            intermediate = {}
            intermediate['train_bound_0'] = np.array(trainbound_0)
            intermediate['train_bound_f'] = np.array(trainbound_f)
            if vae.params['data_type']=='bow':
                k0 = 'valid_perp_0'
                kf = 'valid_perp_f'
            else:
                k0 = 'valid_bound_0'
                kf = 'valid_bound_f'
            intermediate[k0] = np.array(validperp_0)
            intermediate[kf] = np.array(validperp_f)
            intermediate[k0.replace('valid','train')] = np.array(trainperp_0)
            intermediate[kf.replace('valid','train')] = np.array(trainperp_f)
            intermediate['batch_time'] = np.array(batchtimelist)
            intermediate['gmu']        = np.array(gmulist)
            intermediate['glcov']      = np.array(glcovlist)
            intermediate['diff_elbo']  = np.array(diff_elbolist)
            jacob      = vae.jacobian_logprobs(np.zeros((vae.params['dim_stochastic'],)).astype('float32'))
            _,svals,_  = np.linalg.svd(jacob)
            epres      = np.array([epoch] + svals.ravel().tolist())
            svallist.append(epres)
            intermediate['svals']      = np.array(svallist)
            saveHDF5(savefile+'-EP'+str(epoch)+'-stats.h5', intermediate)
    ret_map={}
    ret_map['train_bound_0'] = np.array(trainbound_0)
    ret_map['train_bound_f'] = np.array(trainbound_f)
    ret_map['batch_time'] = np.array(batchtimelist)
    if vae.params['data_type']=='bow':
        k0 = 'valid_perp_0'
        kf = 'valid_perp_f'
    else:
        k0 = 'valid_bound_0'
        kf = 'valid_bound_f'
    ret_map[k0] = np.array(validperp_0)
    ret_map[kf] = np.array(validperp_f)
    ret_map[k0.replace('valid','train')] = np.array(trainperp_0)
    ret_map[kf.replace('valid','train')] = np.array(trainperp_f)
    ret_map['gmu']           = np.array(gmulist)
    ret_map['glcov']         = np.array(glcovlist)
    ret_map['diff_elbo']     = np.array(diff_elbolist)
    ret_map['svals']         = np.array(svallist)
    return ret_map
