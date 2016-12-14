import numpy as np
from theano import config
import theano.tensor as T
from sklearn.decomposition import PCA
"""
External function to deal with evaluation of model
"""

def getPrior(vae, nsamples=100):
    """ Sample from Prior """
    z = np.random.randn(nsamples,vae.params['dim_stochastic'])
    return z

def sample(vae, nsamples=100):
    z   = getPrior(vae,nsamples)
    return vae.reconstruct(z.astype(config.floatX))

def infer(vae, data):
    """ Posterior Inference using recognition network """
    assert len(data.shape)==2,'Expecting 2D data matrix'
    assert data.shape[1]==vae.params['dim_observations'],'Wrong dimensions for observations'
    return vae.inference(X=data.astype(config.floatX))

def evaluateBound(vae, dataset, batch_size, retVals = {}):
    """ Evaluate bound on dataset  """
    N = dataset.shape[0]
    bd_0,bd_f = 0,0
    perp0,perpf = 0,0
    KLmat = 0
    diff_elbo, diff_ent = 0,0
    dbglist = None
    for bnum,st_idx in enumerate(range(0,N,batch_size)):
        end_idx = min(st_idx+batch_size, N)
        X       = dataset[st_idx:end_idx]
        if X.__class__.__name__=='csr_matrix' or X.__class__.__name__=='csc_matrix':
            X   = X.toarray()
        X       = X.astype(config.floatX)
        if vae.params['data_type']=='bow':
            perp_0, perp_f, n_steps, d_elbo,d_ent  = vae.evaluatePerp(X=X)
            bd_0  += perp_0
            bd_f  += perp_f
        else:
            elbo_0, elbo_f, n_steps,d_elbo, d_ent  = vae.evaluate(X=X)
            bd_0  += elbo_0
            bd_f  += elbo_f
        diff_elbo+= d_elbo
        diff_ent += d_elbo
    bd_0 /= float(N)
    bd_f /= float(N)
    diff_elbo /= float(N)
    diff_ent /= float(N)
    if vae.params['data_type']=='bow':
        retVals['perp_0']= np.exp(bd_0)
        retVals['perp_f']= np.exp(bd_f)
    else:
        retVals['elbo_0']= bd_0
        retVals['elbo_f']= bd_f
    retVals['diff_elbo'] = diff_elbo
    retVals['diff_ent']  = diff_ent
    retVals['klmat'] = np.array([0])
    return retVals

def meanSumExp(vae,mat,axis=1):
    """ Estimate log 1/S \sum_s exp[ log k ] in 
    a numerically stable manner where axis represents the sum
    """
    a = np.max(mat, axis=1, keepdims=True)
    return a + np.log(np.mean(np.exp(mat-a.repeat(mat.shape[1],1)),axis=1,keepdims=True))

def impSamplingNLL(vae, dataset, batch_size, S = 10):
    """
                                Importance sampling based log likelihood
    """
    N = dataset.shape[0]
    ll = 0
    for bnum,st_idx in enumerate(range(0,N,batch_size)):
        end_idx = min(st_idx+batch_size, N)
        X       = dataset[st_idx:end_idx].astype(config.floatX)
        batch_lllist = []
        for s in range(S):
            if vae.params['inference_model']=='single':
                batch_ll = vae.likelihood(X=X)
            else:
                assert False,'Should not be here'
            batch_lllist.append(batch_ll)
        ll  += vae.meanSumExp(np.concatenate(batch_lllist,axis=1), axis=1).sum()
    ll /= float(N)
    return -ll
