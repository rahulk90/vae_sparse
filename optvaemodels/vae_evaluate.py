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

def getInitFinal(vae, dataset, batch_size):
    N = dataset.shape[0]
    init_mulist, final_mulist = [], []
    init_logcovlist, final_logcovlist = [], []
    for bnum,st_idx in enumerate(range(0,N,batch_size)):
        if bnum%1000==0:
            print bnum,
        end_idx = min(st_idx+batch_size, N)
        X       = dataset[st_idx:end_idx]
        if X.__class__.__name__=='csr_matrix' or X.__class__.__name__=='csc_matrix':
            X   = X.toarray()
        X       = X.astype(config.floatX)
        mu_0, logcov_0, mu_f, logcov_f = vae.init_final_params(X=X)
        init_mulist.append(mu_0)
        init_logcovlist.append(logcov_0)
        final_mulist.append(mu_f)
        final_logcovlist.append(logcov_f)
    print '... done init_final'
    retVals = {}
    retVals['mu_0']      = np.concatenate(init_mulist, axis=0)
    retVals['logcov_0']  = np.concatenate(init_logcovlist, axis=0)
    retVals['mu_f']      = np.concatenate(final_mulist, axis=0)
    retVals['logcov_f']  = np.concatenate(final_logcovlist, axis=0)
    return retVals

def evaluateBound(vae, dataset, batch_size):
    """ Evaluate bound on dataset  """
    N = dataset.shape[0]
    bd_0,bd_f = 0,0
    perp0,perpf = 0,0
    diff_elbo = 0
    for bnum,st_idx in enumerate(range(0,N,batch_size)):
        if bnum%1000==0:
            print bnum,
        end_idx = min(st_idx+batch_size, N)
        X       = dataset[st_idx:end_idx]
        if X.__class__.__name__=='csr_matrix' or X.__class__.__name__=='csc_matrix':
            X   = X.toarray()
        X       = X.astype(config.floatX)
        if vae.params['data_type']=='bow':
            perp_0, perp_f, n_steps, d_elbo = vae.evaluatePerp(X=X)
            bd_0  += perp_0
            bd_f  += perp_f
        else:
            elbo_0, elbo_f, n_steps, d_elbo = vae.evaluate(X=X)
            bd_0  += elbo_0
            bd_f  += elbo_f
        diff_elbo+= d_elbo
    print '.... done evaluation'
    bd_0 /= float(N)
    bd_f /= float(N)
    diff_elbo /= float(N)
    retVals = {}
    if vae.params['data_type']=='bow':
        retVals['perp_0']= np.exp(bd_0)
        retVals['perp_f']= np.exp(bd_f)
    else:
        retVals['elbo_0']= bd_0
        retVals['elbo_f']= bd_f
    retVals['diff_elbo'] = diff_elbo
    return retVals

