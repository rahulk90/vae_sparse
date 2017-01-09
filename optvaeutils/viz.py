import urllib2
import numpy as np

def getName(params):
    """ Return a name corresponding to the values in params"""
    assert 'opt_type' in params,'otype not found'
    assert 'anneal_rate' in params,'ar not found'
    assert 'replicate_K' in params,'repK not found'
    assert 'lr' in params,'lr not found'
    if params['opt_type']=='none':
        name = 'baseline'
        if params['replicate_K'] is not None:
            name+='_rK'+str(params['replicate_K'])
        if params['anneal_rate']>1:
            name+='_annealKL'
    elif params['opt_type']=='miao':
        name = 'miao'
    elif params['opt_type']=='miao_ascent':
        name = 'miao_ascent'
    elif params['opt_type'] in ['finopt']:
        name = 'fin'+str(params['n_steps'])
    elif params['opt_type'] in ['finopt_mult']:
        name = 'fin_mult'+str(float(params['p_updates']))+'_'+str(int(params['n_steps']/float(params['p_updates'])))
    else:
        assert False,'Invalid choice for otype:'+str(params['opt_type'])
    #if params['batch_normalize']:
    #    name += '_bn'
    return name

def processTrainValidBounds(train_bound, valid_bound, savefreq):
    """ savefreq: freq of saving validation bounds 
        train/valid_bound : 2D mat (epoch - value)
    """
    assert train_bound.ndim==2 and valid_bound.ndim==2,'Expecting 2D arrays'
    new_bd      = np.zeros_like(valid_bound)
    idx         = np.arange(0,train_bound[-1,0].astype(int)+1,savefreq)
    new_bd[:,0] = idx
    new_bd[:,1] = train_bound[idx.ravel().astype(int),1]
    new_bd      = new_bd[~np.isnan(new_bd).any(axis=1)]
    return new_bd
