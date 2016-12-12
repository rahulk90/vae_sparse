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

    if params['projected_gradient']:
        name += '_pg'
    if params['batch_normalize']:
        name += '_bn'
    return name
