import theano
import numpy as np
from theano import config
from collections import OrderedDict
import theano.tensor as T
"""
                                            OPTIMIZERS FOR THEANO
"""

def regularize(cost, params, reg_val, reg_type, reg_spec):
    """
    Return a theano cost
    cost: cost to regularize
    params: list of parameters
    reg_val: multiplier for regularizer
    reg_type: type of regularizer 'l1','l2'
    pnorm_str: simple regex to exclude parameters not satisfying regex
    """
    l1 = lambda p: T.sum(abs(p))
    l2 = lambda p: T.sum(p**2)
    rFxn = {}
    rFxn['l1']=l1
    rFxn['l2']=l2
    
    if reg_type=='l2' or reg_type=='l1':
        assert reg_val is not None,'Expecting reg_val to be specified'
        print "<< Reg:("+reg_type+') Reg. Val:('+str(reg_val)+') Reg. Spec.:('+reg_spec+')>>'
        regularizer= theano.shared(np.asarray(0).astype(config.floatX),name = 'reg_norm', borrow=True)
        for p in params:
            if reg_spec in p.name:
                regularizer += rFxn[reg_type](p)
                print ('<<<<<< Adding '+reg_type+' regularization for '+p.name)+' >>>>>>'
        return cost + reg_val*regularizer
    else:
        return cost

def normalize(grads, grad_norm):
    """
    grads: list of gradients
    grad_norm : None (or positive value)
    returns: gradients rescaled to satisfy norm constraints
    """
    #Check if we're clipping gradients
    if grad_norm is not None:
        assert grad_norm > 0, 'Must specify a positive value to normalize to'
        print '<<<<<< Normalizing Gradients to have norm (',grad_norm,') >>>>>>'
        g2 = 0.
        for g in grads:
            g2 += (g**2).sum()
        new_grads = []
        for g in grads:
            new_grads.append(T.switch(g2 > (grad_norm**2), g/T.sqrt(g2)*grad_norm, g))
        return new_grads
    else:
        return grads

def rescale(grads, divide_grad):
    """
    grads : list of gradients
    divide_grad : scalar or theano variable
    returns: gradients divided by provided variable
    """
    if divide_grad is not None:
        print '<<<<<< Rescaling Gradients >>>>>>'
        new_grads = []
        for g in grads:
            new_grads.append(g/divide_grad)
        return new_grads
    else:
        return grads

def adam(cost, params, lr=0.001, b1=0.1, b2=0.001, e=1e-8, opt_params = None, 
         grad_range= None, #Whether or not you would like to specify a range for grads
         grad_norm = None, #Clip gradients using normalization
         reg_type  = None,# Can be 'l1' or 'l2' or ''
         reg_value = None, #Specify the multiplier for the regularization type
         reg_spec  = 'DOESNOTMATCHANYTHING',#Restricting the weights to consider set to '' to regularize all
         divide_grad=None,  #Rescale the gradient by batch size  
         optsuffix = '', #Suffix for the set of updates. Use this if you would like to be able to update
         grad_noise= 0., rng = None #Add gradient noise using rng
        ):
    """
    ADAM Optimizer
    cost (to be minimized)
    params (list of parameters to take gradients with respect to)
    .... parameters specific to the optimization ...
    opt_params (if available, used to intialize the variables
    """
    updates = []
    regularized_cost = regularize(cost, params, reg_value, reg_type, reg_spec)
    grads = T.grad(regularized_cost, params)
    grads = rescale(grads, divide_grad)
    grads = normalize(grads, grad_norm)
    
    def getName(pname, suffix = optsuffix):
        return 'opt_'+pname+'_'+suffix

    if opt_params is None:
        opt_params=OrderedDict()

    #Track the optimization variable
    vname = getName('i')
    #Create a new shared variable if opt_params is empty or if you cant find the variable name
    if vname not in opt_params:
        i = theano.shared(np.asarray(0).astype(config.floatX), name =vname, borrow=True)
        opt_params[vname] = i
    else:
        i = opt_params[vname]
    
    #No need to reload these theano variables
    g_norm = theano.shared(np.asarray(0).astype(config.floatX),name = 'g_norm',borrow=True)
    p_norm = theano.shared(np.asarray(0).astype(config.floatX),name = 'p_norm',borrow=True)
    opt_norm = theano.shared(np.asarray(0).astype(config.floatX),name = 'opt_norm',borrow=True)
    
    #Initialization for ADAM
    i_t = i + 1.
    #b1=0.1, b2=0.001
    fix1 = 1. - (1. - b1)**i_t
    fix2 = 1. - (1. - b2)**i_t
    lr_t = lr * (T.sqrt(fix2) / fix1)
    
    if grad_noise>0:
        print '      Adding gradient noise     '
        frac  = grad_noise / (1+i_t)**0.55
        grads = [g+rng.normal(g.shape)*frac for g in grads] 

    for p, g in zip(params, grads):
        if grad_range is not None:
            print '<<<<<< ADAM: Truncating Gradients in Range +-(',grad_range,') >>>>>>'
            g = T.clip(g,-grad_range, grad_range)
        vname_m = getName('m+'+p.name)
        vname_v = getName('v+'+p.name)
        #Create a new shared variable if opt_params is empty or if you cant find the variable name
        if vname_m not in opt_params:
            m = theano.shared(p.get_value() * 0.,name = vname_m,borrow=True)
            v = theano.shared(p.get_value() * 0.,name = vname_v,borrow=True)
            opt_params[vname_m] = m
            opt_params[vname_v] = v
        else:
            m = opt_params[vname_m]
            v = opt_params[vname_v]
        m_t = (b1 * g) + ((1. - b1) * m)
        v_t = (b2 * T.sqr(g)) + ((1. - b2) * v)
        g_t = m_t / (T.sqrt(v_t) + e)
        p_t = p - (lr_t * g_t)
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
        #Update norms
        g_norm += (g**2).sum()
        p_norm += (p**2).sum() 
        opt_norm+=(m**2).sum() + (v**2).sum()
    updates.append((i, i_t))
    return updates, [T.sqrt(p_norm), T.sqrt(g_norm), T.sqrt(opt_norm), regularized_cost], opt_params 

def adamNew(cost, params, lr=0.001, b1=0.9, b2=0.999, e=1e-8, opt_params = None, gamma=1.-1e-8,
         grad_range= None, #Whether or not you would like to specify a range for grads
         grad_norm = None, #Clip gradients using normalization
         reg_type  = None,# Can be 'l1' or 'l2' or ''
         reg_value = None, #Specify the multiplier for the regularization type
         reg_spec  = 'DOESNOTMATCHANYTHING',#Restricting the weights to consider set to '' to regularize all
         divide_grad=None,  #Rescale the gradient by batch size  
         optsuffix = '', #Suffix for the set of updates. Use this if you would like to be able to update
         grad_noise= 0., rng = None #Add gradient noise using rng
        ):
    """
    ADAM Optimizer
    cost (to be minimized)
    params (list of parameters to take gradients with respect to)
    .... parameters specific to the optimization ...
    opt_params (if available, used to intialize the variables
    """
    updates = []
    regularized_cost = regularize(cost, params, reg_value, reg_type, reg_spec)
    grads = T.grad(regularized_cost, params)
    grads = rescale(grads, divide_grad)
    grads = normalize(grads, grad_norm)
    
    def getName(pname, suffix = optsuffix):
        return 'opt_'+pname+'_'+suffix

    if opt_params is None:
        opt_params=OrderedDict()

    #Track the optimization variable
    vname = getName('i')
    #Create a new shared variable if opt_params is empty or if you cant find the variable name
    if vname not in opt_params:
        i = theano.shared(np.asarray(0).astype(config.floatX), name =vname, borrow=True)
        opt_params[vname] = i
    else:
        i = opt_params[vname]
    
    #No need to reload these theano variables
    g_norm = theano.shared(np.asarray(0).astype(config.floatX),name = 'g_norm',borrow=True)
    p_norm = theano.shared(np.asarray(0).astype(config.floatX),name = 'p_norm',borrow=True)
    opt_norm = theano.shared(np.asarray(0).astype(config.floatX),name = 'opt_norm',borrow=True)
    #Initialization for ADAM
    i_t  = i + 1.
    #b1=0.9, b2=0.999   
    b1_t = b1*gamma**(i_t-1)   
    if grad_noise>0:
        print '      Adding gradient noise     '
        frac  = grad_noise / (1+i_t)**0.55
        grads = [g+rng.normal(g.shape)*frac for g in grads] 
    for p, g in zip(params, grads):
        if grad_range is not None:
            print '<<<<<< ADAM: Truncating Gradients in Range +-(',grad_range,') >>>>>>'
            g = T.clip(g,-grad_range, grad_range)
        vname_m = getName('m+'+p.name)
        vname_v = getName('v+'+p.name)
        #Create a new shared variable if opt_params is empty or if you cant find the variable name
        if vname_m not in opt_params:
            m = theano.shared(p.get_value() * 0.,name = vname_m,borrow=True)
            v = theano.shared(p.get_value() * 0.,name = vname_v,borrow=True)
            opt_params[vname_m] = m
            opt_params[vname_v] = v
        else:
            m = opt_params[vname_m]
            v = opt_params[vname_v]
        #Update ADAM parameters
        m_t   = b1_t*m + (1 - b1_t)*g                             
        v_t   = b2*v   + (1 - b2)*g**2                              
        m_hat = m_t / (1-b1**i_t) 
        v_hat = v_t / (1-b2**i_t)
        p_t   = p - (lr* m_hat) / (T.sqrt(v_hat) + e) 
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
        #Update norms
        g_norm += (g**2).sum()
        p_norm += (p**2).sum() 
        opt_norm+=(m**2).sum() + v.sum()
    updates.append((i, i_t))
    return updates, [T.sqrt(p_norm), T.sqrt(g_norm), T.sqrt(opt_norm), regularized_cost], opt_params 
