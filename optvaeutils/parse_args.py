"""
Parse command line and store result in params
"""
import argparse,copy
from collections import OrderedDict
p = argparse.ArgumentParser(description="Arguments for VAE")
parser = argparse.ArgumentParser()

#Model specification
parser.add_argument('-ph','--p_dim_hidden', action='store', default = 400, help='Hidden dimensions (in p)', type=int)
parser.add_argument('-pl','--p_layers', action='store',default = 2, help='#Layers in Generative Model', type=int)
parser.add_argument('-ll','--likelihood', action='store',default = 'mult', help='Likelihood for BOW models',type=str, choices=['mult','poisson'])
parser.add_argument('-ds','--dim_stochastic', action='store',default = 100, help='Stochastic dimensions', type=int)
parser.add_argument('-ql','--q_layers', action='store',default = 2, help='#Layers in Recognition Model', type=int)
parser.add_argument('-qh','--q_dim_hidden', action='store', default = 400, help='Hidden dimensions (in q)', type=int)
parser.add_argument('-idrop','--input_dropout', action='store',default = 0.0001, help='Dropout at input',type=float)
parser.add_argument('-nl','--nonlinearity', action='store',default = 'relu', help='Nonlinarity',type=str, choices=['relu','tanh','softplus'])

#Optimization
parser.add_argument('-ischeme','--init_scheme', action='store',default = 'uniform', help='Type of init for weights', type=str, choices=['uniform','normal','xavier','he'])
parser.add_argument('-lky','--leaky_param', action='store',default =0., help='Leaky ReLU parameter',type=float)
parser.add_argument('-iw','--init_weight', action='store',default = 0.1, help='Range to initialize weights during learning',type=float)
parser.add_argument('-dset','--dataset', action='store',default = 'binarized_mnist', help='Dataset', type=str)
parser.add_argument('-lr','--lr', action='store',default = 8e-4, help='Learning rate', type=float)
parser.add_argument('-opt','--optimizer', action='store',default = 'adam', help='Optimizer',choices=['adam','rmsprop'])
parser.add_argument('-bs','--batch_size', action='store',default = 500, help='Batch Size',type=int)
parser.add_argument('-repK','--replicate_K', action='store',default = None, help='Number of samples used for the variational bound. Created by replicating the batch',type=int)

# Additional Features 
# A] lr_p to deal w/ different learning rates B] ar to deal with regularizing KL divergence 
# C] iopt to optimize inference network
# D] popt pre-optimizing variational parameters
# E] popt pre-optimizing variational parameters
parser.add_argument('-plr','--param_lr', action = 'store', default=0.01, help='Learning rates used to update mu/cov', type=float)
parser.add_argument('-ar','--anneal_rate', action = 'store', default=0, help='Number of steps before KL divergence stops being regularized', type=float)
parser.add_argument('-otype','--opt_type', action='store', default='none', choices=['none','finopt','q_only'], 
        help='none-standard training,\n finopt-optimize theta w/ optimized mu/logcov')
parser.add_argument('-ns','--n_steps', action='store', default=200, help='Number of steps of optimization to perform', type=int)
parser.add_argument('-om','--opt_method', action='store', default='adam', help='Optimization',choices=['adam'])

#Setup 
parser.add_argument('-uid','--unique_id', action='store',default = 'uid',help='Unique Identifier',type=str)
parser.add_argument('-seed','--seed', action='store',default = 1, help='Random Seed',type=int)
parser.add_argument('-dir','--savedir', action='store',default = './chkpt', help='Prefix for savedir',type=str)
parser.add_argument('-ep','--epochs', action='store',default = 500, help='MaxEpochs',type=int)
parser.add_argument('-reload','--reloadFile', action='store',default = './NOSUCHFILE', help='Reload from saved model',type=str)
parser.add_argument('-params','--paramFile', action='store',default = './NOSUCHFILE', help='Reload parameters from saved model',type=str)
parser.add_argument('-sfreq','--savefreq', action='store',default = 10, help='Frequency of saving',type=int)
parser.add_argument('-gn','--grad_noise', action='store',default = 0., help='Gradient noise',type=float)

#Regularization
parser.add_argument('-reg','--reg_type', action='store',default = 'l2', help='Type of regularization',type=str,choices=['l1','l2','none'])
parser.add_argument('-rv','--reg_value', action='store',default = 0.01, help='Amount of regularization',type=float)
#Which parameters to regularize
parser.add_argument('-rspec','--reg_spec', action='store',default = '_', help='String to match parameters (Default is generative model)',type=str)

#Type of model being learned
parser.add_argument('-itype','--input_type', action='store',default = 'normalize', help='For BOW: ',choices=['normalize','counts','tfidf'])
parser.add_argument('-etype','--emission_type', action='store',default = 'mlp',choices=['mlp','res'])
params = vars(parser.parse_args())

hmap       = OrderedDict() 
hmap['lr']           ='lr'
hmap['p_dim_hidden'] ='ph'
hmap['q_dim_hidden'] ='qh'
hmap['dim_stochastic']='ds'
hmap['p_layers']     ='pl'
hmap['q_layers']     ='ql'
hmap['nonlinearity'] ='nl'
hmap['batch_size']   ='bs'
hmap['epochs']       ='ep'
hmap['param_lr']     = 'plr'
hmap['anneal_rate']  = 'ar'
hmap['opt_type']     = 'otype'
hmap['n_steps']      = 'ns'
hmap['opt_method']   = 'om'
hmap['emission_type']= 'etype'
hmap['likelihood']   = 'll'
hmap['input_type']   = 'itype'
hmap['n_steps']      = 'ns'
hmap['input_dropout']= 'idrop'
combined   = ''
for k in hmap:
    if k in params:
        if type(params[k]) is float:
            combined+=hmap[k]+'-'+('%.1e')%(params[k])+'-'
        else:
            combined+=hmap[k]+'-'+str(params[k])+'-'
params['unique_id'] = combined[:-1]+str(params['reg_type'])+str(params['reg_value'])+str(params['reg_spec'])+'-'+params['unique_id']
params['unique_id'] = 'VAE_'+params['unique_id'].replace('.','_')
