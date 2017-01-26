"""
Parse command line and store result in params
"""
import argparse,copy
from collections import OrderedDict
p = argparse.ArgumentParser(description="Arguments for DAN")
parser = argparse.ArgumentParser()

#Model specification
parser.add_argument('-dh','--dim_hidden', action='store', default = 400, help='Hidden dimensions (in p)', type=int)
parser.add_argument('-numl','--layers', action='store',default = 2, help='#Layers in Generative Model', type=int)
parser.add_argument('-idrop','--input_dropout', action='store',default = 0.2, help='Dropout words at input',type=float)
parser.add_argument('-etype','--emission_type', action='store',default = 'mlp', help='Dropout words at input',type=str,choices=['mlp','res'])
parser.add_argument('-nl','--nonlinearity', action='store',default = 'relu', help='Nonlinarity',type=str, choices=['relu','tanh','softplus'])

#Optimization
parser.add_argument('-ischeme','--init_scheme', action='store',default = 'uniform', help='Type of init for weights', type=str, choices=['uniform','normal','xavier','he'])
parser.add_argument('-lky','--leaky_param', action='store',default =0., help='Leaky ReLU parameter',type=float)
parser.add_argument('-iw','--init_weight', action='store',default = 0.1, help='Range to initialize weights during learning',type=float)
parser.add_argument('-dset','--dataset', action='store',default = 'sst_binary', help='Dataset', type=str, choices=['sst_binary','sst_fine','rotten_tomatoes','imdb'])
parser.add_argument('-dset_wvecs','--dataset_wvecs', action='store',default = 'wikicorp', help='Dataset for using Jacobian vecs', type=str, choices=['20newsgroups_miao','wikicorp'])
parser.add_argument('-jloc', '--jacobian_location', action='store',default = 'NOTFOUND', help='Must set the location of file containing jacobian', type=str)
parser.add_argument('-jtype','--jacobian_type', action='store', default = 'ejacob', help='Jacobian Vector Type', type=str, choices=['ejacob','ejacob-probs','ejacob-energy'])
parser.add_argument('-otype','--opt_type', action='store',default = 'fixed', help='Keep vectors fixed or learn them', type=str, choices=['fixed','learn'])
parser.add_argument('-lr','--lr', action='store',default = 8e-4, help='Learning rate', type=float)
parser.add_argument('-opt','--optimizer', action='store',default = 'adam', help='Optimizer',choices=['adam'])
parser.add_argument('-bs','--batch_size', action='store',default =25, help='Batch Size',type=int)

#Setup 
parser.add_argument('-uid','--unique_id', action='store',default = 'uid',help='Unique Identifier',type=str)
parser.add_argument('-seed','--seed', action='store',default = 1, help='Random Seed',type=int)
parser.add_argument('-dir','--savedir', action='store',default = './chkpt', help='Prefix for savedir',type=str)
parser.add_argument('-ep','--epochs', action='store',default = 500, help='MaxEpochs',type=int)
parser.add_argument('-reload','--reloadFile', action='store',default = './NOSUCHFILE', help='Reload from saved model',type=str)
parser.add_argument('-params','--paramFile', action='store',default = './NOSUCHFILE', help='Reload parameters from saved model',type=str)
parser.add_argument('-sfreq','--savefreq', action='store',default = 10, help='Frequency of saving',type=int)

#Regularization
parser.add_argument('-reg','--reg_type', action='store',default = 'l2', help='Type of regularization',type=str,choices=['l1','l2','none'])
parser.add_argument('-rv','--reg_value', action='store',default = 0.01, help='Amount of regularization',type=float)
parser.add_argument('-rspec','--reg_spec', action='store',default = 'DONOTMATCH', help='String to match parameters',type=str)

#Type of model being learned
params = vars(parser.parse_args())

hmap       = OrderedDict() 
hmap['lr']             ='lr'
hmap['dim_hidden']     ='dh'
hmap['layers']         ='numl'
hmap['nonlinearity']   ='nl'
hmap['input_dropout']  = 'idrop'
hmap['batch_size']     ='bs'
hmap['epochs']         ='ep'
hmap['opt_type']       ='otype'
combined   = ''
for k in hmap:
    if k in params:
        if type(params[k]) is float:
            combined+=hmap[k]+'-'+('%.1e')%(params[k])+'-'
        else:
            combined+=hmap[k]+'-'+str(params[k])+'-'
params['unique_id'] = combined[:-1]+str(params['reg_type'])+str(params['reg_value'])+str(params['reg_spec'])+'-'+params['unique_id']
params['unique_id'] = 'DAN_'+params['unique_id'].replace('.','_')
