from utils.misc import loadHDF5, getConfigFile, readPickle, savePickle
import os,time,sys,glob
sys.path.append('../')
import numpy as np
DIR     = './'

ctr     = 0
cmdlist = []
for f in glob.glob(DIR+'/chkpt-rcv2_miao-*/*evaluate.h5'):
    params  = readPickle(getConfigFile(f.replace('evaluate.h5','')))[0] 
    rfile   = f.replace('evaluate.h5','EP200-params.npz')
    pfile   = f.replace('evaluate.h5','config.pkl')
    if params['opt_type']=='finopt':
        name = 'pl-'+str(params['p_layers'])+'-qh-'+str(params['q_dim_hidden'])+'-ql-'+str(params['q_layers'])+'-M'+str(params['n_steps'])
    else:
        name = 'pl-'+str(params['p_layers'])+'-qh-'+str(params['q_dim_hidden'])+'-ql-'+str(params['q_layers'])+'-M1'
    cmd      = ''
    cmd+='names["'+name+'"] = {}\n'
    cmd+='names["'+name+'"]["pfile"] = "'+pfile+'"\n'
    cmd+='names["'+name+'"]["rfile"] = "'+rfile+'"\n'
    cmdlist.append( cmd)

with open('template_parallel_qvary.py','r') as f:
    data = f.read() 
cmdreplace =''
indices    = []
for idx, cmd in enumerate(cmdlist):
    print idx, cmd
    if idx>0 and idx%2==0:
        with open('tmplate_'+str(idx)+'.py','w') as f:
            f.write(data.replace('<INSERT HERE>',cmdreplace))
        indices.append(idx)
        cmdreplace = ''
    cmdreplace += cmd+'\n'
with open('tmplate_'+str(idx+1)+'.py','w') as f:
    f.write(data.replace('<INSERT HERE>',cmdreplace))
indices.append(idx+1)


for ctr,idx in enumerate(indices):
    if ctr%2==0:
        gpu = '0'
    else:
        gpu = '1'
    print 'THEANO_FLAGS="compiledir_format=gpu'+gpu+',lib.cnmem=0.95,scan.allow_gc=False" python2.7 tmplate_'+str(idx)+'.py'
