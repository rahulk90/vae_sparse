""" Commands to reproduce experimental results """
from collections import OrderedDict
import sys

expt_type = 'sst_fine'
valid_expts= set(['sse_full','sst_binary','rotten_tomatoes','imdb'])

print 'Valid Expts: ',','.join(list(valid_expts))
print 'Default: ',expt_type
if len(sys.argv)>=2:
    expt_type = sys.argv[-1].strip()
    if expt_type not in valid_expts:
        raise ValueError,(expt_type+' not a valid experiment')
print 'Selected: ',expt_type,'\n'

expt_runs              = OrderedDict() 
gpu_0_half = 'THEANO_FLAGS="compiledir_format=gpu0,lib.cnmem=0.45,scan.allow_gc=False"'
gpu_0_full = 'THEANO_FLAGS="compiledir_format=gpu0,lib.cnmem=0.95,scan.allow_gc=False"'
gpu_1_half = 'THEANO_FLAGS="compiledir_format=gpu1,lib.cnmem=0.45,scan.allow_gc=False"'
gpu_1_full = 'THEANO_FLAGS="compiledir_format=gpu1,lib.cnmem=0.95,scan.allow_gc=False"'
gpu_2_half = 'THEANO_FLAGS="compiledir_format=gpu2,lib.cnmem=0.45,scan.allow_gc=False"'
gpu_2_full = 'THEANO_FLAGS="compiledir_format=gpu2,lib.cnmem=0.95,scan.allow_gc=False"'
gpu_3_half = 'THEANO_FLAGS="compiledir_format=gpu3,lib.cnmem=0.45,scan.allow_gc=False"'
gpu_3_full = 'THEANO_FLAGS="compiledir_format=gpu3,lib.cnmem=0.95,scan.allow_gc=False"'

"""
Experiments on Stanford Sentiment
"""
expt_runs['sst_fine'] = OrderedDict()
expt_runs['sst_fine']['2_fixed']   = gpu_0_full+' '+'python2.7 train_dan.py -dset sst_fine -otype fixed -numl 2 -ep 400'
expt_runs['sst_fine']['2_learn']   = gpu_1_full+' '+'python2.7 train_dan.py -dset sst_fine -otype learn -numl 2 -ep 400'
expt_runs['sst_fine']['0_fixed']   = gpu_0_full+' '+'python2.7 train_dan.py -dset sst_fine -otype fixed -numl 0 -ep 400'
expt_runs['sst_fine']['0_learn']   = gpu_1_full+' '+'python2.7 train_dan.py -dset sst_fine -otype learn -numl 0 -ep 400'

expt_runs['sst_binary'] = OrderedDict()
expt_runs['sst_binary']['2_fixed']   = gpu_0_full+' '+'python2.7 train_dan.py -dset sst_binary -otype fixed -numl 2 -ep 400'
expt_runs['sst_binary']['2_learn']   = gpu_1_full+' '+'python2.7 train_dan.py -dset sst_binary -otype learn -numl 2 -ep 400'
expt_runs['sst_binary']['0_fixed']   = gpu_0_full+' '+'python2.7 train_dan.py -dset sst_binary -otype fixed -numl 0 -ep 400'
expt_runs['sst_binary']['0_learn']   = gpu_1_full+' '+'python2.7 train_dan.py -dset sst_binary -otype learn -numl 0 -ep 400'

"""
Experiments on RT 
"""
expt_runs['rotten_tomatoes'] = OrderedDict()
expt_runs['rotten_tomatoes']['2_fixed']   = gpu_0_full+' '+'python2.7 train_dan.py -dset rotten_tomatoes -otype fixed -numl 2 -ep 400'
expt_runs['rotten_tomatoes']['2_learn']   = gpu_1_full+' '+'python2.7 train_dan.py -dset rotten_tomatoes -otype learn -numl 2 -ep 400'
expt_runs['rotten_tomatoes']['0_fixed']   = gpu_0_full+' '+'python2.7 train_dan.py -dset rotten_tomatoes -otype fixed -numl 0 -ep 400'
expt_runs['rotten_tomatoes']['0_learn']   = gpu_1_full+' '+'python2.7 train_dan.py -dset rotten_tomatoes -otype learn -numl 0 -ep 400'

"""
Experiments on IMDB 
"""
expt_runs['imdb'] = OrderedDict()
expt_runs['imdb']['2_fixed']   = gpu_0_full+' '+'python2.7 train_dan.py -dset imdb -otype fixed -numl 2 -ep 400'
expt_runs['imdb']['2_learn']   = gpu_1_full+' '+'python2.7 train_dan.py -dset imdb -otype learn -numl 2 -ep 400'
expt_runs['imdb']['0_fixed']   = gpu_0_full+' '+'python2.7 train_dan.py -dset imdb -otype fixed -numl 0 -ep 400'
expt_runs['imdb']['0_learn']   = gpu_1_full+' '+'python2.7 train_dan.py -dset imdb -otype learn -numl 0 -ep 400'

for expt in expt_runs[expt_type]:
    print 'screen -S '+expt
    print expt_runs[expt_type][expt]
