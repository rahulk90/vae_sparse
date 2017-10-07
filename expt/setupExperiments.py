""" Commands to reproduce experimental results """
from collections import OrderedDict
import sys

expt_type = 'rcv2_tfidf'
valid_expts= set(['20newsgroups_norm','20newsgroups_tfidf','rcv2_norm','rcv2_tfidf','rcv2_q_vary','rcv2_p_fixed','rcv2_p_fixed_random',
    'wikicorp','wikicorp_sparsity','wikicorp-large','wikicorp_evaluate','wikicorp_evaluate','wikicorp_mixed_training'])

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
Experiments on 20newsgroups
June 11- Rerun
"""
expt_runs['20newsgroups_norm'] = OrderedDict()
expt_runs['20newsgroups_norm']['2_none']   = gpu_0_full+' '+'python2.7 train.py -dset 20newsgroups -ds 100 -nl relu -otype none -pl 2 -ns 100 -ep 200'
expt_runs['20newsgroups_norm']['2_finopt'] = gpu_1_full+' '+'python2.7 train.py -dset 20newsgroups -ds 100 -nl relu -otype finopt -pl 2 -ns 100 -ep 200'
expt_runs['20newsgroups_norm']['0_none']   = gpu_0_full+' '+'python2.7 train.py -dset 20newsgroups -ds 100 -nl relu -otype none -pl 0 -ns 100 -ep 200'
expt_runs['20newsgroups_norm']['0_finopt'] = gpu_1_full+' '+'python2.7 train.py -dset 20newsgroups -ds 100 -nl relu -otype finopt -pl 0 -ns 100 -ep 200'

expt_runs['20newsgroups_tfidf'] = OrderedDict()
expt_runs['20newsgroups_tfidf']['2_none']   = gpu_0_full+' '+'python2.7 train.py -dset 20newsgroups -ds 100 -itype tfidf -nl relu -otype none -pl 2 -ns 100 -ep 200'
expt_runs['20newsgroups_tfidf']['2_finopt'] = gpu_1_full+' '+'python2.7 train.py -dset 20newsgroups -ds 100 -itype tfidf -nl relu -otype finopt -pl 2 -ns 100 -ep 200'
expt_runs['20newsgroups_tfidf']['0_none']   = gpu_0_full+' '+'python2.7 train.py -dset 20newsgroups -ds 100 -itype tfidf -nl relu -otype none -pl 0 -ns 100 -ep 200' 
expt_runs['20newsgroups_tfidf']['0_finopt'] = gpu_1_full+' '+'python2.7 train.py -dset 20newsgroups -ds 100 -itype tfidf -nl relu -otype finopt -pl 0 -ns 100 -ep 200'

"""
Experiments on RCV2
"""
expt_runs['rcv2_norm'] = OrderedDict()
expt_runs['rcv2_norm']['2_none']   = gpu_0_full+' '+'python2.7 train.py -dset rcv2 -ds 100 -nl relu -otype none -pl 2 -ns 100 -ep 200'
expt_runs['rcv2_norm']['2_finopt'] = gpu_1_full+' '+'python2.7 train.py -dset rcv2 -ds 100 -nl relu -otype finopt -pl 2 -ns 100 -ep 200'
expt_runs['rcv2_norm']['0_none']   = gpu_0_full+' '+'python2.7 train.py -dset rcv2 -ds 100 -nl relu -otype none -pl 0 -ns 100 -ep 200'
expt_runs['rcv2_norm']['0_finopt'] = gpu_1_full+' '+'python2.7 train.py -dset rcv2 -ds 100 -nl relu -otype finopt -pl 0 -ns 100 -ep 200'

expt_runs['rcv2_tfidf'] = OrderedDict()
expt_runs['rcv2_tfidf']['2_none']   = gpu_0_full+' '+'python2.7 train.py -dset rcv2 -ds 100 -itype tfidf -nl relu -otype none -pl 2 -ns 100 -ep 200'
expt_runs['rcv2_tfidf']['2_finopt'] = gpu_1_full+' '+'python2.7 train.py -dset rcv2 -ds 100 -itype tfidf -nl relu -otype finopt -pl 2 -ns 100 -ep 200'
expt_runs['rcv2_tfidf']['0_none']   = gpu_0_full+' '+'python2.7 train.py -dset rcv2 -ds 100 -itype tfidf -nl relu -otype none -pl 0 -ns 100 -ep 200' 
expt_runs['rcv2_tfidf']['0_finopt'] = gpu_1_full+' '+'python2.7 train.py -dset rcv2 -ds 100 -itype tfidf -nl relu -otype finopt -pl 0 -ns 100 -ep 200'
expt_runs['rcv2_tfidf']['2-ar10k']   = gpu_0_full+' '+'python2.7 train.py -dset rcv2 -ds 100 -itype tfidf -nl relu -otype none -pl 2 -ns 100 -ep 200 -ar 10000'
expt_runs['rcv2_tfidf']['0-ar10k']   = gpu_0_full+' '+'python2.7 train.py -dset rcv2 -ds 100 -itype tfidf -nl relu -otype none -pl 0 -ns 100 -ep 200 -ar 10000'
expt_runs['rcv2_tfidf']['2-ar50k']   = gpu_0_full+' '+'python2.7 train.py -dset rcv2 -ds 100 -itype tfidf -nl relu -otype none -pl 2 -ns 100 -ep 200 -ar 50000'
expt_runs['rcv2_tfidf']['0-ar50k']   = gpu_0_full+' '+'python2.7 train.py -dset rcv2 -ds 100 -itype tfidf -nl relu -otype none -pl 0 -ns 100 -ep 200 -ar 50000'
expt_runs['rcv2_tfidf']['2-ar100k']   = gpu_0_full+' '+'python2.7 train.py -dset rcv2 -ds 100 -itype tfidf -nl relu -otype none -pl 2 -ns 100 -ep 200 -ar 100000'
expt_runs['rcv2_tfidf']['0-ar100k']   = gpu_0_full+' '+'python2.7 train.py -dset rcv2 -ds 100 -itype tfidf -nl relu -otype none -pl 0 -ns 100 -ep 200 -ar 100000'


expt_runs['rcv2_q_vary']= OrderedDict() 
expt_runs['rcv2_q_vary']['3-400-none']   = gpu_0_full+' '+'python2.7 train.py -dset rcv2 -ds 100 -itype tfidf -nl relu -otype none -pl 2 -ns 100 -ep 200 -qh 400 -ql 3 -uid qvary'
expt_runs['rcv2_q_vary']['3-400-finopt']   = gpu_1_full+' '+'python2.7 train.py -dset rcv2 -ds 100 -itype tfidf -nl relu -otype finopt -pl 2 -ns 100 -ep 200 -qh 400 -ql 3 -uid qvary'
expt_runs['rcv2_q_vary']['2-400-none']   = gpu_0_full+' '+'python2.7 train.py -dset rcv2 -ds 100 -itype tfidf -nl relu -otype none -pl 2 -ns 100 -ep 200 -qh 400 -ql 2 -uid qvary'
expt_runs['rcv2_q_vary']['2-400-finopt']   = gpu_1_full+' '+'python2.7 train.py -dset rcv2 -ds 100 -itype tfidf -nl relu -otype finopt -pl 2 -ns 100 -ep 200 -qh 400 -ql 2 -uid qvary'
expt_runs['rcv2_q_vary']['1-400-none']   = gpu_0_full+' '+'python2.7 train.py -dset rcv2 -ds 100 -itype tfidf -nl relu -otype none -pl 2 -ns 100 -ep 200 -qh 400 -ql 1 -uid qvary'
expt_runs['rcv2_q_vary']['1-400-finopt']   = gpu_1_full+' '+'python2.7 train.py -dset rcv2 -ds 100 -itype tfidf -nl relu -otype finopt -pl 2 -ns 100 -ep 200 -qh 400 -ql 1 -uid qvary'
expt_runs['rcv2_q_vary']['3-100-none']   = gpu_0_full+' '+'python2.7 train.py -dset rcv2 -ds 100 -itype tfidf -nl relu -otype none -pl 2 -ns 100 -ep 200 -qh 100 -ql 3 -uid qvary'
expt_runs['rcv2_q_vary']['3-100-finopt']   = gpu_1_full+' '+'python2.7 train.py -dset rcv2 -ds 100 -itype tfidf -nl relu -otype finopt -pl 2 -ns 100 -ep 200 -qh 100 -ql 3 -uid qvary'
expt_runs['rcv2_q_vary']['2-100-none']   = gpu_0_full+' '+'python2.7 train.py -dset rcv2 -ds 100 -itype tfidf -nl relu -otype none -pl 2 -ns 100 -ep 200 -qh 100 -ql 2 -uid qvary'
expt_runs['rcv2_q_vary']['2-100-finopt']   = gpu_1_full+' '+'python2.7 train.py -dset rcv2 -ds 100 -itype tfidf -nl relu -otype finopt -pl 2 -ns 100 -ep 200 -qh 100 -ql 2 -uid qvary'
expt_runs['rcv2_q_vary']['1-100-none']   = gpu_0_full+' '+'python2.7 train.py -dset rcv2 -ds 100 -itype tfidf -nl relu -otype none -pl 2 -ns 100 -ep 200 -qh 100 -ql 1 -uid qvary'
expt_runs['rcv2_q_vary']['1-100-finopt']   = gpu_1_full+' '+'python2.7 train.py -dset rcv2 -ds 100 -itype tfidf -nl relu -otype finopt -pl 2 -ns 100 -ep 200 -qh 100 -ql 1 -uid qvary'

"""
Experiments on the WikiCorpus Dataset
"""

#chronos
expt_runs['wikicorp'] = OrderedDict() 
expt_runs['wikicorp']['2-none']   = gpu_0_full+' '+'python2.7 train.py -dset wikicorp -ds 100 -itype tfidf -nl relu -otype none -pl 2 -ns 100 -ep 500'
expt_runs['wikicorp']['2-finopt'] = gpu_1_full+' '+'python2.7 train.py -dset wikicorp -ds 100 -itype tfidf -nl relu -otype finopt -pl 2 -ns 100 -ep 50'
expt_runs['wikicorp']['0-none']   = gpu_0_full+' '+'python2.7 train.py -dset wikicorp -ds 100 -itype tfidf -nl relu -otype none -pl 0 -ns 100 -ep 500'
expt_runs['wikicorp']['0-finopt'] = gpu_1_full+' '+'python2.7 train.py -dset wikicorp -ds 100 -itype tfidf -nl relu -otype finopt -pl 0 -ns 100 -ep 50'
expt_runs['wikicorp']['2-ar10k']   = gpu_0_full+' '+'python2.7 train.py -dset wikicorp -ds 100 -itype tfidf -nl relu -otype none -pl 2 -ns 100 -ep 500 -ar 10000'
#expt_runs['wikicorp']['0-ar10k']   = gpu_0_full+' '+'python2.7 train.py -dset wikicorp -ds 100 -itype tfidf -nl relu -otype none -pl 0 -ns 100 -ep 500 -ar 10000'
expt_runs['wikicorp']['2-ar50k']   = gpu_0_full+' '+'python2.7 train.py -dset wikicorp -ds 100 -itype tfidf -nl relu -otype none -pl 2 -ns 100 -ep 500 -ar 50000'
#expt_runs['wikicorp']['0-ar50k']   = gpu_0_full+' '+'python2.7 train.py -dset wikicorp -ds 100 -itype tfidf -nl relu -otype none -pl 0 -ns 100 -ep 500 -ar 50000'
expt_runs['wikicorp']['2-ar100k']   = gpu_0_full+' '+'python2.7 train.py -dset wikicorp -ds 100 -itype tfidf -nl relu -otype none -pl 2 -ns 100 -ep 500 -ar 100000'
#expt_runs['wikicorp']['0-ar100k']   = gpu_0_full+' '+'python2.7 train.py -dset wikicorp -ds 100 -itype tfidf -nl relu -otype none -pl 0 -ns 100 -ep 500 -ar 100000'

#r730
expt_runs['wikicorp_sparsity'] = OrderedDict() 
expt_runs['wikicorp_sparsity']['1000-2-finopt'] = gpu_0_full+' '+'python2.7 train.py -dset wikicorp_1000 -ds 100 -itype tfidf -nl relu -otype finopt -pl 2 -ns 100 -ep 50'
expt_runs['wikicorp_sparsity']['5000-2-finopt'] = gpu_1_full+' '+'python2.7 train.py -dset wikicorp_5000 -ds 100 -itype tfidf -nl relu -otype finopt -pl 2 -ns 100 -ep 50'
expt_runs['wikicorp_sparsity']['10000-2-finopt'] = gpu_0_full+' '+'python2.7 train.py -dset wikicorp_10000 -ds 100 -itype tfidf -nl relu -otype finopt -pl 2 -ns 100 -ep 50'
expt_runs['wikicorp_sparsity']['1000-2-none'] = gpu_0_full+' '+'python2.7 train.py -dset wikicorp_1000 -ds 100 -itype tfidf -nl relu -otype none -pl 2 -ns 100 -ep 500'
expt_runs['wikicorp_sparsity']['5000-2-none'] = gpu_1_full+' '+'python2.7 train.py -dset wikicorp_5000 -ds 100 -itype tfidf -nl relu -otype none -pl 2 -ns 100 -ep 500'
expt_runs['wikicorp_sparsity']['10000-2-none'] = gpu_0_full+' '+'python2.7 train.py -dset wikicorp_10000 -ds 100 -itype tfidf -nl relu -otype none -pl 2 -ns 100 -ep 500'

#k80
expt_runs['wikicorp_mixed_training'] = OrderedDict()
expt_runs['wikicorp_mixed_training']['none_finopt'] = gpu_1_full+' '+'python2.7 train.py -dset wikicorp -ds 100 -itype tfidf -nl relu -otype none_finopt -pl 2 -ns 100 -ep 50'
expt_runs['wikicorp_mixed_training']['finopt_none'] = gpu_0_full+' '+'python2.7 train.py -dset wikicorp -ds 100 -itype tfidf -nl relu -otype finopt_none -pl 2 -ns 100 -ep 50'

"""
* Evaluating different models: comprise evaluating and saving Ejacob and/or CJacob
* Open and setup all the evaluation scripts to point to the relevant checkpoints
"""
expt_runs['wikicorp-large'] = OrderedDict() 
expt_runs['wikicorp-large']['2-none']   = gpu_0_full+' '+'python2.7 train.py -dset wikicorp-large -ds 300 -itype tfidf -nl relu -otype none -pl 2 -ns 200 -ep 50 -uid final'
expt_runs['wikicorp-large']['2-finopt'] = gpu_1_full+' '+'python2.7 train.py -dset wikicorp-large -ds 300 -itype tfidf -nl relu -otype finopt -pl 2 -ns 200 -ep 50 -uid final'
expt_runs['wikicorp-large']['0-none']   = gpu_0_full+' '+'python2.7 train.py -dset wikicorp-large -ds 300 -itype tfidf -nl relu -otype none -pl 0 -ns 200 -ep 50 -uid final'
expt_runs['wikicorp-large']['0-finopt'] = gpu_1_full+' '+'python2.7 train.py -dset wikicorp-large -ds 300 -itype tfidf -nl relu -otype finopt -pl 0 -ns 200 -ep 50 -uid final'

""" Evaluate Wikicorp """
expt_runs['wikicorp_evaluate'] = OrderedDict() 
expt_runs['wikicorp_evaluate']['2-none']   = gpu_0_full+' '+'python2.7 evaluateWikipedia.py pl-2-none'
expt_runs['wikicorp_evaluate']['2-finopt'] = gpu_1_full+' '+'python2.7 evaluateWikipedia.py pl-2-finopt'
expt_runs['wikicorp_evaluate']['0-none']   = gpu_0_full+' '+'python2.7 evaluateWikipedia.py pl-0-none'
expt_runs['wikicorp_evaluate']['0-finopt'] = gpu_1_full+' '+'python2.7 evaluateWikipedia.py pl-0-finopt'

""" Evaluate Wikicorp (Conditional Jacobian) """
expt_runs['wikicorp_conditional_evaluate'] = OrderedDict() 
expt_runs['wikicorp_conditional_evaluate']['2-none']   = gpu_0_full+' '+'python2.7 evaluateConditionalWikipedia.py pl-2-none'
expt_runs['wikicorp_conditional_evaluate']['2-finopt'] = gpu_1_full+' '+'python2.7 evaluateConditionalWikipedia.py pl-2-finopt'
expt_runs['wikicorp_conditional_evaluate']['0-none']   = gpu_0_full+' '+'python2.7 evaluateConditionalWikipedia.py pl-0-none'
expt_runs['wikicorp_conditional_evaluate']['0-finopt'] = gpu_1_full+' '+'python2.7 evaluateConditionalWikipedia.py pl-0-finopt'

for expt in expt_runs[expt_type]:
    print 'screen -S '+expt
    print expt_runs[expt_type][expt]
