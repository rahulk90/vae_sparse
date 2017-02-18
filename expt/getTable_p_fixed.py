from utils.misc import loadHDF5, getConfigFile, readPickle, savePickle
import os,time,sys,glob
sys.path.append('../')
import numpy as np
DIR     = './'

ctr     = 0
cmdlist = []
for f in glob.glob(DIR+'/chkpt-rcv2_miao-q_only/*evaluate.h5'):
    dset    = loadHDF5(f)
    'e-04-ph-400-qh-100-ds-100-pl-2-ql-3-nl-relu-bs-500-ep-100-plr-1_0e-02-ar-0-otype-q_only-ns-100-om-adam-etype-mlp-ll-mult-itype-tfidf-idrop-1_0e-04l20_01_-uid-evaluate.h5'
    def getStatStr(name, fname):
        return fname.split(name+'-')[1].split('-')[0]
    name  = ''
    name += 'qh-'+getStatStr('qh',f)
    name += '-ql-'+getStatStr('ql',f)
    print name, (dset['perp_0_eb'],dset['perp_f_eb'])

