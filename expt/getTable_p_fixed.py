from utils.misc import loadHDF5, getConfigFile, readPickle, savePickle
import os,time,sys,glob
sys.path.append('../')
import numpy as np
DIR     = './'

ctr     = 0
cmdlist = []
print 'Q intialized with previous result'
for f in glob.glob(DIR+'/chkpt-rcv2_miao-q_only/*evaluate.h5'):
    dset    = loadHDF5(f)
    print dset.keys()
    def getStatStr(name, fname):
        return fname.split(name+'-')[1].split('-')[0]
    name  = ''
    name += 'qh-'+getStatStr('qh',f)
    name += '-ql-'+getStatStr('ql',f)
    if 'ql-2' in name:
        continue
    print name, (dset['perp_0_eb'],dset['perp_f_eb'])


print 'Q intialized randomly'
for f in glob.glob(DIR+'/chkpt-rcv2_miao-q_only_random/*evaluate.h5'):
    dset    = loadHDF5(f)
    def getStatStr(name, fname):
        return fname.split(name+'-')[1].split('-')[0]
    name  = ''
    name += 'qh-'+getStatStr('qh',f)
    name += '-ql-'+getStatStr('ql',f)
    if 'ql-2' in name:
        continue
    print name, (dset['perp_0_eb'],dset['perp_f_eb'])

