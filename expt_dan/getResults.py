import os
from utils.misc import loadHDF5
import glob

for dir in glob.glob('chkpt-*'):
#for dir in glob.glob('./expts_feb23/chkpt-*'):
    for finalf in glob.glob(dir+'/*final*'):
        print dir, finalf
        data = loadHDF5(finalf)
        print data['test_acc']
