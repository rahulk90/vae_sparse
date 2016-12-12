import os,urllib,tarfile,gzip
from utils.misc import downloadData 
from collections import OrderedDict

locations = OrderedDict() 
locations['test_pt0.dat.gz'] = 'http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a12-token-files/lyrl2004_tokens_test_pt0.dat.gz'
locations['test_pt1.dat.gz'] = 'http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a12-token-files/lyrl2004_tokens_test_pt1.dat.gz'
locations['test_pt2.dat.gz'] = 'http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a12-token-files/lyrl2004_tokens_test_pt2.dat.gz'
locations['test_pt3.dat.gz'] = 'http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a12-token-files/lyrl2004_tokens_test_pt3.dat.gz'
locations['train.dat.gz']    = 'http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a12-token-files/lyrl2004_tokens_train.dat.gz'
downloadData('./', locations)

all_data = ''
for k in locations:
    print 'Reading ',k
    with gzip.open('./'+k,'r') as f:
        data = f.read()
        if data[-1]=='\n':
            all_data += data
        else:
            print 'adding backslash'
            all_data += data+'\n'
with open('train.txt','w') as f:
    f.write(all_data)
print 'Done'
