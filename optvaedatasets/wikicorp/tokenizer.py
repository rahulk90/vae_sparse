from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import re,time,os
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
import inflect
p = inflect.engine()
with open('stop_words.txt') as f:
    sw    = [k.strip() for k in f.readlines()]
stopWords = set(stopwords.words('english')+sw+['--'])
nouns     = set({x.name().split('.', 1)[0] for x in wn.all_synsets('n')})

if not os.path.exists('WestburyLab.wikicorp.201004.txt'):
    if not os.path.exists('WestburyLab.wikicorp.201004.txt.bz2'):
        raise ValueError('run ../wikicorp.py to download the .txt.bz2 file to the wikicorp directory')
    print 'Unzipping .bz2 file....'
    os.system('bzip2 -d WestburyLab.wikicorp.201004.txt.bz2')

#Include words with hyphens as unique words and 
#split up hyphenated words
splitHyphenated = True

if splitHyphenated:
    print 'SPLIT HYPHENATED'
#DOCNAME    = 'small.txt'; NDOCS=216 #Create small.txt as a subset of the bigger corpus and restrict #docs
DOCNAME    = 'WestburyLab.wikicorp.201004.txt';NDOCS      = 3035070

doclist    = []
"""
1) Remove stop words
2) Remove words w/ freq <= 1
3) (optional keep hyphenated words)
"""

def printTime(task, start, end):
    print 'Task: ',task,' ',(end-start)/60.,' minutes taken'

start_time = time.time()
with open(DOCNAME) as f:
    document = ''
    dnum     = 0
    printed  = False
    for line in f:
        if "---END.OF.DOCUMENT---" in line:
            dnum += 1
            if splitHyphenated:
                document = document.strip().replace('-',' ')
            ws  = re.sub('[^\w\s]','', document.strip())
            ws_lower = ws.lower()
            wsd = re.sub(r'\d', '', ws_lower)
            doclist.append([word.strip() for word in wsd.split(' ') if word not in stopWords])
            """
            doclist.append()
            newdoc   = []
            for word in wsd.split(' '):
                if word not in stopWords:
                    ww = word.strip()
                    sing_ww = p.singular_noun(word)
                    if ww in nouns and sing_ww:
                        newdoc.append(sing_ww)
                    else:
                        newdoc.append(ww)
            """
            document = ''
            printed  = False
        else:
            document += ' '+line.strip()
        if dnum>0 and dnum%1000==0 and not printed:
            printed = True
            print '(',dnum,')',
end_time = time.time()
printTime('Collecting Word Counts', start_time, end_time)


start_time = time.time()
from collections import defaultdict
frequency = defaultdict(int)
for doc in doclist:
    for word in doc:
        frequency[word] += 1
end_time = time.time()
printTime('Estimating Frequency', start_time, end_time)

#Smallest frequency words
start_time = time.time()
min_freq   = np.inf
min_freq_w = ''
for w in frequency:
    if frequency[w]<min_freq:
        min_freq   = frequency[w]
        min_freq_w = w
print 'Smallest Frequency: ',min_freq_w, min_freq
end_time   = time.time()
printTime('Finding smallest frequency: ',start_time, end_time)

start_time = time.time()
doclist = [[word for word in doc if (frequency[word] > 1 and len(word)>2)] for doc in doclist]
end_time = time.time()
printTime('Restricting words in documents: ',start_time, end_time)

#Gensim
start_time = time.time()
import gensim
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from gensim import corpora
dictionary = corpora.Dictionary(doclist)
corpus     = [dictionary.doc2bow(doc) for doc in doclist]
end_time = time.time()
printTime('Generating Corpus and BOW',start_time,end_time)

#Convert to a sparse matrix
from scipy.sparse import csc_matrix
sparse_corpus = gensim.matutils.corpus2csc(corpus).T

import h5py,os
from utils.sparse_utils import saveSparseHDF5
basename = DOCNAME.split('.txt')[0]
if splitHyphenated:
    basename+='-split_hyphen'
print 'Saving to ',basename,' .txt/.h5'
print 'NDOCS: ',NDOCS, 'Shape:',sparse_corpus.shape
os.system('rm -rf '+basename+'.h5; rm -rf '+basename+'.feat')
saveSparseHDF5(sparse_corpus, 'dataset', basename+'.h5')
with open(basename+'.feat','w') as f:
    f.write('\n'.join([('%s %d')%(dictionary.get(idx),frequency[dictionary.get(idx)]) for idx in range(len(dictionary))]))
import ipdb;ipdb.set_trace()
