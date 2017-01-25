"""
This file is taken from https://github.com/miyyer/dan 

The MIT License (MIT)

Copyright (c) 2015 miyyer

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE. 
"""

from glob import glob
import cPickle
import random

def compute_vocab():
    vocab = []
    vdict = {}
    trneg = glob('./imdb/aclImdb/train/neg/*.txt')
    trpos = glob('./imdb/aclImdb/train/pos/*.txt')
    tneg = glob('./imdb/aclImdb/test/neg/*.txt')
    tpos = glob('./imdb/aclImdb/test/pos/*.txt')

    split = []
    for fold in [trneg, trpos, tneg, tpos]:
        fold_docs = []
        for fname in fold:
            doc = []
            f = open(fname, 'r')
            for line in f:
                line = line.strip().replace('.', '').replace(',', '')
                line = line.replace(';', '').replace('<br />', ' ')
                line = line.replace(':', '').replace('"', '')
                line = line.replace('(', '').replace(')', '')
                line = line.replace('!', '').replace('*', '')
                line = line.replace(' - ', ' ').replace(' -- ', '')
                line = line.replace('?', '')
                line = line.lower().split()
                for word in line:
                    try:
                        vdict[word]
                    except:
                        vocab.append(word)
                        vdict[word] = len(vocab) - 1
                    doc.append(vdict[word])
            fold_docs.append(doc)
        split.append(fold_docs)
    train = []
    test = []
    for i in range(0, len(split)):
        for doc in split[i]:
            if i == 0:
                train.append((doc, 0))
            elif i == 1:
                train.append((doc, 1))
            elif i == 2:
                test.append((doc, 0))
            elif i == 3:
                test.append((doc, 1))

    print len(train), len(test)

    random.shuffle(train)
    random.shuffle(test)

    for x in range(3000, 3020):
        print i, train[x][1], ' '.join(vocab[x] for x in train[x][0])
        print '\n'

    cPickle.dump([train, test, vocab, vdict], open('./imdb/aclImdb/imdb_splits', 'wb'),\
        protocol=cPickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    compute_vocab()
