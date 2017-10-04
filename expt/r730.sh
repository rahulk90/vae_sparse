THEANO_FLAGS="compiledir_format=gpu0,lib.cnmem=0.95,scan.allow_gc=False" python2.7 train.py -dset wikicorp_1000 -ds 100 -itype tfidf -nl relu -otype finopt -pl 2 -ns 100 -ep 52
THEANO_FLAGS="compiledir_format=gpu0,lib.cnmem=0.95,scan.allow_gc=False" python2.7 train.py -dset wikicorp_10000 -ds 100 -itype tfidf -nl relu -otype none -pl 2 -ns 100 -ep 52

THEANO_FLAGS="compiledir_format=gpu1,lib.cnmem=0.95,scan.allow_gc=False" python2.7 train.py -dset wikicorp_5000 -ds 100 -itype tfidf -nl relu -otype finopt -pl 2 -ns 100 -ep 52
THEANO_FLAGS="compiledir_format=gpu1,lib.cnmem=0.95,scan.allow_gc=False" python2.7 train.py -dset wikicorp_5000 -ds 100 -itype tfidf -nl relu -otype none -pl 2 -ns 100 -ep 52

THEANO_FLAGS="compiledir_format=gpu0,lib.cnmem=0.95,scan.allow_gc=False" python2.7 train.py -dset wikicorp_10000 -ds 100 -itype tfidf -nl relu -otype finopt -pl 2 -ns 100 -ep 52
THEANO_FLAGS="compiledir_format=gpu0,lib.cnmem=0.95,scan.allow_gc=False" python2.7 train.py -dset wikicorp_1000 -ds 100 -itype tfidf -nl relu -otype none -pl 2 -ns 100 -ep 52
