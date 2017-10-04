#gpu_1
THEANO_FLAGS="compiledir_format=gpu0,lib.cnmem=0.45,scan.allow_gc=False" python2.7 train.py -dset wikicorp -ds 100 -itype tfidf -nl relu -otype none -pl 2 -ns 100 -ep 52
THEANO_FLAGS="compiledir_format=gpu1,lib.cnmem=0.45,scan.allow_gc=False" python2.7 train.py -dset wikicorp -ds 100 -itype tfidf -nl relu -otype finopt -pl 0 -ns 100 -ep 52

#gpu_2
THEANO_FLAGS="compiledir_format=gpu1,lib.cnmem=0.45,scan.allow_gc=False" python2.7 train.py -dset wikicorp -ds 100 -itype tfidf -nl relu -otype finopt -pl 2 -ns 100 -ep 52
THEANO_FLAGS="compiledir_format=gpu0,lib.cnmem=0.45,scan.allow_gc=False" python2.7 train.py -dset wikicorp -ds 100 -itype tfidf -nl relu -otype none -pl 0 -ns 100 -ep 52

#gpu_0
THEANO_FLAGS="compiledir_format=gpu0,lib.cnmem=0.3,scan.allow_gc=False" python2.7 train.py -dset wikicorp -ds 100 -itype tfidf -nl relu -otype none -pl 2 -ns 100 -ep 52 -ar 10000
THEANO_FLAGS="compiledir_format=gpu0,lib.cnmem=0.95,scan.allow_gc=False" python2.7 train.py -dset wikicorp -ds 100 -itype tfidf -nl relu -otype none -pl 2 -ns 100 -ep 52 -ar 50000

#gpu_3
THEANO_FLAGS="compiledir_format=gpu0,lib.cnmem=0.95,scan.allow_gc=False" python2.7 train.py -dset wikicorp -ds 100 -itype tfidf -nl relu -otype none -pl 2 -ns 100 -ep 52 -ar 100000
