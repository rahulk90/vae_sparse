#k80 -0 
THEANO_FLAGS="compiledir_format=gpu0,gpuarray.preallocate=0.95,scan.allow_gc=False" python2.7 train.py -dset wikicorp_10000 -ds 100 -itype tfidf -nl relu -otype none -pl 2 -ns 100 -ep 52

#k80 -1 
THEANO_FLAGS="compiledir_format=gpu1,gpuarray.preallocate=0.95,scan.allow_gc=False" python2.7 train.py -dset wikicorp_10000 -ds 100 -itype tfidf -nl relu -otype finopt -pl 2 -ns 100 -ep 52

#k80 -2 
THEANO_FLAGS="compiledir_format=gpu2,gpuarray.preallocate=0.95,scan.allow_gc=False" python2.7 train.py -dset wikicorp_5000 -ds 100 -itype tfidf -nl relu -otype finopt -pl 2 -ns 100 -ep 52
THEANO_FLAGS="compiledir_format=gpu3,gpuarray.preallocate=0.95,scan.allow_gc=False" python2.7 train.py -dset wikicorp_5000 -ds 100 -itype tfidf -nl relu -otype none -pl 2 -ns 100 -ep 52

#k80 -3 
THEANO_FLAGS="compiledir_format=gpu4,gpuarray.preallocate=0.95,scan.allow_gc=False" python2.7 train.py -dset wikicorp_1000 -ds 100 -itype tfidf -nl relu -otype finopt -pl 2 -ns 100 -ep 52
THEANO_FLAGS="compiledir_format=gpu5,gpuarray.preallocate=0.95,scan.allow_gc=False" python2.7 train.py -dset wikicorp_1000 -ds 100 -itype tfidf -nl relu -otype none -pl 2 -ns 100 -ep 52

#r730 - 0  
THEANO_FLAGS="compiledir_format=gpu6,gpuarray.preallocate=0.95,scan.allow_gc=False" python2.7 train.py -dset wikicorp -ds 100 -itype tfidf -nl relu -otype none_finopt -pl 2 -ns 100 -ep 22

#r730 - 1  
THEANO_FLAGS="compiledir_format=gpu7,gpuarray.preallocate=0.95,scan.allow_gc=False" python2.7 train.py -dset wikicorp -ds 100 -itype tfidf -nl relu -otype finopt_none -pl 2 -ns 100 -ep 22
