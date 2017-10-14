THEANO_FLAGS="compiledir_format=gpu1,gpuarray.preallocate=0.3,scan.allow_gc=False" python2.7 train.py -dset 20newsgroups -ds 100 -nl relu -otype none -pl 2 -ns 100 -ep 200
THEANO_FLAGS="compiledir_format=gpu1,gpuarray.preallocate=0.3,scan.allow_gc=False" python2.7 train.py -dset 20newsgroups -ds 100 -nl relu -otype finopt -pl 2 -ns 100 -ep 200
THEANO_FLAGS="compiledir_format=gpu1,gpuarray.preallocate=0.3,scan.allow_gc=False" python2.7 train.py -dset 20newsgroups -ds 100 -nl relu -otype none -pl 0 -ns 100 -ep 200
THEANO_FLAGS="compiledir_format=gpu1,gpuarray.preallocate=0.3,scan.allow_gc=False" python2.7 train.py -dset 20newsgroups -ds 100 -nl relu -otype finopt -pl 0 -ns 100 -ep 200
THEANO_FLAGS="compiledir_format=gpu1,gpuarray.preallocate=0.3,scan.allow_gc=False" python2.7 train.py -dset 20newsgroups -ds 100 -itype tfidf -nl relu -otype none -pl 2 -ns 100 -ep 200
THEANO_FLAGS="compiledir_format=gpu1,gpuarray.preallocate=0.3,scan.allow_gc=False" python2.7 train.py -dset 20newsgroups -ds 100 -itype tfidf -nl relu -otype finopt -pl 2 -ns 100 -ep 200
THEANO_FLAGS="compiledir_format=gpu1,gpuarray.preallocate=0.3,scan.allow_gc=False" python2.7 train.py -dset 20newsgroups -ds 100 -itype tfidf -nl relu -otype none -pl 0 -ns 100 -ep 200
THEANO_FLAGS="compiledir_format=gpu1,gpuarray.preallocate=0.3,scan.allow_gc=False" python2.7 train.py -dset 20newsgroups -ds 100 -itype tfidf -nl relu -otype finopt -pl 0 -ns 100 -ep 200
#gpu_0
THEANO_FLAGS="compiledir_format=gpu0,gpuarray.preallocate=0.45,scan.allow_gc=False" python2.7 train.py -dset rcv2 -ds 100 -itype tfidf -nl relu -otype none -pl 2 -ns 100 -ep 200

#gpu_1
THEANO_FLAGS="compiledir_format=gpu7,gpuarray.preallocate=0.45,scan.allow_gc=False" python2.7 train.py -dset rcv2 -ds 100 -itype tfidf -nl relu -otype finopt -pl 2 -ns 100 -ep 200
THEANO_FLAGS="compiledir_format=gpu1,gpuarray.preallocate=0.45,scan.allow_gc=False" python2.7 train.py -dset rcv2 -ds 100 -nl relu -otype finopt -pl 0 -ns 100 -ep 200 -itype normalize

#gpu_2
THEANO_FLAGS="compiledir_format=gpu2,gpuarray.preallocate=0.45,scan.allow_gc=False" python2.7 train.py -dset rcv2 -ds 100 -itype tfidf -nl relu -otype finopt -pl 0 -ns 100 -ep 200
THEANO_FLAGS="compiledir_format=gpu3,gpuarray.preallocate=0.45,scan.allow_gc=False" python2.7 train.py -dset rcv2 -ds 100 -nl relu -otype finopt -pl 2 -ns 100 -ep 200 -itype normalize

#gpu_3
THEANO_FLAGS="compiledir_format=gpu4,gpuarray.preallocate=0.25,scan.allow_gc=False" python2.7 train.py -dset rcv2 -ds 100 -itype tfidf -nl relu -otype none -pl 0 -ns 100 -ep 200
THEANO_FLAGS="compiledir_format=gpu5,gpuarray.preallocate=0.25,scan.allow_gc=False" python2.7 train.py -dset rcv2 -ds 100 -nl relu -otype none -pl 2 -ns 100 -ep 200 -itype normalize
THEANO_FLAGS="compiledir_format=gpu6,gpuarray.preallocate=0.25,scan.allow_gc=False" python2.7 train.py -dset rcv2 -ds 100 -nl relu -otype none -pl 0 -ns 100 -ep 200 -itype normalize
#k80 -0 
THEANO_FLAGS="compiledir_format=gpu0,gpuarray.preallocate=0.95,scan.allow_gc=False" python2.7 train.py -dset wikicorp_10000 -ds 100 -itype tfidf -nl relu -otype none -pl 2 -ns 100 -ep 52

#k80 -1 
THEANO_FLAGS="compiledir_format=gpu1,gpuarray.preallocate=0.95,scan.allow_gc=False" python2.7 train.py -dset wikicorp_10000 -ds 100 -itype tfidf -nl relu -otype finopt -pl 2 -ns 100 -ep 52

#k80 -2 
THEANO_FLAGS="compiledir_format=gpu2,gpuarray.preallocate=0.45,scan.allow_gc=False" python2.7 train.py -dset wikicorp_5000 -ds 100 -itype tfidf -nl relu -otype finopt -pl 2 -ns 100 -ep 52
THEANO_FLAGS="compiledir_format=gpu3,gpuarray.preallocate=0.45,scan.allow_gc=False" python2.7 train.py -dset wikicorp_5000 -ds 100 -itype tfidf -nl relu -otype none -pl 2 -ns 100 -ep 52

#k80 -3 
THEANO_FLAGS="compiledir_format=gpu4,gpuarray.preallocate=0.45,scan.allow_gc=False" python2.7 train.py -dset wikicorp_1000 -ds 100 -itype tfidf -nl relu -otype finopt -pl 2 -ns 100 -ep 52
THEANO_FLAGS="compiledir_format=gpu5,gpuarray.preallocate=0.45,scan.allow_gc=False" python2.7 train.py -dset wikicorp_1000 -ds 100 -itype tfidf -nl relu -otype none -pl 2 -ns 100 -ep 52

#r730 - 0  
THEANO_FLAGS="compiledir_format=gpu6,gpuarray.preallocate=0.95,scan.allow_gc=False" python2.7 train.py -dset wikicorp -ds 100 -itype tfidf -nl relu -otype none_finopt -pl 2 -ns 100 -ep 22 -sfreq 3

#r730 - 1  
THEANO_FLAGS="compiledir_format=gpu7,gpuarray.preallocate=0.95,scan.allow_gc=False" python2.7 train.py -dset wikicorp -ds 100 -itype tfidf -nl relu -otype finopt_none -pl 2 -ns 100 -ep 22 -sfreq 3

# extra run
THEANO_FLAGS="compiledir_format=gpu5,gpuarray.preallocate=0.9,scan.allow_gc=False" python2.7 train.py -dset wikicorp -ds 100 -itype norm -nl relu -otype finopt -pl 2 -ns 100 -ep 52
#gpu_0
THEANO_FLAGS="compiledir_format=gpu1,gpuarray.preallocate=0.45,scan.allow_gc=False" python2.7 train.py -dset wikicorp -ds 100 -itype tfidf -nl relu -otype none -pl 2 -ns 100 -ep 52
THEANO_FLAGS="compiledir_format=gpu2,gpuarray.preallocate=0.45,scan.allow_gc=False" python2.7 train.py -dset wikicorp -ds 100 -itype tfidf -nl relu -otype none -pl 2 -ns 100 -ep 52 -ar 10000

#gpu_1
THEANO_FLAGS="compiledir_format=gpu3,gpuarray.preallocate=0.45,scan.allow_gc=False" python2.7 train.py -dset wikicorp -ds 100 -itype tfidf -nl relu -otype finopt -pl 0 -ns 100 -ep 52
THEANO_FLAGS="compiledir_format=gpu4,gpuarray.preallocate=0.45,scan.allow_gc=False" python2.7 train.py -dset wikicorp -ds 100 -itype tfidf -nl relu -otype none -pl 2 -ns 100 -ep 52 -ar 50000

#gpu_2
THEANO_FLAGS="compiledir_format=gpu5,gpuarray.preallocate=0.9,scan.allow_gc=False" python2.7 train.py -dset wikicorp -ds 100 -itype tfidf -nl relu -otype finopt -pl 2 -ns 100 -ep 52

#gpu_3
THEANO_FLAGS="compiledir_format=gpu6,gpuarray.preallocate=0.3,scan.allow_gc=False" python2.7 train.py -dset wikicorp -ds 100 -itype tfidf -nl relu -otype none -pl 0 -ns 100 -ep 52
THEANO_FLAGS="compiledir_format=gpu7,gpuarray.preallocate=0.3,scan.allow_gc=False" python2.7 train.py -dset wikicorp -ds 100 -itype tfidf -nl relu -otype none -pl 2 -ns 100 -ep 52 -ar 100000
