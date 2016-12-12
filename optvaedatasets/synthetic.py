from time import time
import numpy as np
from sklearn import manifold, datasets
from sklearn.utils import check_random_state

""" Synthetic datasets """
def fitReductions(dataset):
    X = dataset['train']
    n_neighbors = 10
    t0 = time()
    dataset['LLE']= manifold.LocallyLinearEmbedding(n_neighbors, 2,method='standard').fit_transform(X)
    t1 = time()
    print("%s: %.2g sec" % ('LLE', t1 - t0))

    # Perform Isomap Manifold learning.
    t0 = time()
    dataset['Isomap'] = manifold.Isomap(n_neighbors, n_components=2)\
        .fit_transform(X)
    t1 = time()
    print("%s: %.2g sec" % ('ISO', t1 - t0))
    
    # Perform Multi-dimensional scaling.
    t0 = time()
    mds = manifold.MDS(2, max_iter=100, n_init=1)
    dataset['MDS'] = mds.fit_transform(X)
    t1 = time()
    print("MDS: %.2g sec" % (t1 - t0))
    
    # Perform Spectral Embedding.
    t0 = time()
    se = manifold.SpectralEmbedding(n_components=2,
                                    n_neighbors=n_neighbors)
    dataset['Spectral Embedding'] = se.fit_transform(X)
    t1 = time()
    print("Spectral Embedding: %.2g sec" % (t1 - t0))
    
    # Perform t-distributed stochastic neighbor embedding.
    t0 = time()
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    dataset['t-SNE'] = tsne.fit_transform(X)
    t1 = time()
    print("t-SNE: %.2g sec" % (t1 - t0))

 
def _loadSynthetic(dsetname):
    if 'ball' in dsetname:
        # Variables for manifold learning.
        n_neighbors = 10
        n_samples = 1000
        # Create our sphere.
        random_state = check_random_state(0)
        p = random_state.rand(n_samples) * (2 * np.pi - 0.55)
        t = random_state.rand(n_samples) * np.pi
        # Sever the poles from the sphere.
        indices = ((t < (np.pi - (np.pi / 8))) & (t > ((np.pi / 8))))
        colors = p[indices]
        x, y, z = np.sin(t[indices]) * np.cos(p[indices]), np.sin(t[indices]) * np.sin(p[indices]), np.cos(t[indices])
        #Set train/valid/test to the same, we're going to visualize the training data anyways
        dataset = {}
        dataset['train'] = np.concatenate([x[:,None], y[:,None], z[:,None]],axis=1)
        dataset['train_y'] = colors 
        dataset['valid'] = dataset['train']
        dataset['test']  = dataset['train']
        dataset['vocabulary'] = ['x','y','z']
        dataset['data_type'] = 'real'
        fitReductions(dataset)
        return dataset
    else:
        n_points = 1000
        dataset  = {}
        dataset['train'], dataset['train_y']= datasets.samples_generator.make_s_curve(n_points, random_state=0)
        dataset['valid'] = dataset['train']
        dataset['test']  = dataset['train']
        dataset['vocabulary'] = ['x','y','z']
        dataset['data_type'] = 'real'
        fitReductions(dataset)
        return dataset


if __name__=='__main__':
    dset = _loadSynthetic('synthetic_ball')
    dset = _loadSynthetic('synthetic_s')
    import ipdb;ipdb.set_trace()
