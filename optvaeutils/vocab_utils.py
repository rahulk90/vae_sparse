import numpy as np

def idxMaskFormat(dataset_old, idx2idx):
    dataset = []
    difflist= [] 
    for elem in dataset_old:
        elem_new = [idx2idx[idx] for idx in elem if idx2idx[idx]>=0.]
        dataset.append(elem_new)
        difflist.append(len(elem)-len(elem_new))
    print 'Diff Max/Min/Mean: ',np.max(difflist), np.min(difflist), np.mean(difflist)

    lenlist = [len(elem) for elem in dataset]
    MAX     = np.max(lenlist)
    print 'MAX: ',MAX,' MIN: ',np.min(lenlist)
    indices = np.zeros((len(dataset), MAX))
    mask    = np.zeros((len(dataset), MAX))
    for idx, elem in enumerate(dataset): 
        indices[idx,:len(elem)] = np.sort(np.array(elem)) 
        mask[idx,:len(elem)]    = 1. 
    return indices, mask 

def mapVocabs(idx2word_orig, vocab, vocab_singular = []):
    """
    Map from the vocabulary used in IMDB/SST/RT to wikicorp (or other dataset) 
    """
    idx2idx  = {}
    ctr_vocab, ctr_vocab_sing, ctr_notfound = 0., 0., 0.
    #Intermediate hashmaps
    word2idx          = {}
    word2idx_singular = {}
    for idx in xrange(vocab_singular.shape[0]):
        word2idx[vocab[idx]] = idx
        word2idx_singular[vocab_singular[idx]] = idx
    #Go through vocab of dataset
    for idx in xrange(len(idx2word_orig)):
        word = idx2word_orig[idx]
        if word in word2idx:
            ctr_vocab +=1.
            new_idx = word2idx[word]
        elif word in word2idx_singular:
            ctr_vocab_sing +=1.
            new_idx = word2idx_singular[word]
        else:
            ctr_notfound +=1.
            new_idx = -1
        idx2idx[idx]= new_idx
    print 'Total/Vocab/Vocan(sing)/Notfound',len(idx2word_orig),ctr_vocab,ctr_vocab_sing,ctr_notfound
    return idx2idx

def cleanDataset(idx, mask, labels):
    idx_to_remove = np.where(mask.sum(1)==0.)[0]
    return np.delete(idx, idx_to_remove, 0), np.delete(mask, idx_to_remove, 0), np.delete(labels, idx_to_remove, 0) 

def reformatDataset(dataset, dataset_wvecs):
    if 'vocabulary_singular' in dataset_wvecs:
        idx2idx = mapVocabs(dataset['idx2word'], dataset_wvecs['vocabulary'], vocab_singular = dataset_wvecs['vocabulary_singular']) 
    else: 
        idx2idx = mapVocabs(dataset['idx2word'], dataset_wvecs['vocabulary'])
    # Set the datasets to use indicies from Jacobian Vectors
    # Ignore empty training sets
    for prefix in ['train','valid','test']:
        dataset[prefix+'_x'], dataset[prefix+'_mask'] = idxMaskFormat(dataset[prefix+'_x'], idx2idx)
        dataset[prefix+'_x'], dataset[prefix+'_mask'], dataset[prefix+'_y'] = cleanDataset(dataset[prefix+'_x'], dataset[prefix+'_mask'], dataset[prefix+'_y'])
        print prefix, dataset[prefix+'_x'].shape, dataset[prefix+'_mask'].shape, dataset[prefix+'_y'].shape
    return dataset
