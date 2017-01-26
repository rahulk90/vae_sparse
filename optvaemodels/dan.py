#Imports
from collections import OrderedDict
import sys, os, time
sys.path.append('../')
import numpy as np
import theano
from theano import config
theano.config.compute_test_value = 'warn'
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from optvaeutils.optimizer import adam
from theano.compile.debugmode import DebugMode
from models import BaseModel
from utils.divergences import KLGaussian,BhattacharryaGaussian
from utils.misc import readPickle, saveHDF5
from sklearn.metrics import confusion_matrix

class DAN(BaseModel, object):
    def __init__(self, params, paramFile=None, reloadFile=None, **kwargs):
        super(DAN,self).__init__(params, paramFile=paramFile, reloadFile=reloadFile, **kwargs)
        self.p_names = [k.name for k in self._getModelParams(restrict='p_')] 
    def _createParams(self):
        """ _createParams: create weights learned and used in model """
        npWeights = OrderedDict()
        DIM_HIDDEN = self.params['dim_hidden']
        for l in range(self.params['layers']):
            dim_input     = DIM_HIDDEN
            dim_output    = DIM_HIDDEN
            if l==0:
                dim_input = self.params['dim_input']
            npWeights['p_'+str(l)+'_W'] = self._getWeight((dim_input, dim_output))
            npWeights['p_'+str(l)+'_b'] = self._getWeight((dim_output, ))
        if self.params['layers']==0:
            DIM_HIDDEN    = self.params['dim_input']
        if self.params['emission_type'] == 'res':
            assert self.params['p_layers']>0,'res makes no sense w/out atleast 1 layer'
            npWeights['p_res_W']= self._getWeight((self.params['dim_input'],self.params['dim_output']))
        npWeights['p_output_W'] = self._getWeight((DIM_HIDDEN, self.params['dim_output']))
        npWeights['p_output_b'] = self._getWeight((self.params['dim_output'],))
        return npWeights
    
    def logsoftmax(self, mat):
        "Logsoftmax along dimension 1 (unless vector)"
        if mat.ndim==1:
            maxval  = mat.max()
            result  = mat-(maxval+T.log(T.sum(T.exp(mat-maxval))+1e-6))
        else:
            maxval  = mat.max(1,keepdims=True)
            result  = mat-(maxval+T.log(T.sum(T.exp(mat-maxval),axis=1,keepdims=True)+1e-6))
        return result

    def _YgivenX(self, X):
        """ Output Probabilities under p_theta(y|x) """
        inp_p    = X 
        for p_l in range(self.params['layers']):
            inp_p= self._LinearNL(self.tWeights['p_'+str(p_l)+'_W'], self.tWeights['p_'+str(p_l)+'_b'], inp_p)
        paramMat = T.dot(inp_p,self.tWeights['p_output_W'])+self.tWeights['p_output_b']
        if self.params['emission_type']=='res':
            paramMat += T.dot(X,self.tWeights['p_res_W'])
        lsf      = self.logsoftmax(paramMat)
        return lsf
    
    ################################    Building Objective Functions #####################
    def _cost(self, idx, mask, labels, input_dropout = 0.): 
        """ Wrapper for training/evaluation cost"""
        # pull out relevant vectors
        if input_dropout>0.:
            assert input_dropout<1.,'bad input dropout value'
            rand_th      = self.srng.binomial(p=input_dropout, size=mask.shape)
            mask_dropout = (mask*rand_th) 
        else:
            mask_dropout = mask
        input_rep        = self.jacobian_th[idx]*mask_dropout[:,:,None]
        input            = input_rep.sum(1)/mask_dropout.sum(1,keepdims=True)
        logprob          = self._YgivenX(input)
        y_pred           = T.argmax(logprob, axis=1)
        nll              = -(logprob[T.arange(labels.shape[0]), labels]).sum()
        return nll, T.exp(logprob)
    
    def setJacobian(self, newJacobian, quiet=False):
        if not quiet:
            ddim,mdim = self.dimData()
            self._p('Original dim:'+str(ddim)+', '+str(mdim))
        self.setData(newJacobian)
        if not quiet:
            ddim,mdim = self.dimData()
            self._p('New dim:'+str(ddim)+', '+str(mdim))
    ################################    Building Model #####################
    def _buildModel(self):
        """ Build DAN Graph """
        self.optimizer     = adam
        idx                = T.matrix('idx',dtype='int64')
        mask               = T.matrix('mask',dtype='int64')
        idx.tag.test_value = np.array([[0,1],[1,1]]).astype('int64')
        mask.tag.test_value= np.array([[1,0],[1,1]]).astype('int64')
        labels             = T.vector('labels',dtype='int64')
        labels.tag.test_value = np.array([0,1]).astype('int64') 
        new_jacobian       = T.matrix('idx',dtype='float32')
        if not hasattr(self,'jacobian_th'):
            raise ValueError('Specify jacobian_th in additional_attr while building model')
        self.setJacobian   = theano.function([new_jacobian],None,updates=[(self.jacobian_th,new_jacobian)])
        self.getJacobian   = theano.function([],self.jacobian_th*1.)
        self.dimData       = theano.function([],self.jacobian_th.shape)
        self.jacobNorm     = theano.function([],T.sqrt((self.jacobian_th**2).sum()))
        #Used in optimizer.py
        self._addWeights('lr',   np.asarray(self.params['lr'],dtype=config.floatX),  borrow=False)
        lr                 = self.tWeights['lr']
        cost_eval,y_probs  = self._cost(idx, mask, labels, input_dropout = 0.)
        self.accuracy      = theano.function([idx, mask, labels], [cost_eval,y_probs],name = 'Evaluate', allow_input_downcast = True)
        if 'validate_only' in self.params or 'EVALUATE' in self.params:
            self.updates_ack = True
            self.tOptWeights = []
            self._p('Not building training functions...')
            return
        cost_train,y_probs_train = self._cost(idx, mask, labels, input_dropout = self.params['input_dropout'])
        self.updates_ack = True
        model_params             = self._getModelParams()
        if self.params['opt_type']=='learn':
            self._p('ADDING JACOBIAN for updating')
            model_params.append(self.jacobian_th)
        #The training cost should be insensitive to batch size
        optimizer_up, norm_list  = self._setupOptimizer(cost_train,  
                                                        model_params,
                                                        divide_grad = T.cast(idx.shape[0],config.floatX),
                                                        lr = lr,  
                                                        rng = self.srng)#,
                                                        #reg_type =self.params['reg_type'], 
                                                        #reg_spec =self.params['reg_spec'], 
                                                        #reg_value= self.params['reg_value'],
                                                        #grad_norm = 1.,
                                                        #divide_grad = T.cast(X.shape[0],config.floatX))
        self._p('# additional updates: '+str(len(self.updates)))
        self.train      = theano.function([idx, mask, labels], [cost_train, y_probs_train], updates = optimizer_up, name = 'Train', allow_input_downcast=True)
        self._p('Done creating functions for training')
"""
Evaluate accuracy
"""
def evaluateAcc(dan, data_x, mask, data_y, batch_size):
    N   = len(data_x)
    acc = 0. 
    pred_y = []
    for bnum, st_idx in enumerate(range(0,N,batch_size)):
        end_idx = min(st_idx+batch_size, N)
        X    = data_x[st_idx:end_idx]
        M    = mask[st_idx:end_idx]
        Y    = data_y[st_idx:end_idx]
        batch_acc, y_probs = dan.accuracy(X, M, Y)
        acc += batch_acc
        pred_y += np.argmax(y_probs,1).tolist()
    cmat     = confusion_matrix(data_y, np.array(pred_y))
    return acc/float(N), cmat

"""
Learn DANs
"""
def learn(dan, dataset = None, mask= None, labels = None,
        epoch_start = 0, epoch_end = 100, batch_size = 25,
        savefile = None, 
        savefreq = None, dataset_eval = None, mask_eval = None, labels_eval = None):
    N = dataset.shape[0]
    idxlist = range(N)
    trainacclist, traintimelist, validconflist, validacclist = [], [], None, []
    for epoch in range(epoch_start, epoch_end+1):
        np.random.shuffle(idxlist)
        start_time  = time.time()
        ep_acc,ep_pred = 0.,[]
        for bnum,st_idx in enumerate(range(0,N,batch_size)):
            end_idx = min(st_idx+batch_size, N)
            idx_X   = dataset[idxlist[st_idx:end_idx]]
            mask_X  = mask[idxlist[st_idx:end_idx]] 
            labels_Y= labels[idxlist[st_idx:end_idx]]
            batch_acc, batch_probs = dan.train(idx_X, mask_X, labels_Y)
            ep_acc += batch_acc
            ep_pred+= np.argmax(batch_probs,axis=1).tolist()
        train_acc = ep_acc/float(N)
        ep_time   = (time.time()-start_time)/60.
        trainacclist.append((epoch,train_acc))
        traintimelist.append((epoch,ep_time))
        print '\n'
        jacob_norm= dan.jacobNorm()
        dan._p(('Ep(%d) Accuracy: %.4f, jacobian(%s)(%.4f), [%.4f mins]')%(epoch, train_acc, dan.params['opt_type'], jacob_norm, ep_time))
        train_cmat= confusion_matrix(labels[idxlist], np.array(ep_pred)) 
        print 'Confusion_Matrix\n',train_cmat 
        if savefreq is not None and epoch%savefreq==0:
            dan ._p(('Saving at epoch %d'%epoch))
            dan._saveModel(fname=savefile+'-EP'+str(epoch))
            valid_acc, valid_conf_mat     = evaluateAcc(dan, dataset_eval, mask_eval, labels_eval, batch_size)
            validacclist.append((epoch, valid_acc))
            if validconflist is None:
                validconflist  = valid_conf_mat[:,:,None] 
            else:
                validconflist  = np.concatenate([validconflist, valid_conf_mat[:,:,None]],axis=2) 
            intermediate = {}
            intermediate['train_acc']      = np.array(trainacclist)
            intermediate['train_time']     = np.array(traintimelist)
            intermediate['valid_acc']      = np.array(validacclist)
            intermediate['valid_conf_mat'] = validconflist 
            saveHDF5(savefile+'-EP'+str(epoch)+'-stats.h5', intermediate)
    ret_map={}
    ret_map['train_acc']      = np.array(trainacclist)
    ret_map['train_time']     = np.array(traintimelist)
    ret_map['valid_acc']      = np.array(validacclist)
    ret_map['valid_conf_mat'] = validconflist
    return ret_map

if __name__=='__main__':
    print 'Initializing DAN'
    pfile = './tmp'
    from optvaeutils.parse_args_vae import params
    params['dim_input'] =2000
    params['data_type']   = 'bow'
    params['opt_type']    = 'none'
    params['opt_method']  = 'adam'
    params['anneal_finopt_rate'] = 100
    params['GRADONLY']=True
    vae   = DAN(params, paramFile=pfile)
    from datasets.load import loadDataset
    dataset = loadDataset('binarized_mnist')
    np.random.seed(1)
    idxlist = np.random.permutation(dataset['train'].shape[0])
    X = dataset['train'][idxlist[:200]].astype('float32')
    os.remove(pfile)
    import ipdb;ipdb.set_trace()
