#Imports
from collections import OrderedDict
import sys, os
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
from utils.misc import readPickle

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
            assert self.params['p_layers']>1,'res makes no sense w/out atleast 1 layer'
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
        for p_l in range(self.params['p_layers']):
            inp_p= self._LinearNL(self.tWeights['p_'+str(p_l)+'_W'], self.tWeights['p_'+str(p_l)+'_b'], inp_p)
        paramMat = T.dot(inp_p,self.tWeights['p_output_W'])+self.tWeights['p_output_b']
        if self.params['emission_type']=='res':
            paramMat += T.dot(X,self.tWeights['p_res_W'])
        lsf      = logsoftmax(paramMat)
        return lsf
    
    ################################    Building Objective Functions #####################
    def _cost(self, idx, mask, input_dropout = 0.): 
        """ Wrapper for training/evaluation cost"""
        # pull out relevant vectors
        if input_dropout>0.:
            assert input_dropout<1.,'bad input dropout value'
            rand_th      = self.srng.binomial(p=input_dropout, size=mask.shape)
            mask_dropout = (mask*rand_th) 
        else:
            mask_dropout = mask
        input_rep        = self.jacobian_th[idx]*mask_dropout[:,:,None]
        input            = input_rep.sum(1)/mask_dropout.sum(1)
        logprob          = _log_p_YgivenX(input)
        y_pred           = T.argmax(logprob, axis=1)
        nll              = -(logprob[T.arange(y.shape[0]), y]).sum()
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
        mask               = T.matrix('idx',dtype='int64')
        idx.tag.test_value = np.array([[0,1],[1,1]]).astype('int64')
        mask.tag.test_value= np.array([[1,0],[1,1]]).astype('int64')

        self.jacobian_th   = theano.shared(np.random.uniform(0,1,size=(5,self.params['dim_input'])).astype(config.floatX))
        self.setJacobian   = theano.function([new_jacobian],None,updates=[(self.jacobian,new_jacobian)])
        self.getJacobian   = theano.function([],[self.jacobian_th*1.])
        self.dimData       = theano.function([],[self.jacobian.shape])
        #Used in optimizer.py
        self._addWeights('lr',   np.asarray(self.params['lr'],dtype=config.floatX),  borrow=False)
        cost_eval,y_probs  = self._cost(idx, mask, input_dropout = 0.)
        self.accuracy      = theano.function([idx, mask], [cost_eval,y_probs])
        if 'validate_only' in self.params or 'EVALUATE' in self.params:
            self.updates_ack = True
            self.tOptWeights = []
            self._p('Not building training functions...')
            return
        lr               = self.tWeights['lr']
        cost_train,y_probs_train = self._cost(idx, mask, input_dropout = self.params['input_dropout'])
        self.updates_ack = True
        model_params             = self._getModelParams()
        if self.params['otype']=='learn':
            model_params.append(self.jacobian_th)
        optimizer_up, norm_list  = self._setupOptimizer(upperbound_train,  model_params,
                                                        lr = lr,  
                                                        rng = self.srng)#,
                                                        #reg_type =self.params['reg_type'], 
                                                        #reg_spec =self.params['reg_spec'], 
                                                        #reg_value= self.params['reg_value'],
                                                        #grad_norm = 1.,
                                                        #divide_grad = T.cast(X.shape[0],config.floatX))
        self._p('# additional updates: '+str(len(self.updates)))
        self.train      = theano.function([idx, mask], [cost_train, y_probs_train], updates = optimizer_up, name = 'Train')
        self._p('Done creating functions for training')
"""
Evaluate accuracy
"""
def evaluateAcc(dan, data_x, mask, data_y, batch_size):
    N   = len(data_x)
    acc = 0. 
    for bnum, st_idx in enumerate(range(0,N,batch_size)):
        end_idx = min(st_idx+batch_size, N)
        X    = data_x[st_idx:end_idx]
        mask = mask[st_idx:end_idx]
        Y    = data_y[st_idx:end_idx]
        batch_acc = dan.accuracy(X, mask, Y)
        acc+= batch_acc
    return acc/float(N)

"""
Learn DANs
"""
def learn()
    assert dataset is not None,'Expecting 2D dataset matrix'
    assert np.prod(dataset.shape[1:])==vae.params['dim_observations'],'dim observations incorrect'
    N = dataset.shape[0]
    idxlist = range(N)
    trainbound_0, trainbound_f, validbound_0, validbound_f, validll, klmat = [],[],[],[],[],[]
    gmulist,glcovlist,diff_entlist, diff_elbolist,batchtimelist = [],[],[],[],[]
    gdifflists = {}
    for name in vae.p_names:
        gdifflists[name] = []

    if replicate_K is not None:
        assert vae.params['opt_type']=='none','Multiple samples to evaluate expectation only valid for simple opt'
    learnBatch = None
    if vae.params['opt_type'] in ['none']:
        learnBatch = _optNone
    elif vae.params['opt_type'] in ['finopt']:
        learnBatch = _optFinopt
    else:
        raise ValueError('Invalid optimization type: '+str(vae.params['opt_type']))
    for epoch in range(epoch_start, epoch_end+1):
        np.random.shuffle(idxlist)
        start_time = time.time()
        bd_0, bd_f, gmu, glcov,diff_elbo, diff_ent, time_taken = 0, 0, 0, 0, 0, 0, 0
        for bnum,st_idx in enumerate(range(0,N,batch_size)):
            end_idx = min(st_idx+batch_size, N)
            X       = dataset[idxlist[st_idx:end_idx]]
            if X.__class__.__name__=='csr_matrix' or X.__class__.__name__=='csc_matrix':
                X   = X.toarray()
            X       = X.astype(config.floatX)
            Nbatch  = X.shape[0]
            if replicate_K is not None:
                X   = X.repeat(replicate_K,0)
            retVal  = learnBatch(vae, bnum, Nbatch, X,replicate_K, calcELBOfinal=(epoch%20==0))
            bd_0    += retVal['elbo_0']
            bd_f    += retVal['elbo_f']
            gmu     += retVal['gmu']
            glcov   += retVal['glcov']
            diff_elbo  += retVal['diff_elbo']
            diff_ent   += retVal['diff_ent']
            time_taken += retVal['time_taken']
        bd_0   /= float(N)
        bd_f   /= float(N)
        gmu    /= float(N)
        glcov  /= float(N)
        diff_elbo  /= float(N)
        diff_ent   /= float(N)
        time_taken /= float(bnum)  
        batchtimelist.append((epoch,time_taken))
        trainbound_0.append((epoch,bd_0))
        trainbound_f.append((epoch,bd_f))
        gmulist.append((epoch, gmu))
        glcovlist.append((epoch, glcov))
        diff_elbolist.append((epoch, diff_elbo))
        diff_entlist.append((epoch,diff_ent))
        end_time   = time.time()
        print '\n'
        vae._p(('Ep(%d) ELBO[0]: %.4f, ELBO[f]: %.4f, gmu: %.4f, glcov: %.4f, [%.3f, %.3f] [%.4f seconds]')%(epoch, bd_0, bd_f, gmu, glcov, diff_elbo, diff_ent, (end_time-start_time)))
        if savefreq is not None and epoch%savefreq==0:
            vae._p(('Saving at epoch %d'%epoch))
            vae._saveModel(fname=savefile+'-EP'+str(epoch))
            eval_retVal = None
            if dataset_eval is not None:
                eval_retVal = VAE_evaluate.evaluateBound(vae, dataset_eval, batch_size=batch_size)
                if vae.params['data_type']=='bow':
                    k0='perp_0'
                    kf='perp_f'
                elif vae.params['data_type']=='image':
                    k0='bdim_0'
                    kf='bdim_f'
                else:
                    k0='elbo_0'
                    kf='elbo_f'
                validbound_0.append((epoch,eval_retVal[k0]))
                validbound_f.append((epoch,eval_retVal[kf]))
                vae._p(('\nEp (%d): Valid[0]: %.4f, Valid[f]: %.4f')%(epoch, eval_retVal[k0], eval_retVal[kf]))
            intermediate = {}
            intermediate['train_bound_0'] = np.array(trainbound_0)
            intermediate['train_bound_f'] = np.array(trainbound_f)
            if vae.params['data_type']=='bow':
                k0 = 'valid_perp_0'
                kf = 'valid_perp_f'
            else:
                k0 = 'valid_bound_0'
                kf = 'valid_bound_f'
            intermediate[k0] = np.array(validbound_0)
            intermediate[kf] = np.array(validbound_f)
            intermediate['batch_time'] = np.array(batchtimelist)
            intermediate['samples']    = VAE_evaluate.sample(vae)
            intermediate['gmu']        = np.array(gmulist)
            intermediate['glcov']      = np.array(glcovlist)
            intermediate['diff_elbo']= np.array(diff_elbolist)
            intermediate['diff_ent']= np.array(diff_entlist)
            if eval_retVal and 'debug' in eval_retVal: 
                intermediate['debug']= np.array(eval_retVal['debug'])
            saveHDF5(savefile+'-EP'+str(epoch)+'-stats.h5', intermediate)
    ret_map={}
    ret_map['train_bound_0'] = np.array(trainbound_0)
    ret_map['train_bound_f'] = np.array(trainbound_f)
    ret_map['batch_time'] = np.array(batchtimelist)
    if vae.params['data_type']=='bow':
        k0 = 'valid_perp_0'
        kf = 'valid_perp_f'
    else:
        k0 = 'valid_bound_0'
        kf = 'valid_bound_f'
    ret_map[k0] = np.array(validbound_0)
    ret_map[kf] = np.array(validbound_f)
    ret_map['samples']       = VAE_evaluate.sample(vae)
    ret_map['gmu']           = np.array(gmulist)
    ret_map['glcov']         = np.array(glcovlist)
    ret_map['diff_elbo']= np.array(diff_elbolist)
    ret_map['diff_ent']= np.array(diff_entlist)
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
