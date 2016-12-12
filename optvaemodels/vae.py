import six.moves.cPickle as pickle
from collections import OrderedDict
import sys, time, os
sys.path.append('../')
import numpy as np
import gzip
import theano
from theano import config
theano.config.compute_test_value = 'warn'
from theano.printing import pydotprint
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from optvaeutils.optimizer import adam
from theano.compile.debugmode import DebugMode
from models import BaseModel
from utils.divergences import KLGaussian,BhattacharryaGaussian
from utils.misc import readPickle

class VAE(BaseModel, object):
    def __init__(self, params, paramFile=None, reloadFile=None, **kwargs):
        if params['data_type']=='image':
            assert params['emission_type']=='conv','Expecting convolutional emission'
        super(VAE,self).__init__(params, paramFile=paramFile, reloadFile=reloadFile, **kwargs)
        self.p_names = [k.name for k in self._getModelParams(restrict='p_')] 
    def _createParams(self):
        """
                    _createParams: create weights for model
        """
        npWeights = OrderedDict()
        DIM_HIDDEN = self.params['q_dim_hidden']

        #Parameters for the prior
        if self.params['prior']   =='loggamma': #p(z) = Log-Gamma(z;\alpha_fixed, \beta_fixed)
            npWeights['p_alpha']   = np.array([self.params['alpha']]*self.params['dim_stochastic']).astype(config.floatX)
        elif self.params['prior'] =='loggamma_learn': #p(z) = Log-Gamma(z;\alpha_learned, \beta_fixed)
            npWeights['p_alpha_W'] = self._getWeight((self.params['dim_stochastic'],))
        elif self.params['prior'] =='abs_gamma': #p(z) = Gamma(|z|;\alpha, \beta)
            npWeights['p_alpha']   = np.array([self.params['alpha']]*self.params['dim_stochastic']).astype(config.floatX)
        elif self.params['prior'] =='abs_gamma_learn': #p(z) = Gamma(|z|;\alpha, \beta)
            npWeights['p_alpha_W'] = self._getWeight((self.params['dim_stochastic'],))
        elif self.params['prior'] =='logit':     #p(z) = Softmax(Normal(O,I))
            pass
        elif self.params['prior'] =='logit_learn':#p(z) = Softmax(Normal(mu_learned,logcov_learned))
            npWeights['p_prior_mu_W']        = self._getWeight((self.params['dim_stochastic'],))
            npWeights['p_prior_logcov_W']    = self._getWeight((self.params['dim_stochastic'],))
        elif self.params['prior'] =='normal':
            pass
        elif self.params['prior'] =='normal_learn':
            npWeights['p_prior_mu_W']        = self._getWeight((self.params['dim_stochastic'],))
            npWeights['p_prior_logcov_W']    = self._getWeight((self.params['dim_stochastic'],))
        else:
            assert False,'Invalid prior specified'
        #Weights in recognition network model
        if self.params['emission_type'] in ['mlp','res']:
            for q_l in range(self.params['q_layers']):
                dim_input     = DIM_HIDDEN
                dim_output    = DIM_HIDDEN
                if q_l==0:
                    dim_input     = self.params['dim_observations']
                npWeights['q_'+str(q_l)+'_W'] = self._getWeight((dim_input, dim_output))
                npWeights['q_'+str(q_l)+'_b'] = self._getWeight((dim_output, ))
            if self.params['q_layers']==0:
                DIM_HIDDEN    = self.params['dim_observations']
        elif self.params['emission_type'] in ['conv']:    
            npWeights['q_filter_W_0'] = self._getWeight((64, 3, 5, 5)) 
            npWeights['q_filter_W_1'] = self._getWeight((128, 64, 5, 5))
            npWeights['q_2_W']        = self._getWeight((128*6*6,DIM_HIDDEN))
            npWeights['q_2_b']        = self._getWeight((DIM_HIDDEN))
        else:
            assert False,'Shouldnt be here'
        npWeights['q_mu_W']  = self._getWeight((DIM_HIDDEN, self.params['dim_stochastic']))
        npWeights['q_logcov_W'] = self._getWeight((DIM_HIDDEN, self.params['dim_stochastic']))
        npWeights['q_mu_b']  = self._getWeight((self.params['dim_stochastic'],))
        npWeights['q_logcov_b'] = self._getWeight((self.params['dim_stochastic'],))
        
        #Generative Model
        if self.params['emission_type'] in ['res','mlp']:
            for p_l in range(self.params['p_layers']):
                dim_input     = self.params['p_dim_hidden']
                dim_output    = dim_input
                if p_l==0:
                    dim_input     = self.params['dim_stochastic']
                npWeights['p_'+str(p_l)+'_W'] = self._getWeight((dim_input, dim_output))
                npWeights['p_'+str(p_l)+'_b'] = self._getWeight((dim_output, ))
            if self.params['emission_type']=='res':
                npWeights['p_linz_W'] = self._getWeight((self.params['dim_stochastic'],self.params['dim_observations']))
            DIM_HIDDEN     = self.params['p_dim_hidden']
            if self.params['p_layers']==0:
                DIM_HIDDEN = self.params['dim_stochastic']
            npWeights['p_mean_W']     = self._getWeight((DIM_HIDDEN, self.params['dim_observations']))
            npWeights['p_mean_b']     = self._getWeight((self.params['dim_observations'],))
        elif self.params['emission_type'] in ['conv']:
            npWeights['p_1_W']  = self._getWeight((self.params['dim_stochastic'],128*8*8)) 
            npWeights['p_1_b']  = self._getWeight((128*8*8,)) 
            #Convolutional filters
            npWeights['p_filter_W_2']   = self._getWeight((64,128,5,5))
            npWeights['p_filter_W_3']   = self._getWeight((3,64,5,5))
            #Final convolution filter to predict
            npWeights['p_filter_W_out'] = self._getWeight((256*3,3,1,1))
        else:
            assert False,'Shouldnt be here'
        return npWeights
    
    def _fakeData(self):
        """ Fake Data for Debugging """
        X = np.random.rand(2, self.params['dim_observations']).astype(config.floatX)
        m1=X>0.5
        m0=X<=0.5
        if self.params['data_type']=='bow':
            X[m0]=0
            X[m1]=400
        elif self.params['data_type']=='image':
            X = np.random.random_integers(0,255,(2,3,32,32)).astype(config.floatX)
        else:
            X[m0]=0
            X[m1]=1
        mu = np.random.randn(2, self.params['dim_stochastic']).astype(config.floatX)
        return X,mu
    
    def _KL(self, mu, logcov, z, keepmat=False):
        """  KL divergence between N(mu,logcov)||N(0,log I) """
        if self.params['prior']=='normal':
            KL      = 0.5*(-logcov-1+T.exp(logcov)+mu**2)
        elif self.params['prior'] in ['loggamma','loggamma_learn']:
            #KL     = -(N/2.)*np.log(2*np.pi) - N/2. - 0.5*logcov.sum() 
            KL     = -0.5* (T.log(2*np.pi) + 1. + logcov) 
            if 'p_alpha_W' in self.tWeights: #(ds,)
                alpha  = T.nnet.softplus(self.tWeights['p_alpha_W'])
                beta   = T.ones_like(alpha)
            else:
                #expecting positivity of shared variables
                alpha  = self.tWeights['p_alpha']
                beta   = T.ones_like(alpha)
                assert np.all(alpha.eval()>=0.),'Checking positivity'
            KL     = KL - alpha*T.log(beta) +T.gammaln(alpha)-alpha*mu +beta*T.exp(mu+T.exp(logcov)/2.) 
        elif self.params['prior'] in ['abs_gamma','abs_gamma_learn']:
            KL = -0.5* (T.log(2*np.pi) + 1. + logcov) 
            if 'p_alpha_W' in self.tWeights: #(ds,)
                alpha  = T.nnet.softplus(self.tWeights['p_alpha_W'])
                beta   = T.ones_like(alpha)
            else:
                assert np.all(self.tWeights['p_alpha'].eval()>=0.),'Checking positivity'
                alpha  = self.tWeights['p_alpha']
                beta   = T.ones_like(alpha)
            KL = KL- (alpha*T.log(beta) -T.gammaln(alpha) + (alpha-1)*T.log(abs(z)) -beta*abs(z)-T.log(2)) 
        elif self.params['prior'] in ['logit','logit_learn','normal_learn']:
            if 'p_prior_mu_W' in self.tWeights:
                mu_p       = self.tWeights['p_prior_mu_W']
                logcov_p   = self.tWeights['p_prior_logcov_W']
            else:
                mu_p       = T.zeros((self.params['dim_stochastic'],))
                logcov_p   = T.zeros((self.params['dim_stochastic'],))
            diff = (mu-mu_p)**2
            KL   = 0.5*(logcov_p-logcov -1 + T.exp(logcov-logcov_p) + diff*T.exp(-logcov_p)) 
        else:
            assert False,'Shouldnt be here. invalid prior model'
        if keepmat:
            return KL
        else:
            return KL.sum(1,keepdims=True)

    def logsoftmax(self, mat):
        if mat.ndim==1:
            maxval  = mat.max()
            result  = mat-(maxval+T.log(T.sum(T.exp(mat-maxval))+1e-6))
        else:
            maxval  = mat.max(1,keepdims=True)
            result  = mat-(maxval+T.log(T.sum(T.exp(mat-maxval),axis=1,keepdims=True)+1e-6))
        return result

    """
        My conv ops
    """
    def deconv(self,X, w, b=None):
        s = int(np.floor(w.get_value().shape[-1]/2.))
        img   = gpu_contiguous(X)
        kerns = gpu_contiguous(w)
        z = conv2d(img, kerns, border_mode='full')[:, :, s:-s, s:-s]
        if b is not None:
            z += b.dimshuffle('x', 0, 'x', 'x')
        return z
    def depool(self,X, factor=2):
        output_shape = [
            X.shape[1],
            X.shape[2]*factor,
            X.shape[3]*factor
        ]
        stride = X.shape[2]
        offset = X.shape[3]
        in_dim = stride * offset
        out_dim = in_dim * factor * factor
        upsamp_matrix = T.zeros((in_dim, out_dim))
        rows = T.arange(in_dim)
        cols = rows*factor + (rows//stride * factor * offset)
        upsamp_matrix = T.set_subtensor(upsamp_matrix[rows, cols], 1.)
        flat = T.reshape(X, (X.shape[0], output_shape[0], X.shape[2] * X.shape[3]))
        up_flat = T.dot(flat, upsamp_matrix)
        upsamp = T.reshape(up_flat, (X.shape[0], output_shape[0],
                                     output_shape[1], output_shape[2]))
        return upsamp
    def deconv_and_depool(self, X, w, b=None, activation=T.nnet.relu):
        return activation(self.deconv(self.depool(X), w, b))

    def conv_and_pool(self, X, W ):
        #Drops spatial dimensions by half
        res = T.nnet.abstract_conv.conv2d(X, W, border_mode='valid', subsample=(1, 1), filter_flip=True)
        return pool_2d(T.nnet.relu(res),(2,2),padding=(1,1), mode='average_inc_pad', ignore_border=True)

    def deconv2(self, hin, W_t): 
        inpshape= hin.shape
        W_shape = self.npWeights[W_t.name].shape
        outpshp = (None, W_shape[1], inpshape[-2]+1, inpshape[-1]+1)
        res = T.nnet.abstract_conv.conv2d_grad_wrt_inputs(hin, W_t, outpshp, 
                                                  filter_shape=None, border_mode='valid', 
                                                  subsample=(1, 1), filter_flip=True)
        return res
    def hard_tanh(self, x):
        return T.clip(x, -1., 1.)
    def _conditionalXgivenZ(self, z, validation = False, additional = {}):
        """
                                Mean Probabilities under p_theta(x|z) 
        """
        assert self.params['data_type'] in ['real','binary','bow','image'],'Only binary data/bow'
        if self.params['prior'] in ['loggamma','loggamma_learn']:
            inp_p   = T.exp(z)
        elif self.params['prior'] in ['abs_gamma','abs_gamma_learn']:
            inp_p   = abs(z)
        elif self.params['prior'] in ['logit','logit_learn']:
            inp_p   = T.nnet.softmax(z)
        elif self.params['prior'] in ['normal','normal_learn']:
            inp_p   = z
        else:
            assert False,'Invalid prior'
        if self.params['emission_type'] in ['res','mlp']:
            for p_l in range(self.params['p_layers']):
                inp_p= self._LinearNL(self.tWeights['p_'+str(p_l)+'_W'], self.tWeights['p_'+str(p_l)+'_b'], inp_p)
            paramMat = T.dot(inp_p,self.tWeights['p_mean_W'])
            if self.params['emission_type']=='res':
                paramMat += T.dot(z,self.tWeights['p_linz_W'])
            #Data types
            if self.params['data_type']=='binary':
                E = paramMat+self.tWeights['p_mean_b']
                additional['E'] = E
                mean_p        = T.nnet.sigmoid(E)
                return mean_p
            elif self.params['data_type'] == 'bow':
                E = paramMat+self.tWeights['p_mean_b']
                additional['E'] = E
                if self.params['likelihood']=='mult':
                    emb    = self.logsoftmax(E)
                    return emb
                elif self.params['likelihood']=='poisson':
                    #Positive real valued parameter
                    loglambda_p=E
                    return loglambda_p
                else:
                    raise ValueError,'Invalid setting of likelihood: '+self.params['likelihood']
            elif self.params['data_type']=='real':
                E = paramMat+self.tWeights['p_mean_b']
                additional['E'] = E
                return (E,T.zeros_like(E))
            else:
                assert False,'shouldnt be here'
        elif self.params['emission_type'] in ['conv']:
            assert False,'not implemented'
            assert self.params['data_type'] == 'image','expecting image'
            #Convolutional decoder
            h2 = T.nnet.relu(T.dot(inp_p, self.tWeights['p_1_W']) + self.tWeights['p_1_b'])
            h2 = h2.reshape((h2.shape[0], 128,8,8))
            h3 = self.deconv_and_depool(h2, self.tWeights['p_filter_W_2'])
            h4 = self.deconv_and_depool(h3, self.tWeights['p_filter_W_3'])
            #h3 = T.nnet.relu(self.deconv(h2, self.tWeights['p_filter_W_2']))
            #h4 = T.nnet.relu(self.deconv(h3, self.tWeights['p_filter_W_3']))
            outp  = self.hard_tanh(conv2d(h4, self.tWeights['p_filter_W_out']))
            outp = outp.reshape((outp.shape[0],256,3*32*32))
            emb= self.logsoftmax(outp)
            return emb
        else:
            assert False,'Shouldnt be here'
    
    def _negCLL(self, z, X, validation = False):
        """Estimate -log p[x|z]"""
        if self.params['data_type']=='binary':
            p_x_z    = self._conditionalXgivenZ(z)
            negCLL_m = T.nnet.binary_crossentropy(p_x_z,X)
        elif self.params['data_type'] =='bow':
            #Likelihood under a multinomial distribution
            if self.params['likelihood'] == 'mult':
                lsf      = self._conditionalXgivenZ(z)
                p_x_z    = T.exp(lsf) 
                negCLL_m = -1*(X*lsf)
            elif self.params['likelihood'] =='poisson':
                loglambda_p = self._conditionalXgivenZ(z)
                p_x_z       = T.exp(loglambda_p)
                negCLL_m    = -X*loglambda_p+T.exp(loglambda_p)+T.gammaln(X+1)
            else:
                raise ValueError,'Invalid choice for likelihood: '+self.params['likelihood']
        elif self.params['data_type']=='real':
            params   = self._conditionalXgivenZ(z)
            mu,logvar= params[0], params[1]
            p_x_z    = mu  
            negCLL_m = 0.5 * np.log(2 * np.pi) + 0.5*logvar + 0.5 * ((X - mu_p)**2)/T.exp(logvar)
        elif self.params['data_type']=='image':
            emb      = self._conditionalXgivenZ(z)
            p_x_z    = T.argmax(T.exp(emb),1).reshape((emb.shape[0],3,32,32))
            #emb should be bs x 256 x (3x32x32)
            idx_t    = X.ravel()
            shuf     = emb.dimshuffle(0,2,1).reshape((-1,256))
            negCLL_m = -1*shuf[T.arange(shuf.shape[0]), T.cast(idx_t,'int64')].reshape((emb.shape[0],emb.shape[2]))
        else:
            assert False,'Shouldnt be here'
        return p_x_z, negCLL_m.sum(1,keepdims=True)
    
    def _inference(self, X, dropout_prob = 0.):
        """
                          Build subgraph to do inference 
        """
        self._p(('Inference with dropout :%.4f')%(dropout_prob))
        if self.params['data_type']=='bow':
            #inp = X/float(self.params['max_word_count']) 
            if self.params['input_type']=='counts':
                #inp  = X #Counts
                inp = X/float(self.params['max_word_count']) 
            elif self.params['input_type']=='normalize':
                #Normalized
                inp  = (X/X.sum(1,keepdims=True))#*float(self.params['max_word_count'])#T.cast(X.shape[1],'float32')
            elif self.params['input_type']=='tfidf':
                #TF-IDF
                assert self.idf is not None,'Expecting tfidf vectors'
                tfidf= (X*self.idf)
                inp  = tfidf/T.sqrt((tfidf**2).sum(1,keepdims=True)) 
            else:
                assert False,'Invalid input type'
        elif self.params['data_type']=='image':
            inp  = X/X.max()
        else:
            inp = self._dropout(X,dropout_prob)

        if self.params['emission_type'] in ['mlp','res']:
            for q_l in range(self.params['q_layers']):
                inp = self._LinearNL(self.tWeights['q_'+str(q_l)+'_W'], self.tWeights['q_'+str(q_l)+'_b'], inp)
        elif self.params['emission_type'] in ['conv']:
            #Convolutional Encoder
            h1  = self.conv_and_pool(inp, self.tWeights['q_filter_W_0'])
            h2  = self.conv_and_pool(h1, self.tWeights['q_filter_W_1'])
            h2  = h2.reshape((h2.shape[0], -1))
            inp = T.nnet.relu(T.dot(h2,self.tWeights['q_2_W']) + self.tWeights['q_2_b'])
        else:
            assert False,'invalid setup'
        mu      = T.dot(inp,self.tWeights['q_mu_W'])    +self.tWeights['q_mu_b']
        logcov  = T.dot(inp,self.tWeights['q_logcov_W'])+self.tWeights['q_logcov_b']
        return mu, logcov
    
    ################################    Building Objective Functions #####################
    def _ELBO(self, X, eps = None, mu_q=None, logcov_q=None, anneal = 1., 
              dropout_prob = 0., savedict = None, batch_vec = False):
        """
                          Wrapper for ELBO
        """
        if mu_q is None and logcov_q is None:
            mu_q, logcov_q = self._inference(X, dropout_prob)
        if eps is None:
            #use in scan operator we need to take eps as arg
            eps = self.srng.normal(size=mu_q.shape,dtype=config.floatX)
        z = mu_q+eps*T.exp(logcov_q*0.5)
        mean_p, negCLL = self._negCLL(z, X, validation=dropout_prob==0.)
        KL             = self._KL(mu_q, logcov_q, z, keepmat=True)
        #Collect statistics
        if type(savedict) is dict:
            savedict['z']        = z
            savedict['mean_p']   = mean_p
            savedict['mu_q']     = mu_q
            savedict['logcov_q'] = logcov_q
            savedict['negCLL'] = negCLL
            savedict['KLmat']    = KL
            savedict['KL']    = KL
            savedict['elbo_batch']=negCLL+KL.sum(1,keepdims=True)
        bound = (negCLL+ anneal*KL.sum(1,keepdims=True)).sum() 
        return bound
    
    def adamUpdates(self, it_k, paramlist, gradlist, m_list, v_list, plr):
        m_new_list, v_new_list, param_new_list = [], [], []
        b1,b2  = 0.9, 0.001
        fix1   = 1. - (1. - b1)**it_k
        fix2   = 1. - (1. - b2)**it_k
        lr_t   = plr * (T.sqrt(fix2) / fix1)
        for p,g,m,v in zip(paramlist, gradlist, m_list, v_list):
            m_t = (b1 * g) + ((1. - b1) * m)
            v_t = (b2 * T.sqr(g)) + ((1. - b2) * v)
            g_t = m_t / (T.sqrt(v_t) + 1e-8)
            p_t = p - (lr_t * g_t)
            m_new_list.append(m_t)
            v_new_list.append(v_t)
            param_new_list.append(p_t)
        return param_new_list, m_new_list, v_new_list
    def _optimizeVariationalParams(self, X, mu0, logcov0, n_steps, plr, savedict = {}, force_resample = False):
        """  Wrapper to optimize variational parameters
                        mu0,logcov0 - initial variational parameters
                        n_steps     - number of steps to perform optimization
                        plr         - learning rate for variational parameters
        """
        epslist = self.srng.normal(size = (n_steps,X.shape[0],self.params['dim_stochastic']),dtype=config.floatX)
        assert self.params['opt_method']=='adam','expecting adam'
        def optAdam(it_k, eps_k, mu, logcov, m_mu, v_mu, m_logcov, v_logcov, X, plr):
            if self.params['anneal_finopt_rate']>0:
                anneal   = T.switch(T.gt(it_k/float(self.params['anneal_finopt_rate']),1.), 
                        1., it_k/float(self.params['anneal_finopt_rate']))
            else:
                anneal   = 1.
            L_prev       = self._ELBO(X, eps = eps_k, mu_q = mu, logcov_q = logcov, anneal = anneal)
            gradlist     = T.grad(L_prev,wrt = [mu,logcov])
            paramlist, mlist, vlist = self.adamUpdates(it_k, [mu,logcov], gradlist, [m_mu, m_logcov], [v_mu, v_logcov], plr)

            L_out        = L_prev
            if self.params['anneal_finopt_rate']>0:
                L_out    = self._ELBO(X, eps = eps_k, mu_q = mu, logcov_q = logcov)
            mu_k, logcov_k     = paramlist[0], paramlist[1]
            m_mu_k, v_mu_k         = mlist[0], vlist[0]
            m_logcov_k, v_logcov_k = mlist[1], vlist[1]
            return mu_k, logcov_k, L_out, T.sqrt((gradlist[0]**2).sum(1)).sum(), T.sqrt((gradlist[1]**2).sum(1)).sum(), m_mu_k, v_mu_k, m_logcov_k, v_logcov_k
        self._p('Optimizing variational parameters w/ ADAM')
        m_mu_0     = T.zeros((X.shape[0], self.params['dim_stochastic']))
        v_mu_0     = T.zeros((X.shape[0], self.params['dim_stochastic']))
        m_logcov_0 = T.zeros((X.shape[0], self.params['dim_stochastic']))
        v_logcov_0 = T.zeros((X.shape[0], self.params['dim_stochastic']))
        result,updates = theano.scan(optAdam, 
                                 sequences = [T.arange(1,n_steps+1,dtype=config.floatX), epslist],
                                 outputs_info=[mu0,logcov0,None,None,None,m_mu_0,v_mu_0,m_logcov_0,v_logcov_0],
                                 non_sequences = [X, plr],
                                 n_steps = n_steps)
        mu_its, logcov_its, elbo_its, grad_mu_its, grad_logcov_its = result[0],result[1],result[2],result[3],result[4]
        savedict['mu_its']          = mu_its
        savedict['logcov_its']      = logcov_its
        savedict['elbo_its']        = elbo_its
        savedict['gradnorm_mu_its']     = grad_mu_its
        savedict['gradnorm_logcov_its'] = grad_logcov_its
        savedict['n_steps'] = mu_its.shape[0]
        return mu_its[-1], logcov_its[-1], elbo_its[-1]
    
    def _buildOptimizationFunction(self, X, n_steps, plr):
        mu_0,logcov_0 = self._inference(X)
        optdict = {}
        _, logcov_f, elbo_final = self._optimizeVariationalParams(X, mu_0, logcov_0, n_steps, plr,
                                                                              savedict = optdict)
        diff_elbo, diff_entropy = self._estimateELBOEntropy(optdict['elbo_its'][0],optdict['elbo_its'][-1], logcov_0, logcov_f)
        self.optimize_mu_logcov = theano.function([X, theano.In(n_steps, value=self.params['n_steps'], name='n_steps'),
                                                 theano.In(plr, value=self.params['param_lr'], name='plr')],
                                                   [optdict['elbo_its'], optdict['gradnorm_mu_its'],
                                                    optdict['gradnorm_logcov_its'],optdict['elbo_its'].shape[0], diff_elbo, diff_entropy], 
                                                 name = 'Optimize ELBO wrt mu/cov')
        diff_elbo, diff_ent = self._estimateELBOEntropy(optdict['elbo_its'][0], optdict['elbo_its'][-1], logcov_0, logcov_f)
        self.final_elbo     = theano.function([X, theano.In(n_steps, value=self.params['n_steps'], name='n_steps'),
                                                 theano.In(plr, value=self.params['param_lr'], name='plr')],
                                               [optdict['elbo_its'][0],optdict['elbo_its'][-1], optdict['elbo_its'].shape[0],
                                               optdict['gradnorm_mu_its'][-1],optdict['gradnorm_logcov_its'][-1], 
                                               diff_elbo, diff_ent], name = 'Optimize ELBO wrt mu/cov')
    def _estimateELBOEntropy(self, elbo_0, elbo_f, logcov_0, logcov_f):
        """ H = 0.5*log |Sigma| + K/2 + K/2 log (2\pi) 
            if the ratio is high, most of the
            We know that elbo_0-elbo_f>0 so the sign of the ratio is given by whether or not there was a contraction
            while the magnitude gives you how much of the change can be attributed to the change in entropy of
            the variational distribution"""
        diff_elbo      = elbo_0-elbo_f #Since we're using upper bounds multiply by -1
        diff_entropy   = 0.5*(logcov_f-logcov_0).sum() 
        return diff_elbo, diff_entropy 

    ##################                EVALUATION        ######################
    def _buildEvaluationFunctions(self, X,n_steps,plr):
        """
            Build functions for evaluation. X: input,evaluation_bound: bound for evaluation
            evaldict: dictionary containing z/mu/logcov and other arrays that might need inspection
            additional_inputs: used to support finopt where you need to have n_steps etc
        """
        self._p('Evaluation: Setting opt_method: ADAM, 100 steps w/ 8e-3 lr')
        evaldict0, evaldictopt, evaldictf     = {}, {}, {}
        elbo_init         = self._ELBO(X,   savedict = evaldict0)

        elbo_init_batch   = evaldict0['elbo_batch'] 

        mu_f, logcov_f, _ = self._optimizeVariationalParams(X,evaldict0['mu_q'],evaldict0['logcov_q'],
                                                            n_steps, plr, savedict = evaldictopt)
        elbo_final        = self._ELBO(X, mu_q = mu_f, logcov_q = logcov_f, savedict = evaldictf)
        elbo_final_batch  = evaldictf['elbo_batch'] 

        fxn_inputs = [X]
        init_val = 100
        if self.params['data_type']=='image':
            init_val = 5
        fxn_inputs.append(theano.In(n_steps, value = init_val, name = 'n_steps'))
        fxn_inputs.append(theano.In(plr, value = 8e-3, name = 'plr'))
        diff_elbo, diff_ent = self._estimateELBOEntropy(elbo_init, elbo_final, evaldict0['logcov_q'], evaldictf['logcov_q'])
        self.evaluate   = theano.function(fxn_inputs, [elbo_init, elbo_final,evaldictopt['n_steps'], diff_elbo, diff_ent], name = 'Evaluate')
        self.reconstruct= theano.function([evaldictf['z']], evaldictf['mean_p'], name='Reconstruct')
        self.inference  = theano.function(fxn_inputs, [evaldictf['z'], evaldictf['mu_q'], evaldictf['logcov_q'] ], 
                                          name = 'Posterior Inference')
        self.inference0 = theano.function([X], [evaldict0['z'], evaldict0['mu_q'], evaldict0['logcov_q'] ,evaldict0['KL']], 
                                          name = 'Posterior Inference 0 ')
        self.inferencef = theano.function(fxn_inputs, [evaldictf['z'], 
                                                       evaldictf['mu_q'], evaldictf['logcov_q'] ,evaldictf['KL']], 
                                          name = 'Posterior Inference F ')
        z0       = T.vector('z')
        z0.tag.test_value = np.random.randn(self.params['dim_stochastic']).astype(config.floatX)
        #m0,_        = self._negCLL(z0, X) #This computes Jacobian wrt probabilities
        additional   = {}
        lsf          = self._conditionalXgivenZ(z0,additional=additional) #This computes Jacobian wrt log-probabilities, For poisson models this is the logmean
        if self.params['data_type']=='real':
            lsf = lsf[0]
        #Grad wrt energy
        jacob_energy   = theano.gradient.jacobian(additional['E'],wrt=z0)
        jacob_logprobs = theano.gradient.jacobian(lsf,wrt=z0)
        jacob_probs    = theano.gradient.jacobian(T.exp(lsf),wrt=z0)
        jacob_logprobs_mnist = theano.gradient.jacobian(T.log(lsf),wrt=z0) #For use w/ binarized mnist only

        self.jacobian_logprobs = theano.function([z0],jacob_logprobs,name='Jacobian wrt Log-Probs')   
        self.jacobian_probs    = theano.function([z0],jacob_probs,name='Jacobian')   
        self.jacobian_energy   = theano.function([z0],jacob_energy,name='Jacobian wrt energy')   

        if self.params['data_type']=='bow':
            X_count     = X.sum(1,keepdims=True)
            self.evaluatePerp = theano.function(fxn_inputs, [(elbo_init_batch/X_count).sum(), 
                (elbo_final_batch/X_count).sum(), evaldictopt['n_steps'], diff_elbo, diff_ent])
        elif self.params['data_type']=='image':
            #X: bs x 3 x 32 x 32
            prod_of_dim = T.cast(X.shape[1:].prod(),config.floatX)*T.log(2.)
            self.evaluateBitsDim = theano.function(fxn_inputs, [elbo_init_batch.sum()/prod_of_dim, 
                elbo_final_batch.sum()/prod_of_dim, evaldictopt['n_steps'], diff_elbo, diff_ent])
        self.debugModel  = theano.function([X], [evaldict0['elbo_batch'].sum(), evaldict0['negCLL'].sum(),evaldict0['KLmat'].sum()])

    ################################    Building Model #####################
    def _buildModel(self):
        """
                                       ******BUILD VAE GRAPH******
        """
        #Update optimizer
        self.optimizer = adam
        
        if self.params['data_type']=='image':
            X   = T.tensor4('X',   dtype=config.floatX)
        else:
            X   = T.matrix('X',   dtype=config.floatX)
        X.tag.test_value, mu_tag  = self._fakeData()
                                       
        #Learning rates and annealing objective function
        self._addWeights('iter_ctr', np.asarray(1.))
        iteration_t    = self.tWeights['iter_ctr']
        self._addWeights('anneal', np.asarray(0.,dtype=config.floatX),borrow=False)
        anneal         = self.tWeights['anneal']
        anneal_rate    = self.params['anneal_rate']+1e-5
        anneal_update  = [(iteration_t, iteration_t+1),
                      (anneal, 
                       T.switch(0.01+iteration_t/anneal_rate>1,1.,0.01+iteration_t/anneal_rate))]
        
        #Used in optimizer.py
        self._addWeights('lr',   np.asarray(self.params['lr'],dtype=config.floatX),  borrow=False)
        lr             = self.tWeights['lr']
        
        n_steps, plr            = T.iscalar('n_steps'), T.fscalar('plr')
        n_steps.tag.test_value, plr.tag.test_value = self.params['n_steps'],self.params['param_lr']
        
        self._buildOptimizationFunction(X, n_steps, plr)
        self._buildEvaluationFunctions(X,n_steps,plr)
        if 'EVALUATION' in self.params:
            self.updates_ack = True
            self.tOptWeights = []
            self._p('Not building training functions...')
            return
        if self.params['opt_type']=='none':
            """
            none: simple vae optimization.
            """
            traindict = {}
            upperbound_train         = self._ELBO(X, anneal = anneal, 
                                        dropout_prob = self.params['input_dropout'], savedict=traindict)
            self.updates_ack = True
            if 'GRADONLY' in self.params:
                self._p('Only computing gradients with respect to cost function')
                p_params                 = self._getModelParams('p_')
                p_grads                  = T.grad(upperbound_train, p_params)
                self.getgrads            = theano.function([X],[pp*1. for pp in p_grads])
                self.tOptWeights = []
                return 
            model_params             = self._getModelParams()
            optimizer_up, norm_list  = self._setupOptimizer(upperbound_train,  model_params,
                                                        lr = lr,  
                                                        grad_noise = self.params['grad_noise'],
                                                        rng = self.srng)#,
                                                        #reg_type =self.params['reg_type'], 
                                                        #reg_spec =self.params['reg_spec'], 
                                                        #reg_value= self.params['reg_value'],
                                                        #grad_norm = 1.,
                                                        #divide_grad = T.cast(X.shape[0],config.floatX))
            self._p('# additional updates: '+str(len(self.updates)))
            optimizer_up+=anneal_update +self.updates
            fxn_inputs      = [X]
            self.train      = theano.function(fxn_inputs, [upperbound_train, norm_list[0], norm_list[1], norm_list[2],
                                                           anneal.sum(), lr.sum()],
                                              updates = optimizer_up, name = 'Train')
        elif self.params['opt_type'] in ['miao','miao_ascent']:
            dictf              = {}
            elbo               = self._ELBO(X, dropout_prob = self.params['input_dropout'], anneal=anneal, savedict=dictf)
            p_params                 = self._getModelParams(restrict='p_')
            optimizer_p, norm_list_p = self._setupOptimizer(elbo, p_params,
                                                        lr = lr, grad_noise = self.params['grad_noise'], rng = self.srng)
            self._p('# additional updates: '+str(len(self.updates)))
            self.updates_ack     = True
            optimizer_p         += self.updates+anneal_update
            self.update_p        = theano.function([X],
                                              [elbo, norm_list_p[0], norm_list_p[1], norm_list_p[2],
                                               anneal.sum(), lr.sum()], 
                                              updates = optimizer_p, name = 'Train P')
            q_params                 = self._getModelParams(restrict='q_')
            optimizer_q, norm_list_q = self._setupOptimizer(elbo, q_params,
                                                            lr = lr, grad_noise = self.params['grad_noise'], rng = self.srng)
            self.update_q   = theano.function([X], elbo, updates = optimizer_q, name = 'Train Q')
        elif self.params['opt_type'] in ['finopt','finopt_mult']:
            ##################                UPDATE  P         ######################
            dict0,dictopt,dictf = {},{},{}
            mu_q_0, logcov_q_0 = self._inference(X)
            mu_q_f, logcov_q_f, _ = self._optimizeVariationalParams(X, mu_q_0, logcov_q_0, 
                                                                          n_steps, plr, savedict=dictopt)
            elbo_init      = self._ELBO(X, mu_q=mu_q_0, logcov_q=logcov_q_0, anneal=anneal)
            mu_f_dgrad     = theano.gradient.disconnected_grad(mu_q_f)
            logcov_f_dgrad = theano.gradient.disconnected_grad(logcov_q_f)
            elbo_final     = self._ELBO(X, mu_q=mu_f_dgrad, logcov_q=logcov_f_dgrad, savedict = dictf, anneal=anneal)
            
            p_params                 = self._getModelParams(restrict='p_')
            self.updates_ack = True
            if 'GRADONLY' in self.params:
                self._p('Only computing gradients with respect to cost function')
                p_grads                  = T.grad(elbo_final, p_params)
                self.getgrads            = theano.function([X,
                                                            theano.In(n_steps, value=self.params['n_steps'], name='n_steps'),
                                                            theano.In(plr, value=self.params['param_lr'], name='plr')],
                                                            [pp*1. for pp in p_grads])
                self.tOptWeights = []
                return 
            optimizer_p, norm_list_p = self._setupOptimizer(elbo_final, p_params, lr = lr, 
                                                        grad_noise = self.params['grad_noise'],
                                                        rng = self.srng)#,
            self._p('# additional updates: '+str(len(self.updates)))
            optimizer_p += self.updates+anneal_update
            diff_elbo, diff_ent = self._estimateELBOEntropy(elbo_init, elbo_final, logcov_q_0, logcov_q_f)
            self.update_p   = theano.function([X,theano.In(n_steps, value=self.params['n_steps'], name='n_steps'),
                                                  theano.In(plr, value=self.params['param_lr'], name='plr')], 
                                              [elbo_init, elbo_final, anneal.sum(), 
                                               norm_list_p[0], norm_list_p[1], norm_list_p[2], 
                                               dictopt['n_steps'], dictopt['gradnorm_mu_its'][-1], 
                                               dictopt['gradnorm_logcov_its'][-1],diff_elbo, diff_ent],#+gdiffs, 
                                              updates = optimizer_p, name = 'Train P')
            
            ##################                UPDATE Q           ######################
            q_params                 = self._getModelParams(restrict='q_')
            if self.params['q_strategy']=='standard':
                elbo                     = self._ELBO(X,  dropout_prob = self.params['input_dropout'], anneal = anneal)
                optimizer_q, norm_list_q = self._setupOptimizer(elbo, q_params,
                                                            lr = lr, 
                                                        grad_noise = self.params['grad_noise'],
                                                        rng = self.srng)
                self.update_q   = theano.function([X], elbo,
                                                  updates = optimizer_q, name = 'Train Q')
            elif self.params['q_strategy']=='bhatt':
                cost_bhatt = BhattacharryaGaussian(mu_q_0, logcov_q_0, mu_f_dgrad, logcov_f_dgrad, logCov = True)
                updates_q_bhatt, _ = self._setupOptimizer(cost_bhatt, q_params,
                                                            lr = lr,  
                                                        grad_noise = self.params['grad_noise'],
                                                        rng = self.srng)
                self.update_q_bhatt = theano.function([X,theano.In(n_steps, value=self.params['n_steps'], name='n_steps'),theano.In(plr, value=self.params['param_lr'], name='plr')], cost_bhatt,
                                                      updates = updates_q_bhatt, name = 'Train Q Bhatt')
            elif self.params['q_strategy']=='kl_f_0':
                cost_kl_f_0 = KLGaussian(mu_f_dgrad, logcov_f_dgrad, mu_q_0, logcov_q_0, logCov = True)
                updates_q_f_0, _ = self._setupOptimizer(cost_kl_f_0, q_params,
                                                        lr = lr,  
                                                        grad_noise = self.params['grad_noise'],
                                                        rng = self.srng)
                self.update_q_kl_f_0= theano.function([X,theano.In(n_steps, value=self.params['n_steps'], name='n_steps'),theano.In(plr, value=self.params['param_lr'], name='plr')], cost_kl_f_0,
                                                      updates = updates_q_f_0, name = 'Train Q f-0')
            elif self.params['q_strategy']=='kl_0_f':
                cost_kl_0_f= KLGaussian(mu_q_0, logcov_q_0, mu_f_dgrad, logcov_f_dgrad, logCov = True)
                updates_q_0_f, _ = self._setupOptimizer(cost_kl_0_f, q_params,
                                                        lr = lr,  
                                                        grad_noise = self.params['grad_noise'],
                                                        rng = self.srng)
                self.update_q_kl_0_f= theano.function([X,theano.In(n_steps, value=self.params['n_steps'], name='n_steps'),theano.In(plr, value=self.params['param_lr'], name='plr')], cost_kl_0_f,
                                                      updates = updates_q_0_f, name = 'Train Q 0-f')
            elif self.params['q_strategy']=='symmetric_kl':
                cost_symkl = KLGaussian(mu_q_0, logcov_q_0, mu_f_dgrad, logcov_f_dgrad, logCov = True)+KLGaussian(mu_f_dgrad, logcov_f_dgrad,mu_q_0, logcov_q_0, logCov = True)
                updates_q_symkl, _ = self._setupOptimizer(cost_symkl, q_params,
                                                        lr = lr, 
                                                        grad_noise = self.params['grad_noise'],
                                                        rng = self.srng)#,
                self.update_q_symkl = theano.function([X,theano.In(n_steps, value=self.params['n_steps'], 
                    name='n_steps'),theano.In(plr, value=self.params['param_lr'], name='plr')], cost_symkl,
                                                      updates = updates_q_symkl, name = 'Train Q SKL')
            else:
                assert False,'Invalid q_strategy'
        else:
            assert False,'Invalid optimization type: '+self.params['opt_type']
        self._p('Done creating functions for training')
    
if __name__=='__main__':
    print 'Initializing VAE'
    pfile = './tmp'
    from optvaeutils.parse_args_vae import params
    params['dim_observations'] =2000
    params['data_type']   = 'bow'
    params['opt_type']    = 'none'
    params['opt_method']  = 'adam'
    params['anneal_finopt_rate'] = 100
    params['GRADONLY']=True
    vae   = VAE(params, paramFile=pfile)
    from datasets.load import loadDataset
    dataset = loadDataset('binarized_mnist')
    np.random.seed(1)
    idxlist = np.random.permutation(dataset['train'].shape[0])
    X = dataset['train'][idxlist[:200]].astype('float32')
    os.remove(pfile)
    import ipdb;ipdb.set_trace()
