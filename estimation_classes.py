import numpy as np
import keras
from keras import optimizers
from keras.backend import clear_session
from numpy.linalg import inv
from scipy.stats import norm
from scipy.optimize import bisect
# from bisect import bisect

class MLP():
    def __init__(self):
        pass
    def train(self, X, Y, w=[100], verbose=0,
              activations = None,
              l2_penalty = 0.001):
        """
        train(self, X, Y, d=3, w=[100], verbose=0)
        
        Creates and trains a MLP based on regressors X and targets Y.
        
        Parameters
        ----------
        X : TYPE
            DESCRIPTION.
        Y : TYPE
            DESCRIPTION.
        d : TYPE, optional
            DESCRIPTION. The default is 3.
        w : TYPE, optional
            DESCRIPTION. The default is [100].
        verbose : TYPE, optional
            DESCRIPTION. The default is 0.

        Returns
        -------
        None.

        """
        # Store data as attributes
        self.X = X
        self.Y = Y
        
        # check dimension
        d = len(w) + 2
        
        # numer of observations and width vector
        n, w1 = X.shape
        wd = Y.shape[1]
        self.w = [w1] + [w[i] for i in range(len(w))] + [wd]
        
        # initialize network
        self.MLP = keras.models.Sequential()
        
        # Standardize data
        self.X_mean = np.mean(X, 0)
        self.X_std = np.std(X, 0)
        X = (X - self.X_mean) / self.X_std
        self.Y_mean = np.mean(Y, 0)
        self.Y_std = np.std(Y, 0)
        Y = (Y - self.Y_mean) / self.Y_std
        
        
        # Model architecture
        self.l2_penalty = l2_penalty # weight decay parameter
        for layer, width in enumerate(self.w):
            if layer == 0:
                continue
            elif (layer == 1) & (layer != d-1):
                self.MLP.add(keras.layers.Dense(width, 
                                                activation = 'tanh',
                                                input_dim = self.w[0],
                                                kernel_regularizer = keras.regularizers.l2(self.l2_penalty)))
            elif (layer == 1) & (layer == d-1):
                self.MLP.add(keras.layers.Dense(width,
                                                input_dim = self.w[0]))
            elif layer != d-1:
                self.MLP.add(keras.layers.Dense(width, 
                                                activation = 'tanh',
                                                kernel_regularizer = keras.regularizers.l2(self.l2_penalty)))
            elif layer == d-1:
                self.MLP.add(keras.layers.Dense(width))
        
        # Compilation and training via Adam
        self.MLP.compile(optimizer = keras.optimizers.Adam(lr=0.015, 
                                                           beta_1=0.9, 
                                                           beta_2=0.999, 
                                                           epsilon=1E-8,
                                                           amsgrad=False),
                         loss = 'mse')
        self.MLP.fit(X, Y,
                     epochs = 500,
                     batch_size = 256,
                     # validation_split = 0.1,
                     # callbacks = [keras.callbacks.EarlyStopping(patience = 100,
                     #                                            restore_best_weights = True)],
                     verbose = verbose)
        
        # Compilation and training via full-batch SGD to be at minimum of loss
        # landscape
        self.MLP.compile(optimizer=optimizers.SGD(lr=0.005),
                         loss = 'mse')
        self.MLP.fit(X, Y,
                     epochs=500,
                     batch_size=n,
                     validation_split = 0.01,
                     verbose = verbose)
        # delete existing estimates of d2L and dLdL from previous trainings
        try:
            del self.d2L, self.dLdL, self.V_theta
        except:
            pass
    
    def compute_ME(self, X0):
        """
        compute_ME(model, x)
        
        Compute the Marginal Effects of a Multilayer Perceptron.
        
        Parameters
        ----------
        x0: (m, k) array
            Evaluation points for the marginal effects
            k - input dimension
            m - number of evaluation points
                
        Returns
        -------
        ME: (m,kp,k) array
            Marginal effects for each evaluation point
            kp - output dimension
            k - input dimension
            m - number of evaluation points
        """
        # rescale evaluation points
        X0 = (X0 - self.X_mean) / self.X_std
        # input dimension and number of evaluation points
        m, k = X0.shape
        
        # Get list of weight matrices and bias terms
        model = self.MLP
        Wb = model.get_weights()
        W = [x.transpose() for i,x in enumerate(Wb[0::2])]
        b = [x.reshape((-1,1)) for i,x in enumerate(Wb[1::2])]
        
        # number of layers
        l = len(b) + 1
        
        # output dimension
        kp = W[-1].shape[0]
        
        # Get list of activation functions
        activation = [None] + [x['config']['activation'] for x in model.get_config()['layers'][1:]]
        
        # Compute the marginal effects. We iterate over all evaluation points to 
        # save memory. Then we iterate over the layers to compute all information.
        ME = np.zeros((m,kp,k))
        for i in range(m):
            # Create list of layer elements
            s = [None] * l # non-activated weighted information of last layer
            z = [None] * l # activated layer-output
            dsig = [None] * l # derivative of activation function
            
            # We initiate the marginal effect as identity matrix with the same
            # dimension as the input. It will be iteratively pre-multiplied with
            # the weight matrices and derivatives of the activation functions.
            me = np.eye(k)
            
            # output of first layer is input x
            z[0] = X0[i, :].reshape((-1,1)) 
            for j in range(l-1):
                # forward propagation
                s[j+1] = W[j] @ z[j] + b[j] 
                # activation
                z[j+1], dsig[j+1] = self.__activate__(s[j+1], activation[j+1], return_derivatives = 1)
                # Marginal effect chain
                me = dsig[j+1] @ W[j] @ me
            ME[i,:,:] = me
        
        # rescale the ME
        ME = ME * self.Y_std.reshape((-1,1)) / self.X_std
        
        return ME
    
    def compute_ME_std(self, X0, verbose=0):
        """
        compute_ME(model, x)
        
        Compute the standard deviations for the Marginal Effects of a 
        Multilayer Perceptron.
        
        Parameters
        ----------
        X0: (m, k) array
            Evaluation points for the marginal effects
            k - input dimension
            m - number of evaluation points
                
        Returns
        -------
        ME_std: (m,kp,k) array
                Standard deviations of marginal effects for each evaluation 
                point
                kp - output dimension
                k - input dimension
                m - number of evaluation points
        """
        
        # obtain the matrices to construct the asymptotic covariance matrix 
        # of the marginal effects
        dM = self.__compute_dM__(X0, verbose) # partial derivative of MEs w.r.t. weights
        try:
            V_theta = self.V_theta
        except:
            V_theta = self.__compute_V_theta__(verbose)
        
        # obtain number of evaluation points, MEs and weights
        m, num_ME, num_w = dM.shape # number of evaluation points, MEs and weights
        n, k = self.X.shape
        kp = self.Y.shape[1]
        
        # for each evaluation point, compute the standard deviations of the
        # marginal effects
        ME_std = np.zeros((m, kp, k))
        for i in range(m): # loop over all evaluation points
            ME_var_vectorized = np.zeros(num_ME)
            for j in range(num_ME): # loop over all marginal effects
                ME_var_vectorized[j] = dM[i,j,:].reshape((1,-1)) @ V_theta @ dM[i,j,:].reshape((-1,1))
            ME_std[i,:,:] = np.sqrt(ME_var_vectorized.reshape((kp,k))/n)
        
        return ME_std * self.Y_std.reshape((-1,1)) / self.X_std
    
    def compute_ME_CI_boot(self, X, Y, X0, w=[100], B=3, alpha=0.05, verbose=1):
        n = X.shape[0]
        ME_list = []
        ME_std_list = []
        if verbose==1:
            print('Bootstrapping CI:')
        for b in range(B):
            if verbose==1:
                print('Iteration '+str(b+1)+' of '+str(B)+': ', end='')
            clear_session()
            ind = np.random.choice(n, n)
            ind = np.arange(0,n)
            Xb = X[ind,:]
            Yb = Y[ind,:]
            self.train(Xb, Yb, w)
            if verbose==1:
                print('Training done. ', end='')
            ME_list.append(self.compute_ME(X0))
            if verbose==1:
                print('ME computed. ', end='')
            ME_std_list.append(self.compute_ME_std(X0))
            if verbose==1:
                print('ME_std computed.')
        
        # obtain the CI for the ME based on the bootstrap results
        m,kp,k = ME_list[0].shape
        ME_CI_low = np.zeros((m,kp,k))
        ME_CI_upp = np.zeros((m,kp,k))
        
        print('Computing quantiles: ', end='')
        for i in range(m):
            for j in range(kp):
                for l in range(k):
                    ME_ = [ME_list[x][i,j,l] for x in range(B)]
                    ME_std_ = [ME_std_list[x][i,j,l] for x in range(B)]
                    F = lambda z: np.mean(norm.cdf(z, ME_, ME_std_))
                    ME_CI_low[i,j,l] = bisect(lambda z: F(z) - alpha/2, 
                                              np.min(ME_) - 2*np.max(ME_std_),
                                              np.max(ME_) + 2*np.max(ME_std_))
                    ME_CI_upp[i,j,l] = bisect(lambda z: F(z) - (1-alpha/2), 
                                              np.min(ME_) - 2*np.max(ME_std_),
                                              np.max(ME_) + 2*np.max(ME_std_))
        print('completed.')
        
        self.ME_list = ME_list
        self.ME_std_list = ME_std_list
        
        return np.mean(ME_list,0), ME_CI_low, ME_CI_upp
        
        
    def __compute_dM__(self, X0, verbose=0):
        
        model = self.MLP
        X0 = (X0 - self.X_mean) / self.X_std
        ##########################################################################
        # Model extraction #######################################################
        
        # number of evaluation points
        m, k = X0.shape
        
        # Get list of weight matrices and bias terms
        Wb = model.get_weights()
        W = [x.transpose() for i,x in enumerate(Wb[0::2])]
        b = [x.reshape((-1,1)) for i,x in enumerate(Wb[1::2])]
        
        # number of layers
        l = len(b) + 1
        
        # output dimension
        d = W[-1].shape[0]
        
        # neurons per layer
        neur = [x.shape[1] for x in W] + [d]
        
        # Get list of activation functions
        activation = [None] + [x['config']['activation'] for x in model.get_config()['layers'][1:]]
        
        # During the loop, we compute dME for the evaluation points from X0.
        dME = [None] * m
        
        if verbose==1:
            print('Working on observation', end=' ')
        for i in range(m):
            if verbose==1:
                print(i+1, end=' ')
            bla = 0
            ######################################################################
            # Preparing calculations #############################################
            
            # Create list of layer elements
            s = [None] * l # non-activated weighted information of last layer
            z = [None] * l # activated layer-output
            dsig = [None] * l # derivative of activation function
            d2sig = [None] * l # 2nd order derivative of activation function
            
            # In each layer, we have derivatives of the activated and non-activated
            # outputs z and s with respect to the activated and non-activated 
            # outputs of lower layers. We store all these derivatives in a list of 
            # lists of the following shape:
            # dzdz = [[dz0dz0],
            #         [dz1dz0,   dz1dz1],
            #         ...
            #         [dzl-1dz0, dzl-1dz1, ..., dzl-1dzl-1]]
            # dzds = [[None],
            #         [None, dz1ds1],
            #         ...
            #         [None, dzl-1ds1, ..., dzl-1dsl-1]]
            # dsds = [[None],
            #         [None, ds1ds1],
            #         ...
            #         [None, dsl-1ds1, ..., dsl-1dsl-1]]
            # The first 'columns' of dzdz and dsds remain 'None', as s0 does not
            # exist.
            dzdz = [None] * l
            dzds = [None] * l
            dsds = [None] * l
            
            sum_expr = [None] * l
            
            # output of first layer is input x
            z[0] = X0[i, :].reshape((-1,1)) 
            
            # We fill in the first elements of dzdz etc, as the loop only iterates 
            # over l-1 elements.
            dzdz[0] = [np.eye(neur[0])]
            dzds[0] = [None]
            dsds[0] = [None]
            
            # forward step to compute s, z, dsig, d2sig for all layers, and me
            for j in range(l-1):
                # forward propagation
                s[j+1] = W[j] @ z[j] + b[j] 
                # activation
                z[j+1], dsig[j+1], d2sig[j+1] = self.__activate__(s[j+1], activation[j+1], return_derivatives = 2)
                
                # Each entry of the list is a new list, containing the derivatives
                # with respect to lower layers. I.e: dz_5/dz_2 = dzdz[5][2]
                dzdz[j+1] = [None] * (j+2)
                dzds[j+1] = [None] * (j+2)
                dsds[j+1] = [None] * (j+2)
                # We fill the first column of dzdz and the 'diagonal elements' of 
                # the lists, then we can iterate over the other elements in a loop
                dzdz[j+1][j+1] = np.eye(neur[j+1])
                dzds[j+1][j+1] = dsig[j+1]
                dsds[j+1][j+1] = np.eye(neur[j+1])
                dzdz[j+1][0] = dsig[j+1] @ W[j] @ dzdz[j][0]
                # Now the remaining elements of dzdz etc are filled in the loop
                for it in np.arange(1, j+1):
                    dzdz[j+1][it] = dsig[j+1] @ W[j] @ dzdz[j][it]
                    dzds[j+1][it] = dzdz[j+1][it] @ dsig[it]
                    dsds[j+1][it] = W[j] @ dzds[j][it]
            
            ######################################################################
            # Preparing calculations 2 ###########################################
            # In this part, we compute the Hessian H = d2ydw2 and its contribution
            # to d2l. 
            # Moreover, we compute the blocks of the derivative of the marginal
            # effects.
            
            # sum expression in the middle. We save it as a list of lists, the 
            # first index is the output dimension (di) and the second index is eta
            sum_expr = [None] * d
            
            for di in range(d):            
                sum_expr[di] = [sum([dzdz[l-1][l-eta-1][di, c] * d2sig[l-eta-1][c, :, :] 
                                for c in range(neur[l-eta-1])]) for eta in range(l-1)]
            
            # We create blocks for dME(di)/dw for di in range(d) 
            dme = [None] * d
            
            # Firstly, we compute the Hessian blocks. The blocks are saved as 
            # lists of lists of lists. The first index is the output dimension, 
            # the second index the first and the third index the second layer with 
            # respect to which the derivative is taken.
            # Secondly, computes from the blocks the derivative of the marginal
            # effect with respect to the parameters dME/dw.
            for di in range(d):
                # for each output dimension, we create l-1 'None's for the l-1 
                # weight and bias matrices.
                Hww = [None] * (l-1)
                Hwb = [None] * (l-1)
                Hbw = [None] * (l-1)
                Hbb = [None] * (l-1)
                
                for p in range(l-1):
                    # We create in each element l-1 '0's for the l-1
                    # weight and bias matrices (2nd derivative). We use zeros, 
                    # because we will add contributions (see sum over eta) while
                    # iterating over eta.
                    Hww[p] = [0] * (l-1)
                    Hwb[p] = [0] * (l-1)
                    Hbw[p] = [0] * (l-1)
                    Hbb[p] = [0] * (l-1)
                    
                    for eta in range(l-p-1):
                        # left hand side for Hw_ and Hb_, respectively
                        lhs_w = np.kron(dsds[l-eta-1][p+1].transpose(), z[p]) @ sum_expr[di][eta]
                        lhs_b = dsds[l-eta-1][p+1].transpose() @ sum_expr[di][eta]
                        
                        for q in np.arange(p, l-eta-1):
                            # kronecker product that builds the right hand side
                            # of blocks H_w
                            rhs_w = np.kron(dsds[l-eta-1][q+1], z[q].transpose())
                            rhs_b = dsds[l-eta-1][q+1]
                            # updating the blocks
                            bla += 1
                            Hww[p][q] += lhs_w @ rhs_w
                            Hbw[p][q] += lhs_b @ rhs_w
                            Hwb[p][q] += lhs_w @ rhs_b
                            Hbb[p][q] += lhs_b @ rhs_b
                    
                # in the following loop, we fill in the remaining blocks, which
                # are given by the transpose of blocks that have already been
                # computed. We also add the additional terms to Hww and Hbw
                for p in range(l-1):
                    for q in np.arange(p+1, l-1):
                        # additional terms for Hbw and Hww, respectively
                        add_Hbw = np.kron(dzds[l-1][q+1][di,:].reshape((1,-1)), dzds[q][p+1].transpose())
                        add_Hww = np.kron(add_Hbw, z[p])
                        Hww[p][q] += add_Hww
                        Hbw[p][q] += add_Hbw
                        # remaining blocks
                        Hww[q][p] = Hww[p][q].transpose()
                        Hbb[q][p] = Hbb[p][q].transpose()
                        Hwb[q][p] = Hbw[p][q].transpose()
                        Hbw[q][p] = Hwb[p][q].transpose()
                
                ##################################################################
                # Computation of the derivative of the marginal effect ###########
                
                # We only compute the derivative for the evaluation points, i.e.
                # not for the first n iterations.
                dmew = [None] * (l-1)
                dmeb = [None] * (l-1)
                for p in range(l-1):
                    dmew[p] = W[0].transpose() @ Hbw[0][p]
                    dmeb[p] = W[0].transpose() @ Hbb[0][p]
                dmew[0] += np.kron(dzds[l-1][1][di,:].reshape((1,-1)), np.eye(neur[0]))
                # We create an array from the separate blocks to obtain the
                # derivative dME(id)/dw
                dme[di] = np.block([np.block(dmew), np.block(dmeb)])
                
            ######################################################################
            # Computation dME ####################################################
            
            # We concatenate the dME(id)/dw to obtain dME/dw=dME
            dME[i] = np.concatenate(dme, axis=0)
        if verbose==1:
            print('')
        
        # construct dM array from dME list
        self.dM = np.array(dME)
        
        return self.dM
    
    
    def __compute_V_theta__(self, verbose=0):
        
        model = self.MLP
        X = (self.X - self.X_mean) / self.X_std
        Y = (self.Y - self.Y_mean) / self.Y_std
        
        ##########################################################################
        # Model extraction #######################################################
        
        # Input dimension and number of observations
        n, k = X.shape
        
        # Get list of weight matrices and bias terms
        Wb = model.get_weights()
        W = [x.transpose() for i,x in enumerate(Wb[0::2])]
        b = [x.reshape((-1,1)) for i,x in enumerate(Wb[1::2])]
        
        # number of layers
        l = len(b) + 1
        
        # output dimension
        d = W[-1].shape[0]
        
        # neurons per layer
        neur = [x.shape[1] for x in W] + [d]
        
        # Get list of activation functions
        activation = [None] + [x['config']['activation'] for x in model.get_config()['layers'][1:]]
        
        # loss function that was used for training
        loss = model.loss
        
        # predicted values
        yhat = model.predict(X)
        
        ##########################################################################
        # Initialisation of V1, V2 and dME #######################################
        
        # We initiate V1 and V2 as zeros and update it during the loop for each
        # observation
        V1 = 0
        V2 = 0
        
        # During the loop, we compute:
        # dl and d2l for each observation and update V1 and V2, respectively.
        
        if verbose==1:
            print('Working on observation', end=' ')
        for i in range(n):
            if verbose==1:
                print(i+1, end=' ')
            bla = 0
            ######################################################################
            # Preparing calculations #############################################
            
            # Create list of layer elements
            s = [None] * l # non-activated weighted information of last layer
            z = [None] * l # activated layer-output
            dsig = [None] * l # derivative of activation function
            d2sig = [None] * l # 2nd order derivative of activation function
            
            # In each layer, we have derivatives of the activated and non-activated
            # outputs z and s with respect to the activated and non-activated 
            # outputs of lower layers. We store all these derivatives in a list of 
            # lists of the following shape:
            # dzdz = [[dz0dz0],
            #         [dz1dz0,   dz1dz1],
            #         ...
            #         [dzl-1dz0, dzl-1dz1, ..., dzl-1dzl-1]]
            # dzds = [[None],
            #         [None, dz1ds1],
            #         ...
            #         [None, dzl-1ds1, ..., dzl-1dsl-1]]
            # dsds = [[None],
            #         [None, ds1ds1],
            #         ...
            #         [None, dsl-1ds1, ..., dsl-1dsl-1]]
            # The first 'columns' of dzdz and dsds remain 'None', as s0 does not
            # exist.
            dzdz = [None] * l
            dzds = [None] * l
            dsds = [None] * l
            
            sum_expr = [None] * l
            
            # output of first layer is input x
            z[0] = X[i, :].reshape((-1,1)) 
            
            # We fill in the first elements of dzdz etc, as the loop only iterates 
            # over l-1 elements.
            dzdz[0] = [np.eye(neur[0])]
            dzds[0] = [None]
            dsds[0] = [None]
            
            # forward step to compute s, z, dsig, d2sig for all layers, and me
            for j in range(l-1):
                # forward propagation
                s[j+1] = W[j] @ z[j] + b[j] 
                # activation
                z[j+1], dsig[j+1], d2sig[j+1] = self.__activate__(s[j+1], activation[j+1], return_derivatives = 2)
                
                # Each entry of the list is a new list, containing the derivatives
                # with respect to lower layers. I.e: dz_5/dz_2 = dzdz[5][2]
                dzdz[j+1] = [None] * (j+2)
                dzds[j+1] = [None] * (j+2)
                dsds[j+1] = [None] * (j+2)
                # We fill the first column of dzdz and the 'diagonal elements' of 
                # the lists, then we can iterate over the other elements in a loop
                dzdz[j+1][j+1] = np.eye(neur[j+1])
                dzds[j+1][j+1] = dsig[j+1]
                dsds[j+1][j+1] = np.eye(neur[j+1])
                dzdz[j+1][0] = dsig[j+1] @ W[j] @ dzdz[j][0]
                # Now the remaining elements of dzdz etc are filled in the loop
                for it in np.arange(1, j+1):
                    dzdz[j+1][it] = dsig[j+1] @ W[j] @ dzdz[j][it]
                    dzds[j+1][it] = dzdz[j+1][it] @ dsig[it]
                    dsds[j+1][it] = W[j] @ dzds[j][it]
            
            ######################################################################
            # Computation dl #####################################################
            
            # We can obtain the first order derivative of the output with respect
            # to the input from the derivatives obtained above
            dydb = [dzds[-1][la+1] for la in range(l-1)]
            dydW = [np.kron(x, z[i].transpose()) for i,x in enumerate(dydb)]
            dy = np.block(dydW + dydb)
            
            # derivative of loss with respect to yhat
            loss_i, dldy, d2ldy2 = self.__compute_loss__(Y[i, :].reshape((-1,1)), 
                                                          yhat[i, :].reshape((-1,1)), 
                                                          loss)
            
            # score function, derivative of loss of observation w.r.t. parameters
            dl = dy.transpose() @ dldy
            
            ######################################################################
            # Preparing calculations 2 ###########################################
            # In this part, we compute the Hessian H = d2ydw2 and its contribution
            # to d2l. 
            # Moreover, we compute the blocks of the derivative of the marginal
            # effects.
            
            d2l_contrib_2 = 0
            
            # sum expression in the middle. We save it as a list of lists, the 
            # first index is the output dimension (di) and the second index is eta
            sum_expr = [None] * d
            
            for di in range(d):            
                sum_expr[di] = [sum([dzdz[l-1][l-eta-1][di, c] * d2sig[l-eta-1][c, :, :] 
                                for c in range(neur[l-eta-1])]) for eta in range(l-1)]
            
            # Firstly, we compute the Hessian blocks. The blocks are saved as 
            # lists of lists of lists. The first index is the output dimension, 
            # the second index the first and the third index the second layer with 
            # respect to which the derivative is taken.
            # Secondly, computes from the blocks the derivative of the marginal
            # effect with respect to the parameters dME/dw.
            for di in range(d):
                # for each output dimension, we create l-1 'None's for the l-1 
                # weight and bias matrices.
                Hww = [None] * (l-1)
                Hwb = [None] * (l-1)
                Hbw = [None] * (l-1)
                Hbb = [None] * (l-1)
                
                for p in range(l-1):
                    # We create in each element l-1 '0's for the l-1
                    # weight and bias matrices (2nd derivative). We use zeros, 
                    # because we will add contributions (see sum over eta) while
                    # iterating over eta.
                    Hww[p] = [0] * (l-1)
                    Hwb[p] = [0] * (l-1)
                    Hbw[p] = [0] * (l-1)
                    Hbb[p] = [0] * (l-1)
                    
                    for eta in range(l-p-1):
                        # left hand side for Hw_ and Hb_, respectively
                        lhs_w = np.kron(dsds[l-eta-1][p+1].transpose(), z[p]) @ sum_expr[di][eta]
                        lhs_b = dsds[l-eta-1][p+1].transpose() @ sum_expr[di][eta]
                        
                        for q in np.arange(p, l-eta-1):
                            # kronecker product that builds the right hand side
                            # of blocks H_w
                            rhs_w = np.kron(dsds[l-eta-1][q+1], z[q].transpose())
                            rhs_b = dsds[l-eta-1][q+1]
                            # updating the blocks
                            bla += 1
                            Hww[p][q] += lhs_w @ rhs_w
                            Hbw[p][q] += lhs_b @ rhs_w
                            Hwb[p][q] += lhs_w @ rhs_b
                            Hbb[p][q] += lhs_b @ rhs_b
                    
                # in the following loop, we fill in the remaining blocks, which
                # are given by the transpose of blocks that have already been
                # computed. We also add the additional terms to Hww and Hbw
                for p in range(l-1):
                    for q in np.arange(p+1, l-1):
                        # additional terms for Hbw and Hww, respectively
                        add_Hbw = np.kron(dzds[l-1][q+1][di,:].reshape((1,-1)), dzds[q][p+1].transpose())
                        add_Hww = np.kron(add_Hbw, z[p])
                        Hww[p][q] += add_Hww
                        Hbw[p][q] += add_Hbw
                        # remaining blocks
                        Hww[q][p] = Hww[p][q].transpose()
                        Hbb[q][p] = Hbb[p][q].transpose()
                        Hwb[q][p] = Hbw[p][q].transpose()
                        Hbw[q][p] = Hwb[p][q].transpose()
                
                # We create H and update the contribution
                # create blocks to obtain H
                H = np.block([[np.block(Hww), np.block(Hwb)],[np.block(Hbw), np.block(Hbb)]])
                
                # update the contribution to d2l
                d2l_contrib_2 += H * dldy[di,0]
            
            # We compute d2l and update V1, V2 for the n observations
            ##################################################################
            # Computation d2l ################################################
            
            # The first contribution to d2l is the weighted outer product of 
            # the derivative dy
            d2l_contrib_1 = dy.transpose() @ d2ldy2 @ dy
            
            # Together with the second contribution from the Hessian, we can 
            # now compute d2l
            d2l = d2l_contrib_1 + d2l_contrib_2
            
            ##################################################################
            # Updating of V1 and V2 ##########################################
            
            # V1 is updated by the outer product of the gradient dl @ dl'
            V2 += dl @ dl.transpose()
            # V2 is updated by the Hessian d2l
            V1 += d2l
        if verbose==1:
            print('')
        
        # To obtain the estimates for V1 and V2, we need to normalize by devide by
        # the number of observations
        self.d2L = V1 / n
        self.dLdL = V2 / n
        
        # number of total parameters
        num_w = self.d2L.shape[1]
        
        # stabilize the Hessian estimate (d2L) by inflating the diagonal elements
        if verbose==1:
            print('Starting to make robust.', end=' ')
        d2L_robust = np.copy(self.d2L)
        its = 0
        while True:
            if np.linalg.matrix_rank(d2L_robust) == num_w:
                break
            np.fill_diagonal(d2L_robust, np.diag(d2L_robust)*1.01+1E-15*2**its)
            its+=1
        if verbose==1:
            print('Done making robust after ' + str(its) + ' iterations.')
        # np.fill_diagonal(d2L_robust, np.diag(d2L_robust)+2*self.l2_penalty)
        
        # construct the asymptotic covariance matrix of the weights
        d2L_inv = inv(d2L_robust)
        self.V_theta = d2L_inv @ self.dLdL @ d2L_inv
        
        return self.V_theta
    
    
    def __activate__(self, s, activation, return_derivatives = 0):
        """
        activate(s, activation)
        
        Activate the weighted output from the last layer.
        
        Parameters
        ----------
        s: (nl, 1) array
            non-activated weighted sum of outputs from last layer
            nl - number of neurons in the layer
        activation: str
            Activation function that shall be applied. Should be one of
            - 'linear'
            - 'tanh'
        return_derivatives: int, optional
            States the order, up to which derivatives are obtained. 
            Should be one of
            - 0 (no derivative returned, only activated layer output)
            - 1 (first oder derivative of activation computed)
            - 2 (first and second order derivative ar computed)
        
        Returns
        -------
        z: (nl, 1) array
            activated output of the layer
            nl - number of neurons in the layer
        dsig: (nl, nl) array, optional 
            (returned if return_derivatives >=1)
            First order derivative of activation function, evaluated at input s
            nl - number of neurons in the layer
        d2sig: (nl, nl, nl) array, optional 
            (returned if return_derivatives ==2)
            Second order derivative of activation function, evaluated at input s
        """
        
        # number of neurons in the layer
        nl = s.shape[0]
        
        if activation == 'linear':
            # for a linear activation function, the output is just the weighted
            # inputs and the derivative is an identity matrix
            z = s 
            dsig = np.eye(nl)
            d2sig = np.zeros((nl, nl, nl))
        elif activation == 'tanh':
            # under tanh activation, the derivative is a diagonal matrix of the 
            # 1/cosh**2 elements.
            z = np.tanh(s)
            dsig = np.diag((1/(np.cosh(s)**2)).reshape((-1)))
            d2sig = np.zeros((nl,nl,nl))
            for i in range(nl):
                d2sig[i,i,i] = -2 * np.sinh(s[i]) / np.cosh(s[i])**3
        else:
            raise NameError('Activation function not supported.')
        
        if return_derivatives == 0:
            return z
        elif return_derivatives == 1:
            return z, dsig
        elif return_derivatives == 2:
            return z, dsig, d2sig
        else:
            raise NameError('return_derivatives must be 0,1 or 2.')
            
    def __compute_loss__(self, y, yhat, loss, return_derivatives=2):
        """
        compute_loss(y, yhat, loss)
        
        Compute the loss and, if needed, its derivatives.
        
        Parameters
        ----------
        y: (d, 1) array
            Observed dependent variables (targets)
            d - output dimension
        yhat: (d, 1) array
            Predicted dependent variables (outputs)
            d - output dimension
        loss: str
            Loss function that shall be computed. 
            Should be one of
            - 'mse'
        return_derivatives: int, optional
            States the order, up to which derivatives are obtained. 
            Should be one of
            - 0 (no derivative returned, only activated layer output)
            - 1 (first oder derivative of activation computed)
            - 2 (first and second order derivative ar computed)
        
        Returns
        -------
        l: numpy.float
            loss
        dl: (d, 1) array
            derivative of loss w.r.t. yhat
            d - output dimension
        d2l: (d, d) array
            2nd order derivative of loss w.r.t. yhat
            d - output dimension
        """
        
        # output dimension
        d = y.shape[0]
        
        if loss == 'mse':
            l = np.sum((y - yhat)**2)
            dl = -2 * (y - yhat)
            d2l = 2 * np.eye(d)
            return l, dl, d2l
        else:
            raise NameError('Loss not supported.')
    
    def predict(self, X_raw):
        """
        X_raw: np.ndarray
            (T, k) array of observed series
        """
        # obtain length and number of series
        n, w1 = X_raw.shape
        
        # Standardize series
        X = (X_raw - self.X_mean) / self.X_std
        
        # predict output
        Y_pred = self.MLP.predict(X)
        
        # return rescaled variables
        return Y_pred * self.Y_std + self.Y_mean


