#!/usr/bin/env python
# -*- coding: utf-8 -*-
#pylint:disable=invalid-name
"""
AMTL method.
"""
import time
import numpy as np
from scipy.optimize import approx_fprime
from sklearn.linear_model import Lasso
from codes.design import Method
from codes.optimization.admm import ADMM, ADMM_Lasso
from codes.optimization.gradient_descent import GradientDescent

VERBOSE = False
TOLERANCE = 1e-3
TOLERANCE_W = 1e-3
TOLERANCE_B = 1e-3
MAX_ITER_W = 25
MAX_ITER_B = 25
MAX_ITER = 700

class MyAMTLBase(Method):
    """ AMTL. """
    def __init__(self, name="MyAMTL", label='amtl', normalize=False, bias=False,
                 mu=1, lamb=1, glm="Gaussian", consider_loss=True,
                 consider_B_restriction=True):
        super().__init__(name, label, normalize, bias)
        assert mu >= 0
        self.mu = mu
        assert lamb >= 0
        self.lamb = lamb
        assert isinstance(glm, str)
        self.glm = glm
        self.consider_loss = consider_loss
        self.consider_B_restriction = consider_B_restriction
        self.W = None
        self.B = None
        self.use_sk = False
        self.cost_hist = None

    def fit(self, X, y, max_iter=MAX_ITER):
        """
            Fits method.
            Args:
                X (list np.array): list of task data. each component is a task data with shape m_t X n.
                y (list np.array): list of task label. each componen is a task label with shape m_t.
                max_iter (int): 100
            Returns:
                W (np.array): parameter matrix n x T.
                f (np.array): if available, the cost function at each iteration.
                time (int): elapsed time in secs.
        """
        assert max_iter >= 0
        assert len(X) == len(y)
        for t, Xt in enumerate(X):
            assert Xt.shape[0] == y[t].shape[0]
            assert Xt.shape[1] == X[0].shape[1]
        X = self.normalize_data(X)
        X = self.add_bias(X)
        n_tasks = len(X)
        n_dim = X[0].shape[1]

        #initialization
        self.W = np.random.randn(n_dim, n_tasks)
        self.B = np.zeros((n_tasks, n_tasks))
        self.cost_hist = np.zeros((max_iter, 2))
        #print("Iniciando treinamento")
        start = time.time()
        for i in range(max_iter):
            W_old = np.copy(self.W)
            W, current_losses = self.train_W(W_old, X, y)
            self.W = W
            if VERBOSE:
                self.cost_hist[i, 0] = self._cost_function(X, y)
                print('W step obj_val: {:.4f}'.format(self.cost_hist[i, 0]))
            B_old = np.copy(self.B)
            B = self.train_B(B_old, current_losses)
            self.B = B
            if VERBOSE:
                self.cost_hist[i, 1] = self._cost_function(X, y)
                print('B step obj_val: {:.4f}'.format(self.cost_hist[i, 1]))
                print('-'*70)
            if np.linalg.norm(self.W - W_old) <= TOLERANCE_W and np.linalg.norm(self.B - B_old) <= TOLERANCE_B:
                print("AMTL tolerance met")
                break
            if i == max_iter-1:
                print('AMTL did not converge...')
        return (self.W, self.cost_hist[:, 1], time.time() - start)

    def train_W(self, W, X, y):
        """ Optimizes the Wt part of the method. """
        W = np.copy(W)
        current_losses = np.ones(W.shape[1])
        for t in range(W.shape[1]):
            #print('\t w{}'.format(t))
            opt_wt = Opt_Wt(W, self.B, t, self.glm, self.mu, self.lamb)
            w_opt = opt_wt.fit(X[t], y[t], w=W[:, t], max_iter=MAX_ITER_W)
            W[:, t] = w_opt
            if self.consider_loss:
                loss = opt_wt.cost_glm(X[t], y[t], W[:, t])
                current_losses[t] = loss #opt_wt.cost_glm(X[t], y[t], W[:, t])
        current_losses = np.maximum(current_losses, 1e-4)
        return (W, current_losses)

    def train_B(self, B, current_losses):
        """ Optimizes the Bgt part of the method. """
        n_tasks = self.W.shape[1]
        B = np.copy(B)
        temp_X = self.W.copy()
        if self.consider_loss:
            temp_X = np.divide(temp_X, current_losses)
        for t in range(n_tasks):
            #print('\t b{}'.format(t))
            selector = [i for i in range(n_tasks) if i != t]
            W_bar = temp_X[:, selector].copy()
            w_tilde = self.W[:, t]
            #w_tilde.shape = (len(w_tilde), 1)
            if self.bias:
                W_bar = W_bar[:-1, :]
                w_tilde = w_tilde[:-1]
            coeff = None
            if self.consider_B_restriction:
                if self.use_sk:
                    lasso_cons = Lasso(alpha=self.mu/self.lamb, positive=True,
                            max_iter=MAX_ITER_B)
                    lasso_cons.fit(W_bar, w_tilde)
                    coeff = lasso_cons.coef_
                else:
                    lasso_cons = ADMM(self.mu/self.lamb, max_iter=MAX_ITER_B)
                    #coeff, _ = lasso_cons.fit(W_bar, w_tilde, w=B[selector, t])
                    coeff, _ = lasso_cons.fit(W_bar, w_tilde)
            else:
                if self.use_sk:
                    lasso = Lasso(alpha=self.mu/self.lamb, max_iter=MAX_ITER_B)
                    lasso.fit(W_bar, w_tilde)
                    coeff = lasso.coef_
                else:
                    lasso = ADMM_Lasso(lamb=self.mu/self.lamb, max_iter=MAX_ITER_B)
                    #coeff, _ = lasso.fit(W_bar, w_tilde, w=B[selector, t])
                    coeff, _ = lasso.fit(W_bar, w_tilde)
            if self.consider_loss:
                B[selector, t] = coeff / current_losses[selector]
            else:
                B[selector, t] = coeff
        return B

    def set_params(self, mu=0.01, lamb=0.1):
        """
        Sets method params.
        Args:
            mu (float): mu parameter.
                Default: 0.01.
            lamb (float): lambda parameter.
                Default: 0.1.
        """
        assert mu >= 0
        self.mu = mu #task contribution penalized by loss
        assert lamb >= 0
        self.lamb = lamb #projection on basis B

    def get_params(self):
        """
        Gets method params.
        Return
            params (dict):
        """
        ret = {'mu': self.mu, 'lamb': self.lamb}
        return ret

    def get_resul(self):
        res = {'W': self.W,
               'B': self.B}
        return res

    def _cost_function(self, X, y):
        def __cost_glm(A, b, w):
            f = 0
            if self.glm == 'Gaussian':
                tt = np.dot(A, w) - b
                # nao é loglik mesmo, é só mse
                loglik = 0.5 * np.linalg.norm(tt) ** 2.0
            elif self.glm == 'Poisson':
                xb = np.maximum(np.minimum(np.dot(A, w), 100), -100)#avoid overflow
                loglik = -(b * xb - np.exp(xb)).sum()
            elif self.glm == 'Gamma':
                loglik = 0
                for i in range(0, A.shape[0]):
                    loglik += scipy.stats.gamma.logpdf(b[i], 1.0 / np.dot(A[i, :], w))
            elif self.glm == 'Binomial':
                Xbeta = np.dot(A, w)
                loglik = -1 * np.sum(((b * Xbeta) - np.log(1 + np.exp(Xbeta))))
            if not np.isnan(loglik):
                loglik /= float(A.shape[0])
                f += loglik
            else:
                print("****** WARNING: loglik is nan.")
            return f

        def termo_1(A, b, w, t):
            """ Computes the first term of the cost function. """
            termo_1 = 0
            if self.mu > 0:
                termo_1 = __cost_glm(A, b, w)
                if self.consider_loss:
                    temp_1 = sum(abs(self.B[t])) #linha t de Bs
                    termo_1 *= (1 + self.mu * temp_1)
            return termo_1

        def termo_2(w, t):
            """ w_t and its projection. """
            termo_2 = 0
            if self.lamb > 0:
                proj = np.zeros(len(w))
                assert self.B[t, t] == 0
                proj = np.dot(self.W, self.B[:, t])
                termo_2 = (self.lamb / 2) * np.linalg.norm(w - proj)**2
            return termo_2
        ret = 0
        for t, Xt in enumerate(X):
            t1 = termo_1(Xt, y[t], self.W[:, t], t)
            t2 = termo_2(self.W[:, t], t)
            ret += t1 + t2
        return ret

class Opt_Wt:
    """ Implements the step of optimization of wt in MyAMTL.  """
    def __init__(self, W, B, t, glm="Gaussian", consider_loss=True, mu=0, lamb=0):
        self.W = W
        self.B = B
        self.t = t
        self.glm = glm
        self.mu = mu
        self.lamb = lamb
        self.mean = True
        self.consider_loss = consider_loss
        #a_s
        self.a_s = self.W - np.dot(self.W, B)
        self.a_s[:, self.t] = 0.

    def fit(self, X, y, w=None, max_iter=MAX_ITER_W):
        """TODO: Docstring for fit.

        :X: TODO
        :y: TODO
        :w: TODO
        :returns: TODO

        """
        optimizer = GradientDescent()
        if w is None:
            w = np.random.randn(X.shape[1])
        w_t, _ = optimizer.fit(X, y, self.cost, self.grad, w=w,
                max_iter=max_iter)
        return w_t

    def cost(self, X, y, w):
        """ Computes cost function. """
        res_sum = 0
        copia = self.W.copy()
        copia[:, self.t] = w
        res_sum = (((copia - np.dot(copia, self.B)) ** 2).sum(axis=0)).sum()
        if self.consider_loss:
            f = (1 + self.mu * abs(self.B[self.t, :]).sum()) * self.cost_glm(X, y, w) \
                + (self.lamb / 2) * res_sum
        else:
            f = self.mu * abs(self.B[self.t, :]).sum() + self.cost_glm(X, y, w) \
                + (self.lamb / 2) * res_sum
        return f

    def grad(self, X, y, w):
        """ Computes the gradient of self.cost at point w. """
        copy = self.W.copy()
        copy[:, self.t] = w
        B = self.B.copy()
        B[self.t, :] = 0.
        res_sum2 = w - np.dot(copy, self.B[:, self.t])
        res_sum2 -= (self.B[self.t, :].reshape(1, self.W.shape[1]) * \
                (self.a_s - (self.B[self.t, :].reshape(1, self.W.shape[1]) * w.reshape(len(w), 1)))).sum(axis=1)
        res_sum = w - np.dot(self.W, self.B[:, self.t])
        for s in range(self.W.shape[1]):
            if s == self.t:
                continue
            selector = list(range(self.W.shape[1]))
            selector.remove(self.t)
            selector.remove(s)
            a_s = self.W[:, s] - np.dot(self.W[:, selector], self.B[selector, s])
            res_sum -= self.B[self.t, s] * (a_s - self.B[self.t, s] * w)
        if self.consider_loss:
            f = (1 + self.mu * sum(abs(self.B[self.t, :]))) * self.grad_glm(X, y, w) + \
                self.lamb * res_sum
        else:
            f = self.grad_glm(X, y, w) + self.lamb * res_sum
        return f

    def grad2(self, X, y, w):
        def my_cost(w):
            return self.cost(X, y, w)
        return approx_fprime(w, my_cost, 1e-10)

    def cost_glm(self, X, y, w):
        """Cost function for the glm part.

        Args:
            A (np.array): design matriz.
            b (np.array): label vector.
            w (np.array): Weight vector for task self.t.
        Returns:
            f (float): log-likelihood according to the self.glm parameter.
        """
        f = 0
        if self.glm == 'Gaussian':
            tt = np.dot(X, w) - y
            loglik = 0.5 * np.linalg.norm(tt) ** 2.0
        elif self.glm == 'Poisson':
            xb = np.maximum(np.minimum(np.dot(X, w), 100), -100)#avoid overflow
            loglik = -(y * xb - np.exp(xb)).sum()
        elif self.glm == 'Gamma':
            loglik = 0
            for i in range(0, X.shape[0]):
                loglik += scipy.stats.gamma.logpdf(y[i], 1.0 / np.dot(X[i, :], w))
        elif self.glm == 'Binomial':
            Xbeta = np.dot(X, w)
            loglik = -1 * np.sum(((y * Xbeta) - np.log(1 + np.exp(Xbeta))))
        if not np.isnan(loglik):
            loglik /= float(X.shape[0])
            f += loglik
        else:
            print("****** WARNING: loglik is nan.")
        return f

    def grad_glm(self, X, y, w):
        """
            Computes the gradient of the glm part.

            Args:
                A (np.array): design matriz.
                B (np.array): label vector.
                w (np.array): Weight vector for task self.t.
            Returns:
                grad (np.array): gradient vector.
        """
        tmp = np.zeros(w.shape)
        if self.glm == 'Gaussian':
            Xwmy = np.dot(X, w) - y
            tmp = np.dot(Xwmy, X)
        elif self.glm == 'Poisson':
            xb = np.maximum(np.minimum(np.dot(X, w), 200), -200) #avoid overflow
            tmp = -np.dot(X.T, y - np.exp(xb))
        elif self.glm == 'Gamma':
            kappa = 0.5
            tmp = kappa * np.dot(X.T, np.reciprocal(np.dot(X, w.T) - y))
        elif self.glm == 'Binomial':
            Xbeta = np.dot(X, w)
            pi = np.reciprocal(1.0 + np.exp(-Xbeta))
            tmp = -np.dot(X.T, (y.flatten() - pi.flatten()))
        if self.mean:
            tmp *= 1.0 / float(X.shape[0])
        return tmp
