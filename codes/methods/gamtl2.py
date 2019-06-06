# -*- coding: utf-8 -*-
#pylint: disable=missing-docstring, arguments-differ,invalid-name,no-member,too-many-arguments,too-many-instance-attributes,too-many-locals,too-many-statements
"""
Classe do GAMTL, segunda implementação, compatível com design.py:Method.
"""
import time
from abc import ABCMeta
import numpy as np
import scipy.stats
import scipy.sparse
import scipy.optimize
from sklearn.linear_model import Lasso
from codes.design import Method
#from codes.optimization.fista import Fista
from codes.optimization.admm import ADMM
from codes.optimization.admm import ADMM_Lasso
from codes.methods.gamtl import Optimize_Wt


DEBUG_G = False
DEBUG_W = False
MAX_ITER_AGMTL = 50
MAX_ITER_WT = 10
MAX_ITER_BGT = 10
TOLERANCE_W = 10e-3
TOLERANCE_Bs = 10e-3
VERBOSE = True


class GroupAMTLBase(Method):
    """
        Retrieves the group relationship of the tasks, while training them.

        Attributes:
            lambda_1 (float): lambda of the task contribution weighted by loss.
                Default: 1.
            lambda_2 (float): lambda of the projection on basis.
                Default: 1.
            lambda_3 (float): lambda for the group lasso regularization.
                Default: 1.
            W (np.array): n_features x n_tasks Weight matrix. Init with None.
            Bs (np.array): n_groups x n_tasks x n_tasks. Relationship matrix
                per group. Initialized with None.
            groups (np.array()): [[start g1, start g2, ..., start gG]
                                  [end g1, end g2, ..., end gG]
                                  [weight g1, weight g2, ... weight gG]]
                      Group representation. Each column of this matrix should
                      contain the start index, stop index, and group weight.
                      Recommended value = np.sqrt(end g - start g).
            glm (str): valid values: 'Gaussian', 'Poisson', 'Gamma'.
            cost_hist (list): values of the cost function per iteration.
                first column is the cost after all w_t optimization, second
                column is the cost after all Bg optimization.
                The number of rows equals the number of iterations, up to
                max_iter.

        References:
            [1] - A. Banerjee, S. Chen, F. Fazayeli, V. Sivakumar, Estimation
            with Norm Regularization, in: Advances in Neural Information
            Processing Systems (NIPS), 2014, pp. 1556-1564.
    """
    __metaclass__ = ABCMeta

    def __init__(self,
                 name="GroupAMTL2",
                 label='gamtl2',
                 normalize=False,
                 bias=False,
                 threshold=0.5,
                 groups=None,
                 lambda_1=0.01,
                 lambda_2=0.01,
                 lambda_3=0.01,
                 glm="Gaussian",
                 cache=True,
                 vary=False,
                 consider_loss=True,
                 consider_cross_loss=False,
                 consider_B_restriction=True,
                 use_sk=False):
        """
            Inicializa as variáveis internas.
            Args:
                name (str): Default 'GroupAMTL'
                label (str): Default 'GroupAMTL'
                consider_loss (bool): Consider loss when penalizing Bgt.
                    Default True.
                consider_B_restriction (bool): Bgt >= 0. Default True.
        """
        super().__init__(name, label, normalize, bias)
        assert groups is not None
        assert groups.shape[0] == 3
        assert groups.shape[1] > 0
        self.groups = groups
        assert isinstance(glm, str)
        self.glm = glm
        assert lambda_1 >= 0
        self.lambda_1 = lambda_1  # task contribution penalized by loss
        assert lambda_2 >= 0
        self.lambda_2 = lambda_2  # projection on basis B
        assert lambda_3 >= 0
        self.lambda_3 = lambda_3  # group lasso reg parameter
        self.Bs = None
        self.cost_hist = []
        self.mean = True  # MAKE SURE!!!! (grad more stable)
        self.W = None
        self.consider_loss = consider_loss
        self.consider_cross_loss = consider_cross_loss
        self.consider_B_restriction = consider_B_restriction
        self.scale_for_B = False
        self.use_sk = use_sk
        self.cache = cache
        self.vary = vary

    def fit(self, X, y,
            max_iter=MAX_ITER_AGMTL,
            max_iter_wt=MAX_ITER_WT,
            max_iter_bgt=MAX_ITER_BGT,
            W=None,
            Bs=None):
        """
            Fits method GroupAMTL to data.

            Args
                X (np.array()): list of np.arrays representing datasets, one
                    per task. All datasets must contain the same number of
                    features.
                y (np.array()): list of np.arrays representing the labels, one
                    per task. All components must have the same number of
                    entries as the respective X.
                max_iter (int): maximum number of iterations of the outer
                    optimization proccess. One iteration includes the whole
                    process of optimizing W and Bs.
                max_iter_wt (int): maximum number of iterations of the min w_t
                    step.
                max_iter_bgt (int): maximum number of iterations of the
                    admm constrained lasso step.
                W (np.array()): optional, parameter matrix.
            Return
                W  (np.array()): estimated parameter matrix (self.W).
                f  (float): history of function values per iteration.
                time  (int): processing time spent in the whole proccess.
        """
        assert self.groups is not None
        assert self.groups.shape[0] == 3
        assert self.groups.shape[1] >= 1
        assert max_iter > 0
        assert max_iter_wt > 0
        assert max_iter_bgt > 0
        assert len(X) == len(y)
        for t, Xt in enumerate(X):
            assert Xt.shape[0] == y[t].shape[0]
            assert Xt.shape[1] == X[0].shape[1]
        X = self.normalize_data(X)
        X = self.add_bias(X)
        n_tasks = len(X)
        n_features = X[0].shape[1]
        n_groups = self.groups.shape[1]
        if W is None:
            self.W = np.random.randn(n_features, n_tasks)
        else:
            assert W.shape == (n_features, n_tasks)
            self.W = W
        if Bs is None:
            self.Bs = np.zeros((n_groups, n_tasks, n_tasks))
            # self.Bs = np.random.rand(n_groups, n_tasks, n_tasks) * 30
            # diag_zero = np.ones((n_tasks, n_tasks)) - np.eye(n_tasks)
            # for i in np.arange(n_groups):
            #     self.Bs[i, :, :] = np.multiply(self.Bs[i, :, :], diag_zero)
        else:
            assert Bs.shape == (n_groups, n_tasks, n_tasks)
            for i in range(len(Bs)):
                assert np.trace(Bs[i]) == 0
            self.Bs = Bs
        self.cost_hist = np.zeros((max_iter, 2))
        if VERBOSE:
            print('W\tBs')
        start = time.time()
        for i in np.arange(max_iter):
            # Wt
            current_losses = np.ones((n_tasks, n_tasks))
            old_W = self.W.copy()
            if not DEBUG_G:
                self.W, current_losses = self._train_W(X, y, max_iter_wt)
            self.cost_hist[i, 0] = self._cost_function(X, y)

            # B
            old_Bs = self.Bs.copy()
            if not DEBUG_W:
                self.Bs = self._train_Bs(current_losses, max_iter_bgt)
            self.cost_hist[i, 1] = self._cost_function(X, y)

            # Convergence criteria
            delta_W = np.linalg.norm(old_W - self.W)
            delta_Bs = np.mean([np.linalg.norm(old_Bs[i, :, :] - self.Bs[i, :, :])
                                for i in np.arange(self.Bs.shape[0])])
            if VERBOSE:
                # print('{}\t{}'.format(delta_W, delta_Bs))
                print('{}\t{}'.format(self.cost_hist[i, 0],
                                      self.cost_hist[i, 1]))
            if delta_W < TOLERANCE_W and delta_Bs < TOLERANCE_Bs:
                if VERBOSE:
                    print('Convergence criterion has been met!')
                break
        return (self.W, self.cost_hist[:, 1], time.time() - start)

    def set_params(self, lambda_1, lambda_2, lambda_3):
        """
            Configura os parâmetros de execução do metodo.
            Args:
                lambda_1 (float): lambda of the task contribution weighted by
                    loss. Default: 1.
                lambda_2 (float): lambda of the projection on basis.
                    Default: 1.
                lambda_3 (float): lambda for the group lasso regularization.
                    Default: 1.
        """
        assert lambda_1 >= 0
        self.lambda_1 = lambda_1
        assert lambda_2 >= 0
        self.lambda_2 = lambda_2
        assert lambda_3 >= 0
        self.lambda_3 = lambda_3

    def get_params(self):
        """
        Retorna os parâmetros de execução do metodo.
        Return
            params (dict):
        """
        ret = {'lambda_1': self.lambda_1,
               'lambda_2': self.lambda_2,
               'lambda_3': self.lambda_3}
        return ret

    def get_resul(self):
        res = {'W': self.W,
               'Bs': self.Bs}
        return res

    def _cost_function(self, X, y):
        def __cost_glm(A, b, w):
            f = 0
            if self.glm == 'Gaussian':
                tt = np.dot(A, w) - b
                # nao é loglik mesmo, é só mse
                loglik = 0.5 * np.power(np.linalg.norm(tt), 2.0)
            elif self.glm == 'Poisson':
                # avoid overflow
                xb = np.maximum(np.minimum(np.dot(A, w), 100), -100)
                loglik = -(b * xb - np.exp(xb)).sum()
            elif self.glm == 'Gamma':
                loglik = 0
                for i in np.arange(0, A.shape[0]):
                    loglik += scipy.stats.gamma.logpdf(b[i],
                                                       1.0 / np.dot(A[i, :], w))
            elif self.glm == 'Binomial':
                # Xbeta = np.dot(A, w)
                # avoid overflow
                xb = np.maximum(np.minimum(np.dot(A, w), 100), -100)
                loglik = -1 * np.sum(((b * xb) - np.log(1 + np.exp(xb))))
            if not np.isnan(loglik):
                loglik /= float(A.shape[0])
                f += loglik
            else:
                print("****** WARNING: GroupAMTL loglik is nan.")
            return f

        def termo_1(A, b, w, t):
            temp_1 = 0
            if self.lambda_1 > 0:
                for g in np.arange(self.Bs.shape[0]):
                    temp_1 += np.linalg.norm(self.Bs[g][t], ord=1)
            if self.consider_loss:
                termo_1 = (1 + self.lambda_1 * temp_1) * __cost_glm(A, b, w)
            else:
                termo_1 = __cost_glm(A, b, w) + self.lambda_1 * temp_1
            return termo_1

        def termo_2(w, t):
            """diferença de w_t pra sua projeção.
            """
            termo_2 = 0
            if self.lambda_2 > 0:
                proj = np.zeros(len(w))
                for ind_g in np.arange(self.groups.shape[1]):
                    assert self.Bs[ind_g][t, t] == 0
                    group = np.arange(int(self.groups[0, ind_g]),
                                      int(self.groups[1, ind_g]))
                    proj[group] = np.dot(self.W[group, :],
                                         self.Bs[ind_g, :, t])
                termo_2 = (self.lambda_2 / 2) * np.linalg.norm(w - proj)**2
            return termo_2

        def reg(w):
            fgl = 0  # group lasso cost
            if self.lambda_3 > 0:
                ngroups = self.groups.shape[1]  # number of groups
                for i in np.arange(0, ngroups):
                    ids = np.arange(self.groups[0, i], self.groups[1, i],
                                    dtype=np.int)
                    fgl += self.groups[2, i] * np.linalg.norm(w[ids])
            return self.lambda_3 * fgl

        ret = 0
        for t, Xt in enumerate(X):
            t1 = termo_1(Xt, y[t], self.W[:, t], t)
            t2 = termo_2(self.W[:, t], t)
            t3 = reg(self.W[:, t])
            ret += t1 + t2 + t3
        return ret / len(X)

    def _train_W(self, X, y, max_iter_wt):
        W = self.W.copy()
        current_losses = np.ones((W.shape[1], W.shape[1]))
        for t in np.arange(W.shape[1]):
            opt_wt = Optimize_Wt(W=W, Bs=self.Bs, t=t, inds=self.groups,
                                 glm=self.glm,
                                 lambda_1=self.lambda_1,
                                 lambda_2=self.lambda_2,
                                 lambda_3=self.lambda_3,
                                 consider_loss=self.consider_loss,
                                 max_iter=max_iter_wt)
            w_opt = opt_wt.fit(W[:, t], X[t], y[t])
            W[:, t] = w_opt
            if self.consider_loss:
                if self.consider_cross_loss:
                    for s in np.arange(W.shape[1]):
                        current_losses[t, s] = \
                            opt_wt.cost_glm(X[s], y[s], W[:, t])
                else:
                    current_losses[t, :] = opt_wt.cost_glm(X[t], y[t], W[:, t])
        return (W, current_losses)

    def _train_Bs(self, current_losses, max_iter):
        n_tasks = self.W.shape[1]
        n_groups = self.groups.shape[1]
        Bs = self.Bs.copy()
        if self.lambda_2 > 0:
            reg_param = (2 * self.lambda_1) / self.lambda_2
        else:
            reg_param = 0
        for t in np.arange(n_tasks):
            all_but_t = list(range(self.W.shape[1]))
            all_but_t.pop(t)

            W_bar = self._get_w_bar(t)
            if self.consider_loss:
                weights = np.tile(current_losses[all_but_t, t], n_groups)
                # W_bar = np.divide(W_bar, weights)
                W_bar = (W_bar.T / weights[:, None]).T
            coeff = None
            if self.bias:
                all_but_t = all_but_t[:-1]
            w_t = self.W[:, t].copy()
            # w_t.shape = (w_t.shape[0], 1)

            print('B task being trained: {}'.format(t))
            if self.consider_B_restriction:
                if self.use_sk:
                    sk_lasso_cons = Lasso(alpha=reg_param, positive=True)
                    sk_lasso_cons.fit(W_bar, w_t)
                    coeff = sk_lasso_cons.coef_
                else:
                    lasso_cons = ADMM(reg_param,
                                      cache=self.cache,
                                      vary=self.vary,
                                      max_iter=max_iter,
                                      sparse=True)
                    coeff, _ = lasso_cons.fit(W_bar, w_t)
            else:
                if self.use_sk:
                    sk_lasso = Lasso(alpha=reg_param)
                    sk_lasso.fit(W_bar, w_t)
                    coeff = sk_lasso.coef_
                else:
                    lasso = ADMM_Lasso(lamb=reg_param,
                                       cache=self.cache,
                                       vary=self.vary,
                                       max_iter=max_iter)
                    coeff = lasso.fit(W_bar, w_t)
            if self.consider_loss:
                weights = np.tile(current_losses[all_but_t, t], n_groups)
                coeff = np.divide(coeff, weights)
            for ind_g in range(n_groups):
                start = (n_tasks - 1) * ind_g
                stop = start + (n_tasks - 1)
                Bs[ind_g][all_but_t, t] = coeff[start:stop]
        return Bs

    def _get_w_bar(self, t, sparse=True):
        blocks = list()
        for ind_g in np.arange(self.groups.shape[1]):
            sta = int(self.groups[0, ind_g].flatten())
            sto = int(self.groups[1, ind_g].flatten())
            group = np.arange(sta, sto)
            W_g = self.W[group, :].copy()
            all_but_t = list(range(self.W.shape[1]))
            all_but_t.pop(t)
            W_g = W_g[:, all_but_t]
            blocks.append(W_g)
        if sparse:
            W_bar = scipy.sparse.block_diag(blocks)
        else:
            W_bar = scipy.linalg.block_diag(*blocks)
        return W_bar


'''
class Optimize_Wt:
    """Optimize_Wt represents the problem of optimizing Wt.
    It contains all the methods and details needed by solve_fista to optimize
    the problem.
    """
    def __init__(self, W, Bs, t, inds, glm, lambda_1, lambda_2, lambda_3,
                 consider_loss, max_iter=MAX_ITER_WT):
        """
        Args:
            W (np.array): Weight matrix (n x t).
            Bs (list of np.arrays): G matrices of t x t.
            t (int): index of the task being optimized. Starts at 0.
            inds (np.array()): [[start g1, start g2, ..., start gG]
                                  [end g1, end g2, ..., end gG]
                                  [weight g1, weight g2, ... weight gG]]
                      Group representation. Each column of this matrix should
                      contain the start index, stop index, and group weight.
                      Recommended value = np.sqrt(end g - start g).
            glm (str): valid values: 'Gaussian', 'Poisson', 'Gamma'.
            lambda_1 (float): lambda of the task contribution weighted by loss.
            lambda_2 (float): lambda of the projection on basis.
            lambda_3 (float): lambda for the group lasso regularization.
            max_iter (int): maximum number of iterations of the min w_t
                step (FISTA).
        """
        assert W.shape[1] == Bs[0].shape[0]
        self.W = W
        self.Bs = Bs
        assert t >= 0
        assert t < W.shape[1]
        self.t = t
        self.inds = inds
        assert glm in ('Gaussian', 'Binomial')
        self.glm = glm
        assert lambda_1 >= 0
        self.lambda_1 = lambda_1
        assert lambda_2 >= 0
        self.lambda_2 = lambda_2
        assert lambda_3 >= 0
        self.lambda_3 = lambda_3
        assert isinstance(consider_loss, bool)
        self.consider_loss = consider_loss
        assert max_iter >= 0
        self.max_iter = max_iter
        self.mean = True
        groups = []
        for ind_g in np.arange(self.inds.shape[1]):
            group = np.arange(int(self.inds[0, ind_g]),
                              int(self.inds[1, ind_g]))
            groups.append(group)
        self.groups = groups

    def fit(self, w, X, y):
        """
        Uses optimization/fista.py to optimize Wt.
        See file for help.
        """
        fista = Fista(self, self.lambda_3)
        w_opt = fista.fit(w, X, y, self.inds,
                          max_iter=self.max_iter)
        return w_opt

    def cost(self, A, b, w):
        """Cost function.
            Args:
                A (np.array): design matrix.
                B (np.array): label vector.
                w (np.array): parameter vector for task self.t.
            Returns:
                f (float): cost function value at w.
        """
        def termo_1(w):
            f = 0
            custo = self.cost_glm(A, b, w)
            if self.lambda_1 > 0:
                norm_b = 0
                for g in np.arange(self.Bs.shape[0]):
                    norm_b += abs(self.Bs[g][self.t]).sum()
                if self.consider_loss:
                    f = (1 + self.lambda_1 * norm_b) * custo
                else:
                    f = custo + (self.lambda_1 * norm_b)
                return f
            else:
                return custo

        def termo_2(w):
            """diferença de w_t pra sua projeção.
            assume que não há sobreposição nos grupos!!!
            """
            termo_2 = 0
            if self.lambda_2 > 0:
                proj = np.zeros(len(w))
                for ind_g, group in enumerate(self.groups):
                    assert self.Bs[ind_g, self.t, self.t] == 0
                    proj[group] += np.dot(self.W[group, :],
                                          self.Bs[ind_g, :, self.t])
                termo_2 = (self.lambda_2 / 2) * np.power(np.linalg.norm(w - proj), 2)
            return termo_2

        def termo_3(w):
            """diferença de cada w_s(s!=t) para a respectiva projeção."""
            termo_3 = 0
            if self.lambda_2 > 0:
                for s, ws in enumerate(self.W.T):
                    if s == self.t:
                        continue
                    proj = np.zeros(len(w))
                    proj_w = np.zeros(self.W.shape[0])
                    for ind_g, group in enumerate(self.groups):
                        group = np.arange(int(self.inds[0, ind_g]),
                                          int(self.inds[1, ind_g]))
                        bzin = self.Bs[ind_g, :, s]
                        bzin[s] = 0.
                        bzin[self.t] = 0.
                        proj[group] += np.dot(self.W[group, :], bzin)
                        proj_w[group] = w[group] * self.Bs[ind_g, self.t, s]
                    termo_3 += ((ws - proj - proj_w) ** 2).sum()
                termo_3 = (self.lambda_2 / 2) * termo_3
            return termo_3
        ret = termo_1(w) + termo_2(w) + termo_3(w)
        return ret

    def cost_glm(self, A, b, w):
        """Cost function for the glm part.
            Args:
                A (np.array): design matriz.
                B (np.array): label vector.
                w (np.array): Weight vector for task self.t.
            Returns:
                f (float): log-likelihood according to the self.glm parameter.
        """
        loglik = 0
        if self.glm == 'Gaussian':
            tt = np.dot(A, w) - b
            loglik = 0.5 * np.linalg.norm(tt) ** 2.0
        elif self.glm == 'Poisson':
            # avoid overflow
            xb = np.maximum(np.minimum(np.dot(A, w), 100), -100)
            loglik = -(b * xb - np.exp(xb)).sum()
        elif self.glm == 'Gamma':
            loglik = 0
            for i in np.arange(0, A.shape[0]):
                loglik += scipy.stats.gamma.logpdf(b[i], 1.0 / np.dot(A[i, :], w))
        elif self.glm == 'Binomial':
            # avoid overflow
            ov_lim = 100
            Xbeta = np.maximum(np.minimum(np.dot(A, w), ov_lim), -ov_lim)
            loglik = -1 * np.sum(((b * Xbeta) - np.log(1 + np.exp(Xbeta))))
        if self.mean:
            loglik /= float(A.shape[0])
        if np.isnan(loglik):
            print("****** WARNING: loglik is nan.")
        return loglik

    def grad(self, A, b, w):
        """
            Computes the gradient of the cost function without the
            regularization part.

            Args:
                A (np.array): design matriz.
                B (np.array): label vector.
                w (np.array): weight vector.
            Returns:
                grad (np.array): gradient vector.
        """
        def grad_termo_1(w):
            termo_1 = self._grad_glm(A, b, w)
            if self.consider_loss:
                temp = 0
                if self.lambda_1 > 0:
                    for g in np.arange(self.Bs.shape[0]):
                        temp += (abs(self.Bs[g][self.t])).sum()
                constante = (1 + self.lambda_1 * temp)
                termo_1 = termo_1 * constante
            return termo_1

        def grad_termo_2(w):
            termo_2 = np.zeros(self.W.shape[0])
            if self.lambda_2 > 0:
                proj = np.zeros_like(termo_2)
                for ind_g, group in enumerate(self.groups):
                    assert self.Bs[ind_g, self.t, self.t] == 0
                    proj[group] += np.dot(self.W[group, :],
                                          self.Bs[ind_g, :, self.t])
                termo_2 = self.lambda_2 * (w - proj)
            return termo_2

        def grad_termo_3(w):
            termo_3 = np.zeros(self.W.shape[0])
            if self.lambda_2 > 0:
                for s, ws in enumerate(self.W.T):
                    if s == self.t:
                        continue
                    temp = np.zeros(w.shape)  # proj de w_s sem a parte de w_t
                    for ind_g, group in enumerate(self.groups):
                        bzin = self.Bs[ind_g, :, s]
                        bzin[s] = 0.
                        bzin[self.t] = 0.
                        proj_u = self.W[group, :].dot(bzin)
                        bts = self.Bs[ind_g, self.t, s]
                        temp[group] = bts * (bts * w[group] -
                                             (ws[group] - proj_u))
                    termo_3 += temp
                termo_3 = self.lambda_2 * termo_3
            return termo_3
        g1 = grad_termo_1(w)
        g2 = grad_termo_2(w)
        g3 = grad_termo_3(w)
        return g1 + g2 + g3

    def _grad_glm(self, A, b, w):
        """
            Computes the gradient of the glm part.
            Args:
                A (np.array): design matriz.
                B (np.array): label vector.
                w (np.array): Weight vector for task self.t.
            Returns:
                grad (np.array): gradient vector.
        """
        assert A.shape[0] == b.shape[0]
        tmp = np.zeros(w.shape)
        if self.glm == 'Gaussian':
            Xwmy = np.dot(A, w) - b
            tmp = np.dot(Xwmy, A)
        elif self.glm == 'Poisson':
            xb = np.maximum(np.minimum(np.dot(A, w), 200), -200)  # overflow
            tmp = -np.dot(A.T, b - np.exp(xb))
        elif self.glm == 'Gamma':
            kappa = 0.5
            tmp = kappa * np.dot(A.T, np.reciprocal(np.dot(A, w.T) - b))
        elif self.glm == 'Binomial':
            h = scipy.special.expit(np.dot(A, w))
            tmp = np.dot(A.T, h - b)
        if self.mean:
            tmp /= float(A.shape[0])
        return tmp
'''
