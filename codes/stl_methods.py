# -*- coding: utf-8 -*-
#pylint: disable=arguments-differ, invalid-name
"""
Single Task Learning Methods for Multi Task Learning datasets.

"""
import time

import numpy as np
import scipy
from sklearn.linear_model import Lasso, LogisticRegression

from codes.design import Method, MethodClassification, MethodRegression
from codes.optimization.fista import Fista

MAX_ITER = 100

class LassoRegression(MethodRegression):
    """
    LASSO for Regression tasks.

    Args:
        name (str): name that will be used in Experiment results
        label (str): label that will be used in Experiment results
        normalize (bool): standardize data
        bias (bool): add bias term
        positive (bool): restrict to positive values. Default False.

    Attributes:
        W (np.array) = parameter matrix
        lambda_1 (float): regularization parameter

    Methods:
        fit
        get_params
        set_params
        get_resul

    Reference:
        Tibshirani, R. (1996). Regression shrinkage and selection via the
        lasso. J. Royal. Statist. Soc B., Vol. 58, No. 1, pages 267-288).
    """
    def __init__(self, name='Lasso', label='lasso', normalize=False, bias=False, positive=False):
        super().__init__(name, label, normalize, bias)
        self.W = None
        self.lambda_1 = None
        self.positive = positive

    def fit(self, X, y, max_iter=MAX_ITER):
        """
        Trains with supplied data.

        Args:
            :param X: each position contains the data of a task as an (m, n)
                np.array with data (rows are samples, cols are features).
            :param y: each position contains the labels of a task as an (m)
                np.array.
            :param max_iter: max number of iterations

        Returns:
            W (np.array): (n, T) array with estimated parameters of all tasks.
            cost (np.array): cost at the end of each iteration.
            time (float): number of seconds spent in training.
        """
        X = self.normalize_data(X)
        X = self.add_bias(X)
        n_tasks = len(X)
        n_feats = X[0].shape[1]
        W = np.random.randn(n_feats, n_tasks)
        start = time.time()
        cost_function = 0
        for t in range(n_tasks):
            #print('Training {} task with lasso regression'.format(t))
            lasso = Lasso(alpha=self.lambda_1, positive=self.positive, max_iter=max_iter)
            lasso.fit(X[t], y[t])
            W[:, t] = lasso.coef_
            cost_function += np.linalg.norm(np.dot(X[t], W[:, t]) - y[t]) \
                    + sum(abs(W[:, t]))
        stop = time.time() - start
        self.W = W
        return W, np.array([cost_function]), stop

    def get_params(self):
        """
        Returns used params.

        Contains:
            lambda_1: regularization parameter

        :returns: ret
        :rtype: dict
        """
        return {'lambda_1': self.lambda_1}

    def set_params(self, lambda_1):
        """
        Sets parameters of LASSO.

        :param lambda_1: regularization parameter
        """
        assert lambda_1 >= 0
        self.lambda_1 = lambda_1

    def get_resul(self):
        """ Returns the estimated variables of LASSO.
        Contains:
          W: parameter matrix

        :returns: res
        :rtype: dict
        """
        return {'W': self.W}


class LassoClassification(MethodClassification):
    """
    LASSO for Classification tasks.

    Args:
        name (str): name that will be used in Experiment results
        label (str): label that will be used in Experiment results
        normalize (bool): standardize data
        bias (bool): add bias term
        positive (bool): restrict to positive values. Default False.

    Attributes:
        W (np.array) = parameter matrix
        lambda_1 (float): regularization parameter

    Methods:
        fit
        get_params
        set_params
        get_resul

    Reference:
        Tibshirani, R. (1996). Regression shrinkage and selection via the
        lasso. J. Royal. Statist. Soc B., Vol. 58, No. 1, pages 267-288).
    """
    def __init__(self, name='Lasso', label='lasso', normalize=False, bias=False, positive=False):
        super().__init__(name, label, normalize, bias)
        self.W = None
        self.lambda_1 = None
        self.normalize = normalize
        self.positive = positive
        self.threshold = 0.5

    def fit(self, X, y, max_iter=MAX_ITER):
        """
        Trains with supplied data.

        Args:
            :param X: each position contains the data of a task as an (m, n)
                np.array with data (rows are samples, cols are features).
            :param y: each position contains the labels of a task as an (m)
                np.array.
            :param max_iter: max number of iterations

        Returns:
            W (np.array): (n, T) array with estimated parameters of all tasks.
            cost (np.array): cost at the end of each iteration.
            time (float): number of seconds spent in training.
        """
        X = self.normalize_data(X)
        X = self.add_bias(X)
        n_tasks = len(X)
        n_feats = X[0].shape[1]
        W = np.random.randn(n_feats, n_tasks)
        start = time.time()
        cost_function = 0
        for t in range(n_tasks):
            #print('Training {} task with lasso regression'.format(t))
            lasso = LogisticRegression(C=self.lambda_1,
                                       penalty='l1',
                                       max_iter=max_iter)
            lasso.fit(X[t], y[t])
            W[:, t] = lasso.coef_
        stop = time.time() - start
        self.W = W
        return W, np.array([cost_function]), stop

    def get_params(self):
        """
        Returns used params.

        Contains:
            lambda_1: regularization parameter

        :returns: ret
        :rtype: dict
        """
        return {'lambda_1': self.lambda_1}

    def set_params(self, lambda_1):
        """
        Sets parameters of LASSO.

        :param lambda_1: regularization parameter
        """
        assert lambda_1 >= 0
        self.lambda_1 = lambda_1

    def get_resul(self):
        """ Returns the estimated variables of LASSO.
        Contains:
          W: parameter matrix

        :returns: res
        :rtype: dict
        """
        return {'W': self.W}


class GroupLassoRegression(MethodRegression):
    """
    GroupLasso for regression tasks (STL).
    Uses FISTA to solve the group lasso problem independently in
    each task.

    Args:
        name (str): name that will be used in Experiment results
        label (str): label that will be used in Experiment results
        normalize (bool): standardize data
        bias (bool): add bias term
        groups (np.array()): [[start g1, start g2, ..., start gG]
                                [end g1, end g2, ..., end gG]
                                [weight g1, weight g2, ... weight gG]]
                    Group representation. Each column of this matrix should
                    contain the start index, stop index, and group weight.
                    Recommended value = np.sqrt(end g - start g).

    Attributes:
        W (np.array): Weight matrix (n x t).
        glm (str): valid values: 'Gaussian', 'Poisson', 'Gamma'.
        lambda_1 (float): lambda of the task contribution weighted by loss.

    Methods
        fit
        get_params
        set_params
        get_resul

    Reference:
        Obozinski, G., Jacob, L., & Vert, J. (2011). Group Lasso with Overlaps:
        the Latent Group Lasso approach. CoRR, abs/1110.0413.
    """
    def __init__(self, name='GroupLasso', label='glasso',
                 normalize=False, bias=False, groups=None):
        super().__init__(name, label, normalize, bias)
        assert groups is not None
        assert groups.shape[0] == 3
        assert groups.shape[1] > 0
        self.groups = groups
        self.glm = 'Gaussian'
        self.W = None
        self.mean = True
        self.lambda_1 = 1.0
        self.max_iter = 100

    def fit(self, X, y):
        """
        Trains with supplied data.

        Args:
            :param X: each position contains the data of a task as an (m, n)
                np.array with data (rows are samples, cols are features).
            :param y: each position contains the labels of a task as an (m)
                np.array.
            :param max_iter: max number of iterations

        Returns:
                W (np.array): (n, T) array with estimated parameters of all tasks.
                cost (np.array): cost at the end of each iteration.
                time (float): number of seconds spent in training.

            Uses optimization/solve_fista.py.
        """
        X = self.normalize_data(X)
        X = self.add_bias(X)
        n_tasks = len(X)
        n_feats = X[0].shape[1]
        W = np.random.randn(n_feats, n_tasks)
        cost_function = 0
        start = time.time()
        for t in range(n_tasks):
            fista = Fista(self, self.lambda_1)
            w_opt = fista.fit(W[:, t], X[t], y[t], self.groups,
                              max_iter=self.max_iter)
            W[:, t] = w_opt
            cost_function += self.cost(X[t], y[t], W[:, t])
        stop = time.time() - start
        self.W = W
        return W, np.array([cost_function]), stop

    def set_params(self, lambda_1):
        """
        Sets parameters of GroupLASSO.

        :param lambda_1: regularization parameter
        """
        assert lambda_1 >= 0
        self.lambda_1 = lambda_1

    def get_params(self):
        """
        Returns used params.

        Contains:
            lambda_1: regularization parameter

        :returns: ret
        :rtype: dict
        """
        ret = {'lambda_1': self.lambda_1}
        return ret

    def get_resul(self):
        """ Returns the estimated variables of GroupLASSO.
        Contains:
          W: parameter matrix

        :returns: res
        :rtype: dict
        """
        return {'W': self.W}

    def cost(self, A, b, w):
        """Cost function for the glm part.

        Args:
            :param A: design matrix
            :param B: label vector
            :param w: weight vector for task self.t.

        Returns:
            f (float): log-likelihood according to the self.glm parameter.
        """
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

            if self.mean:
                loglik /= float(A.shape[0])

        if not np.isnan(loglik):
            f += loglik
        else:
            print("****** WARNING: loglik is nan.")
        return f

    def grad(self, A, b, w):
        """
        Computes the gradient of the glm part.

        Args:
            :param A: design matrix
            :param B: label vector
            :param w: weight vector for task self.t.

        Returns:
            grad (np.array): gradient vector.
        """
        tmp = np.zeros(w.shape)
        kappa = 0.5

        wk = w
        if self.glm == 'Gaussian':
            Xwmy = np.dot(A, w) - b
            tmp = np.dot(Xwmy, A)
        elif self.glm == 'Poisson':
            xb = np.maximum(np.minimum(np.dot(A, w), 200), -200) #avoid overflow
            tmp = -np.dot(A.T, b - np.exp(xb))
        elif self.glm == 'Gamma':
            tmp = kappa * np.dot(A.T, np.reciprocal(np.dot(A, wk.T) - b))
        elif self.glm == 'Binomial':
            Xbeta = np.dot(A, w)
            pi = np.reciprocal(1.0 + np.exp(-Xbeta))
            tmp = -np.dot(A.T, (b.flatten() - pi.flatten()))
            if self.mean:
                tmp *= 1.0 / float(A.shape[0])
        return tmp


class GroupLassoClassification(MethodClassification):
    """
    GroupLasso for classification tasks (STL).
    Uses FISTA to solve the group lasso problem independently in
    each task.

    Args:
        name (str): name that will be used in Experiment results
        label (str): label that will be used in Experiment results
        normalize (bool): standardize data
        bias (bool): add bias term
        groups (np.array()): [[start g1, start g2, ..., start gG]
                                [end g1, end g2, ..., end gG]
                                [weight g1, weight g2, ... weight gG]]
                    Group representation. Each column of this matrix should
                    contain the start index, stop index, and group weight.
                    Recommended value = np.sqrt(end g - start g).
        threshold (float): default 0.5, threshold for classification.

    Attributes:
        W (np.array): Weight matrix (n x t).
        glm (str): valid values: 'Gaussian', 'Poisson', 'Gamma'.
        lambda_1 (float): lambda of the task contribution weighted by loss.

    Methods
        fit
        get_params
        set_params
        get_resul

    Reference:
        Obozinski, G., Jacob, L., & Vert, J. (2011). Group Lasso with Overlaps:
        the Latent Group Lasso approach. CoRR, abs/1110.0413.
    """
    def __init__(self, name='GroupLasso', label='glasso', normalize=False,
                 bias=False, groups=None, threshold=0.5):
        super().__init__(name, label, normalize, bias)
        self.glm = 'Binomial'
        self.W = None
        assert groups is not None
        assert groups.shape[0] == 3
        assert groups.shape[1] > 0
        self.groups = groups
        self.mean = True
        self.lambda_1 = 1.0
        self.max_iter = 100
        self.threshold = threshold

    def fit(self, X, y):
        """
        Trains with supplied data.

        Args:
            :param X: each position contains the data of a task as an (m, n)
                np.array with data (rows are samples, cols are features).
            :param y: each position contains the labels of a task as an (m)
                np.array.

        Returns:
                W (np.array): (n, T) array with estimated parameters of all tasks.
                cost (np.array): cost at the end of each iteration.
                time (float): number of seconds spent in training.

            Uses optimization/solve_fista.py.
        """
        X = self.normalize_data(X)
        X = self.add_bias(X)
        n_tasks = len(X)
        n_feats = X[0].shape[1]
        W = np.random.randn(n_feats, n_tasks)
        cost_function = 0
        start = time.time()
        for t in range(n_tasks):
            #print('Training task {} with group lasso'.format(t))
            fista = Fista(self, self.lambda_1)
            w_opt = fista.fit(W[:, t], X[t], y[t], self.groups,
                              max_iter=self.max_iter)
            W[:, t] = w_opt
            cost_function += self.cost(X[t], y[t], W[:, t])
        stop = time.time() - start
        self.W = W
        return W, np.array([cost_function]), stop

    def get_params(self):
        """
        Returns used params.

        Contains:
            lambda_1: regularization parameter

        :returns: ret
        :rtype: dict
        """
        return {'lambda_1': self.lambda_1}

    def get_resul(self):
        """ Returns the estimated variables of GroupLASSO.
        Contains:
          W: parameter matrix

        :returns: res
        :rtype: dict
        """
        return {'W': self.W}

    def set_params(self, lambda_1):
        """
        Sets parameters of GroupLASSO.

        :param lambda_1: regularization parameter
        """
        assert lambda_1 >= 0
        self.lambda_1 = lambda_1

    def cost(self, A, b, w):
        """Cost function for the glm part.

        Args:
            :param A: design matrix
            :param B: label vector
            :param w: weight vector for task self.t.

        Returns:
            f (float): log-likelihood according to the self.glm parameter.
        """
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
            xb = np.maximum(np.minimum(np.dot(A, w), 100), -100)#avoid overflow
            loglik = -1 * np.sum(((b * xb) - np.log(1 + np.exp(xb))))
        if self.mean:
            loglik /= float(A.shape[0])
        if not np.isnan(loglik):
            f += loglik
        else:
            print("****** WARNING: loglik is nan.")
        return f

    def grad(self, A, b, w):
        """
        Computes the gradient of the glm part.

        Args:
            :param A: design matrix
            :param B: label vector
            :param w: weight vector for task self.t.

        Returns:
            grad (np.array): gradient vector.
        """
        tmp = np.zeros(w.shape)
        kappa = 0.5
        wk = w
        if self.glm == 'Gaussian':
            Xwmy = np.dot(A, w) - b
            tmp = np.dot(Xwmy, A)
        elif self.glm == 'Poisson':
            xb = np.maximum(np.minimum(np.dot(A, w), 200), -200) #avoid overflow
            tmp = -np.dot(A.T, b - np.exp(xb))
        elif self.glm == 'Gamma':
            tmp = kappa * np.dot(A.T, np.reciprocal(np.dot(A, wk.T) - b))
        elif self.glm == 'Binomial':
            Xbeta = np.dot(A, w)
            pi = scipy.special.expit(Xbeta)
            #pi = np.reciprocal(1.0 + np.exp(-Xbeta))
            tmp = -np.dot(A.T, (b.flatten() - pi.flatten()))
            if self.mean:
                tmp *= 1.0 / float(A.shape[0])
        return tmp


class TestOpt(Method):
    """
    Lasso for classification. Internal purposes.

    Args:
        groups (np.array()): [[start g1, start g2, ..., start gG]
                                [end g1, end g2, ..., end gG]
                                [weight g1, weight g2, ... weight gG]]
                    Group representation. Each column of this matrix should
                    contain the start index, stop index, and group weight.
                    Recommended value = np.sqrt(end g - start g).
        normalize (bool): standardize data

    Attributes:
        W (np.array): Weight matrix (n x t).
        glm (str): valid values: 'Gaussian', 'Poisson', 'Gamma'.
        lambda_1 (float): lambda of the task contribution weighted by loss.
        threshold (float): default 0.5, threshold for classification.

    Methods
        fit
        get_params
        set_params
        get_resul
    """
    def __init__(self, groups, normalize=False):
        super().__init__('LassoSTL', 'Lasso')
        self.W = None
        self.lambda_1 = None
        self.normalize = normalize
        self.threshold = 0.5
        self.glm = 'Binomial'
        self.groups = groups
        self.mean = True

    def get_params(self):
        return {'lambda_1': self.lambda_1}

    def set_params(self, lambda_1):
        assert lambda_1 >= 0
        self.lambda_1 = lambda_1

    def get_resul(self):
        return {'W': self.W}

    def fit(self, X, y, max_iter=MAX_ITER):
        """
        Trains with supplied data.

        Args:
            :param X: each position contains the data of a task as an (m, n)
                np.array with data (rows are samples, cols are features).
            :param y: each position contains the labels of a task as an (m)
                np.array.

        Returns:
                W (np.array): (n, T) array with estimated parameters of all tasks.
                cost (np.array): cost at the end of each iteration.
                time (float): number of seconds spent in training.
        """
        n_tasks = len(X)
        n_feats = X[0].shape[1]
        W = np.random.randn(n_feats, n_tasks)
        start = time.time()
        cost_function = 0
        X = self.normalize_data(X)
        X = self.add_bias(X)
        for t in range(n_tasks):
            #print('Training {} task with lasso regression'.format(t))
            lasso = Fista(self, self.lambda_1)
            w = lasso.fit(xk=W[:, t], A=X[t], b=y[t], ind=self.groups,
                          max_iter=max_iter)
            W[:, t] = w
        stop = time.time() - start
        self.W = W
        return W, np.array([cost_function]), stop

    def cost(self, A, b, w):
        """Cost function for the glm part.

        Args:
            :param A: design matrix
            :param B: label vector
            :param w: weight vector for task self.t.

        Returns:
            f (float): log-likelihood according to the self.glm parameter.
        """
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
            for i in np.arange(0, A.shape[0]):
                loglik += scipy.stats.gamma.logpdf(b[i], 1.0 / np.dot(A[i, :], w))
        elif self.glm == 'Binomial':
            ov_lim = 50
            Xbeta = np.maximum(np.minimum(np.dot(A, w), ov_lim), -ov_lim)#avoid overflow
            loglik = -1 * np.sum(((b * Xbeta) - np.log(1 + np.exp(Xbeta))))
        if self.mean:
            loglik /= float(A.shape[0])
        if not np.isnan(loglik):
            f += loglik
        else:
            print("****** WARNING: loglik is nan.")
        return f

    def grad(self, A, b, w):
        """
        Computes the gradient of the glm part.

        Args:
            :param A: design matrix
            :param B: label vector
            :param w: weight vector for task self.t.

        Returns:
            grad (np.array): gradient vector.
        """
        assert A.shape[0] == b.shape[0]
        tmp = np.zeros(w.shape)
        if self.glm == 'Gaussian':
            Xwmy = np.dot(A, w) - b
            tmp = np.dot(Xwmy, A)
        elif self.glm == 'Poisson':
            xb = np.maximum(np.minimum(np.dot(A, w), 200), -200) #avoid overflow
            tmp = -np.dot(A.T, b - np.exp(xb))
        elif self.glm == 'Gamma':
            kappa = 0.5
            tmp = kappa * np.dot(A.T, np.reciprocal(np.dot(A, w.T) - b))
        elif self.glm == 'Binomial':
            z = np.dot(A, w)
            h = scipy.special.expit(z)
            tmp = np.dot(A.T, h - b)
        if self.mean:
            tmp *= 1.0 / float(A.shape[0])
        return tmp

    def predict(self, X):
        """
        Predicts input data.

        Args:
            :param X: each position contains the data of a task as an (m, n)
                np.array with data (rows are samples, cols are features).

        Returns:
            y (np.array): predictions.
        """
        X = self.add_bias(X)
        y = []
        for t, Xt in enumerate(X):
            Xt = self.normalize_in(Xt, t)
            z = np.dot(Xt, self.W[:, t])
            h = scipy.special.expit(z)
            y_temp = (h >= self.threshold).astype('int')
            y.append(y_temp)
        return y
