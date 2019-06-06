# -*- coding: utf-8 -*-
#pylint: disable=invalid-name, too-few-public-methods, too-many-arguments, too-many-locals
"""
File: gradient_descent.py
Author: yourname
Github: https://github.com/shgo
Description: Implements gradient descent with inexact backtracking line search for the
step size.
"""

import time
import numpy as np

MAX_ITER = 130
TOLERANCE = 10e-6
VERBOSE = False

class GradientDescent(object):
    """
    Fista will optimize a group lasso problem with the given cost_function,
    cost_grad, and prox in the object cv_fun.

    Args:
        alpha
        beta

    Attributes:
        history

    Methods:
        fit
    """
    def __init__(self, alpha=0.15, beta=0.8):
        assert alpha > 0
        assert alpha <= 0.5
        assert beta > 0
        assert beta <= 1
        self.alpha = alpha
        self.beta = beta
        self.history = {'obj_val': list(), 'step_vals': list()}

    def fit(self, X, y, cost, grad, w=None, max_iter=MAX_ITER):
        """Fitsssss
        Args:
            :X: Samples
            :y: Labels
            :cost: Cost function
            :grad: Grad of cost function
            :w: Initial weights (Default: None, which implies random)
            :max_iter: default set in MAX_ITER

        Returns:
            :returns: w
        """
        assert callable(cost)
        assert callable(grad)
        assert X.shape[0] == y.shape[0]
        assert max_iter > 3
        _, n = X.shape
        if w is None:
            w = np.random.randn(n)

        step = 1
        start = time.time()
        for k in range(max_iter):
            direction = -grad(X, y, w)
            # line search
            backtracking = True
            #print('start backtracking')
            while backtracking:
                left = cost(X, y, w + step * direction)
                right = cost(X, y, w) + self.alpha * step \
                        * np.dot(grad(X, y, w), direction)
                if left > right:
                    step *= self.beta
                else:
                    backtracking = False
            #print('end backtracking step = {}'.format(step))
            w = w + step * direction
            self.history['obj_val'].append(cost(X, y, w))
            norm_grad = np.linalg.norm(grad(X, y, w))
            if VERBOSE:
                print('Iter {} - obj: {:.4f} norm_grad: {:.12f}'.format(k, self.history['obj_val'][-1], norm_grad))
            if norm_grad < TOLERANCE:
                if VERBOSE:
                    print('BREAK')
                break
        elapsed = time.time() - start
        return w, elapsed
