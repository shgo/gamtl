# -*- coding: utf-8 -*-
#pylint: disable=invalid-name,missing-docstring
"""
Fista module. WARNING: the code is a litle coupled with agtml and
proximal_operator.
"""

import matplotlib.pyplot as plt
import numpy as np

import codes.optimization.proximal_operator as prxopt

MAX_ITER = 100
TOLERANCE = 1e-8
STOPPING_GROUND_TRUTH = True
STOPPING_DEFAULT = STOPPING_GROUND_TRUTH
VERBOSE = False
PLOT_COST = False


class Fista(object):
    """
    Fista will optimize a group lasso problem with the given cost_function,
    cost_grad, and prox in the object cv_fun.

    Args:
        cv_fun (class): Must have the following methods with signature:
            cost(A, b, w) computes the cost function.
            grad(A, b, w) computes the gradient of the cost function.
        lambda_gl (float): regularization parameter for the group lasso.
        eta (int): L update rate in Backtracking. Default = 2.

    Attributes:
        hist
        L

    Methods:
        fit

    References:
        [1] Liu et al, 2017, Modeling Alzheimer's Disease Cognitive Scores using
            Multi-task Sparse Group Lasso.
        [2] Beck, Teboulle, 2009, A Fast Iterative Shrinkage-Thresholding
            Algorithm.
    """

    def __init__(self, cv_fun, lambda_gl, L=1, eta=2):
        assert cv_fun is not None
        self.cv_fun = cv_fun
        self.hist = []
        self.lambda_gl = lambda_gl
        assert L >= 1
        self.L = L
        assert eta > 1
        self.eta = eta

    def fit(self, xk, A, b, ind, max_iter=MAX_ITER, tolerance=TOLERANCE):
        """
            Fita FISTA.
            Retorna FISTA fitado.

            Args:
                xk (np.array): parameter vector.
                A (list of lists): Dataset matrix.
                b (list of lists): Label vector.
                ind (np.array): Group list [[start, stop, weight] for each group].
                max_iter (int): maximum number of iterations.
                tolerance (float): if the norm of (xk - xkm1) is less than the
                    value of this parameter, the optimization ends.
            Returns:
                x (np.array): parameter vector optimized.
        """
        assert len(xk) == A.shape[1]
        assert b.shape == (A.shape[0], )
        assert max_iter > 1
        assert tolerance > 0

        # Initializing optimization variables
        xkm1 = xk
        yk = xk.copy()
        t_k = 1
        ind_temp = ind.copy()  # ind_temp may be changed during backtracking
        hist = list()

        # FISTA optimization loop
        keep_going = True

        # initial estimate for L: from FISTA paper
        # this is for lasso, but let's see how is it for other related problems
        #self.L = 2*np.real(np.linalg.eigvals(np.dot(A.T, A))).max()
        # another form of estimating Lipschitz constant
        # self.L = sp.linalg.norm(A) ** 2
        if PLOT_COST:
            flag = True
        nIter = 0
        while keep_going and (nIter < max_iter):
            grad_yk = self.cv_fun.grad(A, b, yk)
            if PLOT_COST or VERBOSE:
                hist_f = []
                hist_q = []
                hist_l = []
            self.L = 1
            # backtracking: adjust step-size at every iteration
            # Fixed step:
            # the effectiveness of this approach will dependent on how good is
            # our L estimate.
            while True:
                ind_temp[2, :] = ind[2, :] / self.L
                proj_yk = prxopt.proximal_group(yk - (1.0 / self.L) * grad_yk,
                                                ind_temp, self.lambda_gl)
                # como os dois adicionam g(yk) no final, não adicionei em nenhum
                F_proj_yk = self.cv_fun.cost(A, b, proj_yk)  #F
                cost_yk = self.cv_fun.cost(A, b, yk)  #Q
                ap_grad = np.dot((proj_yk - yk).T, grad_yk)  #Q
                ap_norm = (self.L / 2.0) * np.square(
                    np.linalg.norm(proj_yk - yk))  #Q
                Q_proj_yk = cost_yk + ap_grad + ap_norm  #Q
                if VERBOSE:
                    hist_f.append(F_proj_yk)
                    hist_q.append(Q_proj_yk)
                    text = 'BT: F(proj y): {:.8f} Q(proj y, y): {:.4f} L={}'.format(
                        F_proj_yk, Q_proj_yk, self.L)
                    print(text)
                if F_proj_yk <= Q_proj_yk:
                    if VERBOSE:
                        print('fim backtracking: %d' % (self.L))
                    break
                else:
                    self.L = self.L * self.eta
                    if PLOT_COST:
                        hist_l.append(self.L)
            if PLOT_COST and flag:
                flag = False
                plt.subplot(211)
                plt.plot(hist_f, label='F')
                plt.plot(hist_q, label='Q')
                plt.title('Backtracking F and Q')
                plt.legend()
                plt.subplot(212)
                plt.plot(hist_l)
                plt.title('L')
                plt.show(block=False)
            #atualiza variáveis
            xk = proj_yk.copy()
            t_kp1 = 0.5 * (1 + np.sqrt(1 + 4 * t_k * t_k))
            yk = xk + ((t_k - 1) / float(t_kp1)) * (xk - xkm1)
            t_k = t_kp1
            keep_going = np.linalg.norm(xkm1 - xk) > tolerance
            xkm1 = xk.copy()

            hist.append(self.cv_fun.cost(A, b, xk))
            if VERBOSE:
                up = ' '
                if nIter > 0:
                    if hist[-1] > hist[-2]:
                        up = '#'
                text = '{}cost: {:8.8f} L: {:4} lambda_gl: {:3.2f}{}'.format(
                    up, hist[-1], self.L, self.lambda_gl, up)
                print(text)
            nIter = nIter + 1
        self.hist = hist
        if PLOT_COST:
            plt.figure()
            plt.semilogy(self.hist)
            plt.title('Cost function')
            plt.show(block=False)
        return xk
