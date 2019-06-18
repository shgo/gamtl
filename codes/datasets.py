# -*- coding: utf-8 -*-
#pylint:disable=invalid-name,too-many-arguments
"""
This module implements Datasets defined on design.py.
"""
import numpy as np
from codes.design import ArtificialDataset

class ArtGAMTLRegression(ArtificialDataset):
    """
    Generates artificial data as following:

    Attributes:
        base (int): # of samples per task (will vary between base and
            1.1 * base.
        m (int): sampled # of samples per task
        n (int): 24 features for all tasks
        noise (float): parameter indicating the noise to be used.
        groups (np.array()): [[start g1, start g2, ..., start gG]
                                [end g1, end g2, ..., end gG]
                                [weight g1, weight g2, ... weight gG]]
                    Group representation. Each column of this matrix should
                    contain the start index, stop index, and group weight.
                    Recommended value = np.sqrt(end g - start g).
                    * groups = [[0, 12, sqrt(12)][12, 24, sqrt(12)]]
        T (int): 8
        split (tuple): (60%, 20%, 20%) number of samples in
            train, val and test sets.
        W (n x T): zeros.
        Bs (qtd_groups, T, T) : zeros.
            Seu formato é (m_tr, m_val, m_te) e sum(split) = m.
            Caso não seja informado, 90% dos dados vão para treinamento, e os
            10% restantes vão para o conjunto de teste.
    Methods:
        generate() - Generates data.
        get_train() - Returns training data. The portion argument is used to
        choose betweel 'all' training data, 'tr' portion, and 'val' portion.
        get_test() - Returns test data.
        get_params() - Returns the parameters of the dataset.
        set_params() - Sets the parameters of the dataset.
    """

    def __init__(self):
        """
            Attributes will be initialized as:
                base (int): 500. # of samples per task (will vary between base and
                    1.1 * base.
                m (int): sampled # of samples per task
                n (int): 24
                groups = [[0, 12, sqrt(12)][12, 24, sqrt(12)]]
                T (int): 18
                split (tuple): (60%, 20%, 20%) number of samples in
                    train, val and test sets.
                W (n x T): zeros.
                Bs (qtd_groups, T, T) : zeros.
                noise (float): np.Inf
        """
        T = 8
        split = 0.7
        n = 50
        m = 1000
        W = np.zeros((n, T))
        groups = np.array([[0, 25], [25, 50], [np.sqrt(25), np.sqrt(25)]])
        super().__init__('Art1Regression', m, n, T, split, W=W, groups=groups)
        self.noise = np.Inf
        self.Bs = np.zeros((2, T, T))
        self.var_W = 1
        self.var_B = 1
        self.noise = 0.3
        self.noise_diff = 0.9

    def generate(self):
        """
            Generates  dataset, that can be obtained by calling
            self.get_train(), and self.get_test().
            Populates:
                m
                W
                Bs
                X
                y
            Saves .mat file to be used by Matlab Methods.
        """
        grupo_1 = range(int(self.groups[0, 0]), int(self.groups[1, 0]))
        grupo_2 = range(int(self.groups[0, 1]), int(self.groups[1, 1]))
        # seed tasks
        self.W[grupo_1, 0] = np.random.normal(0, self.var_W, len(grupo_1))
        self.W[grupo_1, 1] = np.random.normal(0, self.var_W, len(grupo_1))
        self.W[grupo_2, 2] = np.random.normal(0, self.var_W, len(grupo_2))
        self.W[grupo_2, 3] = np.random.normal(0, self.var_W, len(grupo_2))
        # relationship among tasks
        self.Bs = [None, None]
        self.Bs[0] = np.zeros((self.T, self.T))
        self.Bs[1] = np.zeros((self.T, self.T))
        for t in range(4, self.T):
            self.Bs[0][0:2, t] = abs(np.random.randn(2) * self.var_B)
            self.Bs[0][0:2, t] = abs(np.random.randn(2) * self.var_B)
            self.Bs[1][2:4:, t] = abs(np.random.randn(2) * self.var_B)
            self.Bs[1][2:4:, t] = abs(np.random.randn(2) * self.var_B)
        self.Bs = np.array(self.Bs)
        # derived tasks
        for t in range(4, self.T):
            for ind_g in range(2):
                ind_grupo = range(
                    int(self.groups[ind_g, 0]), int(self.groups[ind_g, 1]))
                self.W[ind_grupo, t] = np.dot(self.W[ind_grupo, :],
                                              self.Bs[ind_g, :, t])
        # data generation
        X = [[] for i in range(self.T)]
        y = [[] for i in range(self.T)]
        for t in range(self.T):
            temp_X = np.random.randn(self.m[t], self.n)
            X[t] = temp_X
        self.X = X
        for t in range(self.T):
            temp_y = np.dot(X[t], self.W[:, t])
            if t >= 4:
                temp_y += np.random.randn(len(temp_y)) * \
                    np.sqrt(self.noise + self.noise_diff)
            else:
                temp_y += np.random.randn(len(temp_y)) * \
                    np.sqrt(self.noise)
            y[t] = temp_y
        self.y = y
        super().generate()

    def get_params(self):
        """
        Gets dataset params.

        Returns:
            params (dict): keys:
                'm' - #rows design matrix
                'n' - #cols design matrix
                'split' - how to split data
                'T' - number of tasks
        """
        res = super().get_params()
        res['noise_diff'] = self.noise_diff
        return res


    def set_params(self, m, n, T, split, noise_diff):
        """ Sets parameters of dataset.

        :param m: (list) number of samples for each task
        :param n: (int) number of attributes
        :param T: (int) number of tasks
        :param split: (float) percent for training set split
        :param noise_diff: (float) noise of last 4 tasks
        """
        super().set_params(m=m, n=n, T=T, split=split)
        assert noise_diff >= 0
        self.noise_diff = noise_diff
