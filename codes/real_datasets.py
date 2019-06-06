# -*- coding: utf-8 -*-
#pylint:disable=invalid-name,no-member,too-few-public-methods
"""
Real datasets for MTL.

"""
import os
import pickle as pkl

import numpy as np
import pandas as pd
import scipy.io

import codes
from codes.design import Dataset


class PD(Dataset):
    """
        Loads Parkinson's Disease (PD) dataset.

        Reference:
            TODO

        Attributes:
            base (int): # of samples per task (will vary between base and
                1.1 * base.
            m (int): sampled # of samples per task
            n (int): 24 features for all tasks
            noise (float): parameter indicating the noise to be used.
            T (int): 8
            split (tuple): (60%, 20%, 20%) number of samples in
                train, val and test sets.
            W (n x T): zeros.
            groups

        Methods
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
                m (int): sampled # of samples per task
                n (int): 2634
                T (int): 18
                split (tuple): not used, but for information (600, 156) number of samples in
                    train, test sets.
                W (n x T): zeros.
                noise (float): np.Inf
        """
        T = 15
        split = 0.7
        n = 37
        m = np.Inf
        super().__init__(name='PD', m=m, n=n, T=T, split=split)
        arquivo = os.path.dirname(codes.__file__) + '/dataset/dataPD.pkl'
        with open(arquivo, 'rb') as arq:
            loaded = pkl.load(arq)
            self.X = loaded['X']
            self.y = loaded['y']
            groups = loaded['groups']
            self.groups = groups
            self.task_names = loaded['task_names']
        self.means_y = []
        self.stds_y = []

    def generate(self):
        super().generate()
        for t in range(self.T):
            mean = np.mean(self.y[t][self.inds_tr[t]])
            std = np.std(self.y[t][self.inds_tr[t]])
            self.means_y.append(mean)
            self.stds_y.append(std)
            self.oldy = self.y.copy()
            y_tr = self.y[t][self.inds_tr[t]]
            self.y[t][self.inds_tr[t]] = (y_tr - mean) / std
            y_te = self.y[t][self.inds_te[t]]
            self.y[t][self.inds_te[t]] = (y_te - mean) / std


class NCEP(Dataset):
    """
        Loads NCEP dataset.

        Reference:
            TODO

        Attributes:
            base (int): # of samples per task (will vary between base and
                1.1 * base.
            m (int): sampled # of samples per task
            n (int): 24 features for all tasks
            noise (float): parameter indicating the noise to be used.
            T (int): 8
            split (tuple): (60%, 20%, 20%) number of samples in
                train, val and test sets.
            W (n x T): zeros.
            groups
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
                m (int): sampled # of samples per task
                n (int): 2634
                T (int): 18
                split (tuple): not used, but for information (600, 156) number of samples in
                    train, test sets.
                W (n x T): zeros.
                noise (float): np.Inf
        """
        T = 18
        split = 0.7
        n = 2634
        m = np.Inf
        super().__init__(name='NCEP', m=m, n=n, T=T, split=split)
        ## LEITURA DOS DADOS
        arquivo = os.path.dirname(codes.__file__) + '/dataset/dataNCEP.pkl'
        with open(arquivo, 'rb') as arq:
            loaded = pkl.load(arq)
            self.X = loaded['X']
            self.y = loaded['y']
            self.groups = loaded['groups']
            self.task_names = loaded['task_names']

    def generate(self):
        self.inds_tr = [list(range(0, 600)) for i in range(self.T)]
        self.inds_te = [list(range(600, 756)) for i in range(self.T)]


class HorseColicNew(Dataset):
    """
        Loads HorseColic dataset.

        Reference:
        This dataset was originally published by the UCI Machine Learning
        Database: http://archive.ics.uci.edu/ml/datasets/Horse+Colic

        Attributes:
            base (int): # of samples per task (will vary between base and
                1.1 * base.
            m (int): sampled # of samples per task
            n (int): 24 features for all tasks
            noise (float): parameter indicating the noise to be used.
            T (int): 8
            split (tuple): (60%, 20%, 20%) number of samples in
                train, val and test sets.
            W (n x T): zeros.
            groups
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
                T (int): 18
                split (tuple): (60%, 20%, 20%) number of samples in
                    train, val and test sets.
                W (n x T): zeros.
                noise (float): np.Inf
        """
        T = 2
        split = 0.7
        n = 40
        m = np.Inf
        super().__init__(name='HorseColic', m=m, n=n, T=T, split=split)
        ## LEITURA DOS DADOS
        arquivo = os.path.dirname(codes.__file__) + '/dataset/horse-colic_new.pkl'
        with open(arquivo, 'rb') as arq:
            X, y, groups = pkl.load(arq)
            self.X = X
            self.y = y
            self.groups = groups

    def generate(self):
        self.inds_tr = [list(range(0, 67)) for i in range(self.T)]
        self.inds_te = [list(range(67, 82)) for i in range(self.T)]


class HorseColic2Tasks(Dataset):
    """
        Loads HorseColic dataset.

        Reference:
        This dataset was originally published by the UCI Machine Learning
        Database: http://archive.ics.uci.edu/ml/datasets/Horse+Colic

        Attributes:
            base (int): # of samples per task (will vary between base and
                1.1 * base.
            m (int): sampled # of samples per task
            n (int): 24 features for all tasks
            noise (float): parameter indicating the noise to be used.
            T (int): 8
            split (tuple): (60%, 20%, 20%) number of samples in
                train, val and test sets.
            W (n x T): zeros.
            groups
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
                T (int): 18
                split (tuple): (60%, 20%, 20%) number of samples in
                    train, val and test sets.
                W (n x T): zeros.
                noise (float): np.Inf
        """
        T = 2
        split = 0.7
        n = 50
        m = np.Inf
        super().__init__(name='HorseColic 2 tasks', m=m, n=n, T=T, split=split)
        ## LEITURA DOS DADOS
        arquivo = os.path.dirname(codes.__file__) + '/dataset/horse-colic2tasks.pkl'
        with open(arquivo, 'rb') as arq:
            X, y, groups = pkl.load(arq)
            self.X = X
            self.y = y
            self.groups = groups

    def generate(self):
        self.inds_tr = [list(range(0, 67)) for i in range(self.T)]
        self.inds_te = [list(range(67, 82)) for i in range(self.T)]


class HorseColic2(Dataset):
    """
        Loads HorseColic dataset.

        Reference:
        This dataset was originally published by the UCI Machine Learning
        Database: http://archive.ics.uci.edu/ml/datasets/Horse+Colic

        Attributes:
            base (int): # of samples per task (will vary between base and
                1.1 * base.
            m (int): sampled # of samples per task
            n (int): 24 features for all tasks
            noise (float): parameter indicating the noise to be used.
            T (int): 8
            split (tuple): (60%, 20%, 20%) number of samples in
                train, val and test sets.
            W (n x T): zeros.
            groups
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
                T (int): 18
                split (tuple): (60%, 20%, 20%) number of samples in
                    train, val and test sets.
                W (n x T): zeros.
                noise (float): np.Inf
        """
        T = 4
        split = 0.7
        n = 49
        m = np.Inf
        super().__init__(name='HorseColic', m=m, n=n, T=T, split=split)
        ## LEITURA DOS DADOS
        arquivo = os.path.dirname(codes.__file__) + '/dataset/horse-colic2.pkl'
        with open(arquivo, 'rb') as arq:
            X, y, groups = pkl.load(arq)
            self.X = X
            self.y = y
            self.groups = groups

    def generate(self):
        self.inds_tr = [list(range(0, 32)) for i in range(self.T)]
        self.inds_te = [list(range(32, 41)) for i in range(self.T)]


class HorseColic(Dataset):
    """
        Loads HorseColic dataset.

        Reference:
        This dataset was originally published by the UCI Machine Learning
        Database: http://archive.ics.uci.edu/ml/datasets/Horse+Colic

        Attributes:
            base (int): # of samples per task (will vary between base and
                1.1 * base.
            m (int): sampled # of samples per task
            n (int): 24 features for all tasks
            noise (float): parameter indicating the noise to be used.
            T (int): 8
            split (tuple): (60%, 20%, 20%) number of samples in
                train, val and test sets.
            W (n x T): zeros.
            groups
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
                T (int): 18
                split (tuple): (60%, 20%, 20%) number of samples in
                    train, val and test sets.
                W (n x T): zeros.
                noise (float): np.Inf
        """
        T = 4
        split = 0.7
        n = 50
        m = np.Inf
        super().__init__(name='HorseColic', m=m, n=n, T=T, split=split)
        ## LEITURA DOS DADOS
        arquivo = os.path.dirname(codes.__file__) + '/dataset/horse-colic.pkl'
        with open(arquivo, 'rb') as arq:
            X, y, groups = pkl.load(arq)
            self.X = X
            self.y = y
            self.groups = groups

    def generate(self):
        self.inds_tr = [list(range(0, 67)) for i in range(self.T)]
        self.inds_te = [list(range(67, 82)) for i in range(self.T)]


class Column3C(Dataset):
    """
        Loads Column3C dataset.

        References:
        This dataset was originally published by the UCI Machine Learning
        Database http://archive.ics.uci.edu/ml/datasets/vertebral+column

        Attributes:
            base (int): # of samples per task (will vary between base and
                1.1 * base.
            m (int): sampled # of samples per task
            n (int): 24 features for all tasks
            noise (float): parameter indicating the noise to be used.
            T (int): 8
            split (tuple): (60%, 20%, 20%) number of samples in
                train, val and test sets.
            W (n x T): zeros.
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
                T (int): 18
                split (tuple): (60%, 20%, 20%) number of samples in
                    train, val and test sets.
                W (n x T): zeros.
                noise (float): np.Inf
        """
        T = 3
        split = 0.7
        n = 6
        m = np.Inf
        super().__init__(name='Column3C', m=m, n=n, T=T, split=split)
        filename = os.path.dirname(codes.__file__) + '/dataset/column_3C.pkl'
        self.X = list()
        self.y = list()
        with open(filename, 'rb') as arq:
            X, y = pkl.load(arq)
            for t in range(self.T):
                self.X.append(X[t].values)
                self.y.append(y[t].values)


class Landmine(Dataset):
    """
        Loads Landmine dataset.

        Reference:
        Available in http://www.ee.duke.edu/~lcarin/LandmineData.zip

        Attributes:
            base (int): # of samples per task (will vary between base and
                1.1 * base.
            m (int): sampled # of samples per task
            n (int): 24 features for all tasks
            noise (float): parameter indicating the noise to be used.
            T (int): 8
            split (tuple): (60%, 20%, 20%) number of samples in
                train, val and test sets.
            W (n x T): zeros.
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
                T (int): 18
                split (tuple): (60%, 20%, 20%) number of samples in
                    train, val and test sets.
                W (n x T): zeros.
                noise (float): np.Inf
        """
        T = 19
        split = 0.7
        n = 9
        m = np.Inf
        super().__init__(name='Landmine', m=m, n=n, T=T, split=split)
        filename = os.path.dirname(codes.__file__) + '/dataset/LandmineData_19.mat'
        self.X = list()
        self.y = list()
        data = scipy.io.loadmat(filename)
        for t in range(len(data['x'][0])):
            self.X.append(data['x'][0][t])
            temp_y = data['y'][0][t].flatten()
            self.y.append(temp_y)


class ADNI(Dataset):
    """
        Loads ADNI dataset.

        Attributes:
            base (int): # of samples per task (will vary between base and
                1.1 * base.
            m (int): sampled # of samples per task
            n (int): 24 features for all tasks
            noise (float): parameter indicating the noise to be used.
            T (int): 8
            split (tuple): (60%, 20%, 20%) number of samples in
                train, val and test sets.
            W (n x T): zeros.
            groups
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
                T (int): 18
                split (tuple): (60%, 20%, 20%) number of samples in
                    train, val and test sets.
                W (n x T): zeros.
                noise (float): np.Inf
        """
        T = 5
        split = 0.95
        n = 327
        m = np.Inf
        super().__init__(name='ADNI', m=m, n=n, T=T, split=split)
        # LEITURA DOS DADOS
        folder = os.path.dirname(codes.__file__) + '/dataset/adnidataset'
        dfAD = pd.read_csv('{}/Data_ROI_AD.txt'.format(folder),
                           delim_whitespace=True,
                           header=None,
                           index_col=False)
        dfCN = pd.read_csv('{}/Data_ROI_CN.txt'.format(folder),
                           delim_whitespace=True,
                           header=None,
                           index_col=False)
        dfLMCI = pd.read_csv('{}/Data_ROI_LMCI.txt'.format(folder),
                             delim_whitespace=True,
                             header=None,
                             index_col=False)
        X = pd.concat([dfAD, dfCN, dfLMCI])
        total = X.T[0:5].T.sum(axis=1)
        X = X.drop(labels=[0, 1, 2, 3, 4], axis=1)
        X = pd.concat([total, X], axis=1, ignore_index=True)
        X = (X - X.mean()) / X.std()  # standardization
        self.y = [X.pop(i).values for i in range(T)]
        m = [X.shape[0] for i in range(T)]
        self.X = [X.values for i in range(T)]
        self.inds_ad = np.arange(dfAD.shape[0])
        last = dfAD.shape[0]
        self.inds_cn = np.arange(last, last + dfCN.shape[0])
        last = last + dfCN.shape[0]
        self.inds_lmci = np.arange(last, last + dfLMCI.shape[0])
        with open('{}/ind.txt'.format(folder), 'r') as arq:
            inds = [int(i) for i in arq.read().splitlines()]
        groups = [[], [], []]
        for i in range(len(inds)-1):
            start = int(inds[i]-1)
            stop = int(inds[i+1]-1)
            groups[0].append(start)
            groups[1].append(stop)
            tamanho = int(inds[i+1]) - int(inds[i])
            groups[2].append(np.sqrt(tamanho))
            # print('start: {} stop: {} size: {}'.format(start, stop, tamanho))
        self.groups = np.array(groups)


class Spam(Dataset):
    """
        Loads Spam 15 users  dataset.

        Reference:
        Available in https://andreric.github.io/files/datasets/spam_15users_mtl.mat

        Attributes:
            base (int): # of samples per task (will vary between base and
                1.1 * base.
            m (int): sampled # of samples per task
            n (int): 24 features for all tasks
            noise (float): parameter indicating the noise to be used.
            T (int): 8
            split (tuple): (60%, 20%, 20%) number of samples in
                train, val and test sets.
            W (n x T): zeros.
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
                T (int): 18
                split (tuple): (60%, 20%, 20%) number of samples in
                    train, val and test sets.
                W (n x T): zeros.
                noise (float): np.Inf
        """
        T = 15
        split = 0.7
        n = 2000
        m = np.Inf
        super().__init__(name='Spam', m=m, n=n, T=T, split=split)
        filename = os.path.dirname(codes.__file__) + '/dataset/spam_15users_mtl.mat'
        self.X = list()
        self.y = list()
        data = scipy.io.loadmat(filename)
        for t in range(len(data['x'][0])):
            self.X.append(data['x'][0][t])
            temp_y = data['y'][0][t].flatten()
            self.y.append(temp_y)


class MNIST(Dataset):
    """
        Loads MNIST dataset.

        References:
        Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based learning
        applied to document recognition." Proceedings of the IEEE,
        86(11):2278-2324, November 1998. [on-line version]

        Attributes:
            base (int): # of samples per task (will vary between base and
                1.1 * base.
            m (int): sampled # of samples per task
            n (int): 24 features for all tasks
            noise (float): parameter indicating the noise to be used.
            T (int): 8
            split (tuple): (60%, 20%, 20%) number of samples in
                train, val and test sets.
            W (n x T): zeros.
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
                T (int): 18
                split (tuple): (60%, 20%, 20%) number of samples in
                    train, val and test sets.
                W (n x T): zeros.
                noise (float): np.Inf
        """
        T = 10
        split = 0.7
        n = 784
        m = np.Inf
        super().__init__(name='mnist', m=m, n=n, T=T, split=split)
        """
        filename = os.path.dirname(codes.__file__) + '/dataset/mnist.pkl.gz'
        data = gzip.open(filename, 'rb')
        X = list()
        y = list()
        for t in range(T):
            temp_y = (data.target == t).astype('int')
            X.append(data.data)
            y.append(temp_y)
        self.X = X
        self.y = y
        """


class Letter(Dataset):
    """
        Loads Letter dataset.

        Reference:
        Available in https://andreric.github.io/files/datasets/letter.mat

        Attributes:
            base (int): # of samples per task (will vary between base and
                1.1 * base.
            m (int): sampled # of samples per task
            n (int): 24 features for all tasks
            noise (float): parameter indicating the noise to be used.
            T (int): 8
            split (tuple): (60%, 20%, 20%) number of samples in
                train, val and test sets.
            W (n x T): zeros.
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
                T (int): 18
                split (tuple): (60%, 20%, 20%) number of samples in
                    train, val and test sets.
                W (n x T): zeros.
                noise (float): np.Inf
        """
        T = 8
        split = 0.7
        n = 128
        m = np.Inf
        super().__init__(name='Letter', m=m, n=n, T=T, split=split)
        filename = os.path.dirname(codes.__file__) + '/dataset/letter.mat'
        X = list()
        y = list()
        data = scipy.io.loadmat(filename)
        for t in range(len(data['x'][0])):
            X.append(data['x'][0][t])
            temp_y = data['y'][0][t].flatten()
            y.append(temp_y)
        self.X = X
        self.y = y


class Yale(Dataset):
    """
        Loads Yale dataset.

        Reference:
        Available in https://andreric.github.io/files/datasets/yale32_alltogether.mat

        Attributes:
            base (int): # of samples per task (will vary between base and
                1.1 * base.
            m (int): sampled # of samples per task
            n (int): 24 features for all tasks
            noise (float): parameter indicating the noise to be used.
            T (int): 8
            split (tuple): (60%, 20%, 20%) number of samples in
                train, val and test sets.
            W (n x T): zeros.
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
                T (int): 18
                split (tuple): (60%, 20%, 20%) number of samples in
                    train, val and test sets.
                W (n x T): zeros.
                noise (float): np.Inf
        """
        filename = os.path.dirname(codes.__file__) + '/dataset/yale32.mat'
        data = scipy.io.loadmat(filename)
        T = len(data['x'])
        split = 0.7
        n = data['x'][0][0].shape[1]
        m = np.Inf
        super().__init__(name='Yale32', m=m, n=n, T=T, split=split)
        X = list()
        y = list()
        for t in range(len(data['x'])):
            X.append(data['x'][t][0])
            temp_y = data['y'][t][0].flatten()
            y.append(temp_y)
        self.X = X
        self.y = y


class Fiveloc(Dataset):
    """
        Loads Fiveloc dataset.

        Reference:
        TODO
        Attributes:
            base (int): # of samples per task (will vary between base and
                1.1 * base.
            m (int): sampled # of samples per task
            n (int): 24 features for all tasks
            noise (float): parameter indicating the noise to be used.
            T (int): 8
            split (tuple): (60%, 20%, 20%) number of samples in
                train, val and test sets.
            W (n x T): zeros.
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
                T (int): 18
                split (tuple): (60%, 20%, 20%) number of samples in
                    train, val and test sets.
                W (n x T): zeros.
                noise (float): np.Inf
        """
        n = 10
        T = 25
        split = 0.7
        m = np.Inf
        super().__init__(name='Fiveloc', m=m, n=n, T=T, split=split)
        filename = os.path.dirname(codes.__file__) + '/dataset/fiveloc.mat'
        X = list()
        y = list()
        data = scipy.io.loadmat(filename)
        for t in range(len(data['x'][0])):
            X.append(data['x'][0][t])
            temp_y = data['y'][0][t].flatten()
            y.append(temp_y)
        self.X = X
        self.y = y


class NorthAmerica(Dataset):
    """
        Loads NorthAmerica dataset.

        References:
        Available in https://andreric.github.io/files/datasets/north_america.mat

        Attributes:
            base (int): # of samples per task (will vary between base and
                1.1 * base.
            m (int): sampled # of samples per task
            n (int): 24 features for all tasks
            noise (float): parameter indicating the noise to be used.
            T (int): 8
            split (tuple): (60%, 20%, 20%) number of samples in
                train, val and test sets.
            W (n x T): zeros.
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
                T (int): 18
                split (tuple): (60%, 20%, 20%) number of samples in
                    train, val and test sets.
                W (n x T): zeros.
                noise (float): np.Inf
        """
        T = 490
        split = 0.7
        n = 10
        m = np.Inf
        super().__init__(name='NorthAmerica', m=m, n=n, T=T, split=split)
        filename = os.path.dirname(codes.__file__) + '/dataset/north_america.mat'
        self.X = list()
        self.y = list()
        data = scipy.io.loadmat(filename)
        for t in range(len(data['x'][0])):
            self.X.append(data['x'][0][t])
            temp_y = data['y'][0][t].flatten()
            self.y.append(temp_y)


class SouthAmerica(Dataset):
    """
        Loads SouthAmerica dataset.

        Reference:
        Available in https://andreric.github.io/files/datasets/south_america.mat

        Attributes:
            base (int): # of samples per task (will vary between base and
                1.1 * base.
            m (int): sampled # of samples per task
            n (int): 24 features for all tasks
            noise (float): parameter indicating the noise to be used.
            T (int): 8
            split (tuple): (60%, 20%, 20%) number of samples in
                train, val and test sets.
            W (n x T): zeros.
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
                T (int): 18
                split (tuple): (60%, 20%, 20%) number of samples in
                    train, val and test sets.
                W (n x T): zeros.
                noise (float): np.Inf
        """
        T = 250
        split = 0.7
        n = 10
        m = np.Inf
        super().__init__(name='SouthAmerica', m=m, n=n, T=T, split=split)
        filename = os.path.dirname(codes.__file__) + '/dataset/south_america.mat'
        X = list()
        y = list()
        data = scipy.io.loadmat(filename)
        for t in range(len(data['x'][0])):
            X.append(data['x'][0][t])
            temp_y = data['y'][0][t].flatten()
            y.append(temp_y)
        self.X = X
        self.y = y
