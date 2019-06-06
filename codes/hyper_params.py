#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint:disable=too-many-instance-attributes,too-many-arguments
"""
Implements methods to find hyper-parameters, such as Cross Validation.
"""
import os
import copy
import pickle as pkl
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from codes.design import HyperParameterization
from codes.metrics import nmse
PREFIX = os.path.dirname(__file__)


class CrossValidation(HyperParameterization):
    """
    Cross Validation.

    Args:
        dataset (ArtificialDatasetMTL, RealDatasetMTL): dataset to be
            used.
        split (float): pct of train data to use for validation.
        metric (name, func): metric to be used.
        bb (bool): bigger is better flag.
        permutation (bool): permute samples before split

    Attributes:
        inds (list of lists): inds of all samples for all tasks
        inds_tr (list of lists): inds of training samples for all tasks
        inds_val (list of lists): inds of validation samples for all tasks

    Methods:
        _generate_inds
        get_train
        get_val
        fit
        get_params
        set_params
    """
    def __init__(self, dataset, split=0.7, metric=('nmse', nmse), bb=False,
                 permutation=True):
        super().__init__(dataset, metric, bb)
        self.data = self.dataset.get_train()
        T = len(self.data['X'])
        self.T = T
        self.split = split
        self.inds = [[] for i in range(T)]
        self.inds_tr = [[] for i in range(T)]
        self.inds_val = [[] for i in range(T)]
        self.permutation = permutation
        self._generate_inds()

    def _generate_inds(self):
        """ Generates the proper division in the dataset."""
        ms = [self.data['X'][t].shape[0] for t in range(self.T)]
        for ind_task, m_val in enumerate(ms):
            if self.permutation:
                inds = np.random.permutation(m_val)
                self.inds[ind_task] = inds
            else:
                self.inds[ind_task] = list(range(m_val))
            end_tr = round(self.split * m_val)
            self.inds_tr[ind_task] = self.inds[ind_task][:end_tr]
            self.inds_val[ind_task] = self.inds[ind_task][end_tr:]

    def get_train(self):
        """
            Returns X, y training data of all tasks.

        :returns: ret
        :rtype: dict
        """
        ret = dict()
        ret['X'] = list()
        ret['y'] = list()
        for t in range(self.T):
            inds = self.inds_tr[t]
            ret['X'].append(self.data['X'][t][inds])
            ret['y'].append(self.data['y'][t][inds])
        return ret

    def get_val(self):
        """
            Returns X, y validation data of all tasks.

        :returns: ret
        :rtype: dict
        """
        ret = dict()
        ret['X'] = list()
        ret['y'] = list()
        for t in range(self.T):
            inds = self.inds_val[t]
            ret['X'].append(self.data['X'][t][inds])
            ret['y'].append(self.data['y'][t][inds])
        return ret

    def fit(self, method_ref, init_params, params, n_jobs=1):
        """Runs procedure to find hyper parameters.

        Args:
            :param method_ref: method to run
            :param init_params: method initialization parameters
            :param params: grid of parameters
            :param n_jobs: number of cores to use. Default: -1, which will use
                all possible cores.

        Returns:
            :returns: best_params
            :rtype: dict
            :returns: df_res
            :rtype: pd.DataFrame

        """
        print('\t CROSS-VALIDATION {}'.format(method_ref.__name__))
        if self.bb:
            best_cost = -np.Inf
        else:
            best_cost = np.Inf
        df_res = pd.DataFrame()
        best_params = None
        results = list()
        if n_jobs == 1:
            for param in params:
                results.append(self._fit_param(method_ref, init_params, param))
        else:
            results = Parallel(n_jobs=n_jobs)(delayed(self._fit_param)(method_ref,
                                              init_params, param) for param in params)
        for result, param in results:
            df_res = df_res.append(result, ignore_index=True)
            f_tr = result['tr']
            f_val = result['val']
            print('\t\t metric tr: {:.3f} metric val: {:.3f}'.format(f_tr, f_val))
            if self.bb:
                if f_val > best_cost:
                    best_cost = f_val
                    best_params = copy.copy(param)
            else:
                if f_val < best_cost:
                    best_cost = f_val
                    best_params = copy.copy(param)
        print('\t Best params found:')
        print('\t {}'.format(best_params))
        return best_params, df_res

    def get_params(self):
        """ Returns method parameters.

        Returns:
            :returns: ret
            :rtype: dict
        """
        ret = dict()
        ret['split'] = self.split
        ret['metric'] = self.metric
        ret['bb'] = self.bb
        return ret

    def set_params(self, split, metric, bb):
        """Set parameters of Cross-Validation.

        Args:
            :param split: pct of samples to be in the training set
            :param metric: name, func
            :param bb: bigger is better flag
        """
        assert isinstance(split, float)
        self.split = split
        assert callable(metric)
        self.metric = metric
        assert isinstance(bb, bool)
        self.bb = bb


class KFold(HyperParameterization):
    """ K-Fold Cross Validation.

    Args:
        dataset (ArtificialDatasetMTL, RealDatasetMTL): dataset to be
            used.
        n_folds (int): number of folds
        metric (name, func): metric to be used
        bb (bool): bigger is better flag
        permutation (bool): permute samples before split

    Attributes:
        inds (list of lists): inds of all samples for all tasks
        inds_tr (list of lists): inds of training samples for all tasks
        inds_val (list of lists): inds of validation samples for all tasks

    Methods:
        _generate_folds
        set_fold
        get_train
        get_val
        fit
        get_params
        set_params
    """
    def __init__(self, dataset, n_folds=5, metric=('nmse', nmse), bb=False,
                 permutation=False):
        super().__init__(dataset, metric, bb)
        self.data = self.dataset.get_train()
        T = len(self.data['X'])
        self.T = T
        assert isinstance(n_folds, int)
        assert n_folds > 0
        self.n_folds = n_folds
        assert isinstance(permutation, bool)
        self.permutation = permutation
        self.inds = range(self.data['X'][0].shape[0])
        self._generate_folds()
        self.inds_tr = None
        self.inds_val = None

    def _generate_folds(self):
        """ Generates the proper division in the folds."""
        self.inds_folds = \
            [self.inds[i::self.n_folds] for i in range(self.n_folds)]

    def set_fold(self, k):
        """ sets actual fold. """
        assert isinstance(k, int)
        but_k = [i for i in range(self.n_folds) if i != k]
        self.inds_tr = np.concatenate([self.inds_folds[i] for i in but_k])
        self.inds_val = self.inds_folds[k]

    def get_train(self):
        """
            Returns X, y training data of all tasks.

        :returns: ret
        :rtype: dict
        """
        ret = dict()
        ret['X'] = list()
        ret['y'] = list()
        for t in range(self.T):
            inds = self.inds_tr
            ret['X'].append(self.data['X'][t][inds])
            ret['y'].append(self.data['y'][t][inds])
        return ret

    def get_val(self):
        """
            Returns X, y validation data of all tasks.

        :returns: ret
        :rtype: dict
        """
        ret = dict()
        ret['X'] = list()
        ret['y'] = list()
        for t in range(self.T):
            inds = self.inds_val
            ret['X'].append(self.data['X'][t][inds])
            ret['y'].append(self.data['y'][t][inds])
        return ret

    def fit(self, method_ref, init_params, params, n_jobs=1):
        """Runs procedure to find hyper parameters.

        Args:
            :param method_ref: method to run
            :param init_params: method initialization parameters
            :param params: grid of parameters
            :param n_jobs: number of cores to use. Default: -1, which will use
                all possible cores.

        Returns:
            :returns: best_params
            :rtype: dict
            :returns: df_res
            :rtype: pd.DataFrame

        """
        print('\t K-FOLD CROSS-VALIDATION {}'.format(method_ref.__name__))
        if self.bb:
            best_cost = -np.Inf
        else:
            best_cost = np.Inf
        df_res = pd.DataFrame()
        best_params = None
        results = list()
        for fold in range(self.n_folds):
            print('FOLD {}'.format(fold))
            self.set_fold(fold)
            if n_jobs == 1:
                for param in params:
                    results.append(self._fit_param(method_ref, init_params, param))
            else:
                results = Parallel(n_jobs=n_jobs)(delayed(self._fit_param)(method_ref,
                                                init_params, param) for param in params)
            for result, param in results:
                result['fold'] = fold
                df_res = df_res.append(result, ignore_index=True)
                f_tr = result['tr']
                f_val = result['val']
                print('\t\t fold: {} tr: {:.5f} val: {:.5f}'.format(fold, f_tr, f_val))
                if self.bb:
                    if f_val > best_cost:
                        best_cost = f_val
                        best_params = copy.copy(param)
                else:
                    if f_val < best_cost:
                        best_cost = f_val
                        best_params = copy.copy(param)
        print('\tBest params found:')
        print(best_params)
        return best_params, df_res

    def get_params(self):
        """ Returns method parameters.

        Returns:
            :returns: ret
            :rtype: dict
        """
        ret = dict()
        ret['n_folds'] = self.n_folds
        ret['metric'] = self.metric
        ret['bb'] = self.bb
        return ret

    def set_params(self, n_folds, metric, bb):
        """Set parameters of Cross-Validation.

        Args:
            :param n_folds: number of folds in training set
            :param metric: name, func
            :param bb: bigger is better flag
        """
        assert isinstance(n_folds, int)
        assert n_folds > 0
        self.n_folds = n_folds
        assert callable(metric)
        self.metric = metric
        assert isinstance(bb, bool)
        self.bb = bb


class StabilitySelection():
    """
    Stability Selection procedure from [1].

    [1] - Nicolai Meinshausen and Peter Bühlmann, “Stability selection (with
    discussion and rejoinder by the authors),” Journal of the Royal
    Statistical Society: Series B (Statistical Methodology), vol. 72,
    no. 4, pp. 417--473, 2010 [Online].

    Args:
        dataset (ArtificialDatasetMTL, RealDatasetMTL): dataset to be
            used.
        size (float): pct of data to use in sub-sampling.
        runs (int): number of runs.

    Attributes:
        data (dict): copies data from dataset.

    Methods:
        fit - fits method
        get_samples - samples data
        transform - apply threshold and save results
        save - saves object on disk
    """
    def __init__(self, dataset, filename, size=0.5, runs=100):
        assert size > 0
        assert size < 1
        self.size = size
        assert runs > 0
        self.runs = runs
        assert isinstance(filename, str)
        if not os.path.exists('results'):
            os.makedirs('results')
        self.filename = 'results/{}'.format(filename)
        data = dataset()
        data.generate()
        self.data = {}
        self.data['X'] = data.X
        self.data['y'] = data.y
        self.T = len(self.data['X'])
        self.resul_inds = {}
        self.resul = {}
        self.resul_mats = {}
        self._check_resul()

    def _check_resul(self):
        """ Deletes method from the results

        :param method_name: name of method in Strategies
        """
        if os.path.isfile(self.filename):
            print("Já existe, [p]arar, [c]ontinuar ou [r]esetar? (C/r)")
            aqui = input()
            if aqui in ('p', 'P'):
                os._exit(1)
            elif aqui in ('r', 'R'):
                os.remove(self.filename)
            elif aqui in ('c', 'C'):
                with open(self.filename, 'rb') as arq:
                    temp = pkl.load(arq)
                    self.resul = temp.resul
                    self.resul_inds = temp.resul_inds
                    self.resul_mats = temp.resul_mats

    def fit(self, method_ref, init_params, params, n_jobs=-1):
        """Runs procedure to find stable set.

        Args:
            :param method_ref: method to run
            :param init_params: method initialization parameters
            :param params: grid of parameters
            :param n_jobs: number of cores to use. Default: -1, which will use
                all possible cores.

        Returns:
            :returns: mask
            :rtype: dict
            :returns: res_matrix
            :rtype: pd.DataFrame

        """
        met = method_ref(**init_params)
        print('STABILITY SELECTION FOR {}'.format(met.name.upper()))
        if met.name in self.resul.keys():
            print('Já executou runs...')
        else:
            self.resul[met.name] = []
            self.resul_inds[met.name] = []
            for param in params:
                results = []
                if n_jobs == 1:
                    print('One at a time...')
                    for run in range(self.runs):
                        results.append(self._one_run(method_ref, init_params,
                                                     param, run))
                else:
                    print('The power of multiple cores! n_jobs:', n_jobs)
                    results = Parallel(n_jobs=n_jobs)(delayed(self._one_run)(
                        method_ref, init_params, param, run)
                                                      for run in range(self.runs))
                for row, res in results:
                    self.resul_inds[met.name].append(row)
                    self.resul[met.name].append(res)
                self.save()
        if met.name in self.resul_mats.keys():
            print('Já processou...')
        else:
            res_matrix = self._transform(met.name)
            self.resul_mats[met.name] = res_matrix
            self.save()
        print("FIM")

    def _one_run(self, method_ref, init_params, param, run):
        row = pd.Series(param)
        row['run'] = run
        data = self.get_samples()
        method = method_ref(**init_params)
        method.set_params(**param)
        print('.', end='', flush=True)
        method.fit(**data)
        res = method.get_resul()
        return row, res

    def get_samples(self):
        """ Generates the proper division in the dataset."""
        ms = [self.data['X'][t].shape[0] for t in range(self.T)]
        data = {'X': [], 'y': []}
        for ind_task, m_val in enumerate(ms):
            inds = np.random.permutation(m_val)
            data['X'].append(self.data['X'][ind_task][inds])
            data['y'].append(self.data['y'][ind_task][inds])
        return data

    def _transform(self, method_name):
        samples = []
        res = self.resul[method_name]
        for row, ex in enumerate(res):
            sample = self.resul_inds[method_name][row].to_numpy()
            sample = np.concatenate(sample, ex['W'].flatten())
            for B in ex['Bs']:
                sample = np.concatenate((sample, B.flatten()))
            samples.append(sample)
        res_matrix = pd.DataFrame(data=samples)
        return res_matrix

    def save(self):
        """saves results for posterior recovery"""
        file = self.filename
        with open(file, 'wb') as arq:
            pkl.dump(self, arq)
