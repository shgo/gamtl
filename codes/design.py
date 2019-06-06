#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint:disable=invalid-name,len-as-condition,too-many-instance-attributes,too-many-arguments,consider-using-enumerate,too-many-locals,too-few-public-methods,no-member,protected-access
"""
Module that contains class definitions for experimental setups.
"""
import copy
import os.path
import pickle as pkl
import time
import warnings
from abc import ABCMeta, abstractmethod

import numpy as np
import pandas as pd
import scipy.special
from joblib import Parallel, delayed

from codes.metrics import macc

warnings.simplefilter('ignore')

PREFIX = os.path.dirname(__file__)


class Dataset:
    """ Dataset representation for Multi-Task Learning (MTL) methods.

    Args:
        name (str): name of the dataset
        m (int): number of samples per task
        n (int): number of attributes
        T (int): number of tasks
        split (float): pct of samples to be in the training set
        permutation (bool): Permute samples when spliting into train/ test sets

    Methods:
        generate
        generate_inds
        get_data
        get_train
        get_test
        get_params
        set_params
        get_metadata
        __str__

    Attributes:
        W (np.array n x T): parameter matrix
        X (list of (m x n)): samples
        y (list of (m)): labels
    """
    __metaclass__ = ABCMeta

    def __init__(self, name, m, n, T, split, permutation=False):
        assert isinstance(name, str)
        self.name = name
        assert split <= 1.0
        assert split > 0.0
        self.split = split
        assert n > 0
        self.n = n
        assert m > 0
        self.T = T
        self.m = [m for i in range(T)]
        self.X = [np.array([]) for i in range(T)]
        self.y = [np.array([]) for i in range(T)]
        self.inds = [[] for i in range(T)]
        self.inds_tr = [[] for i in range(T)]
        self.inds_te = [[] for i in range(T)]
        self.permutation = permutation

    def generate(self):
        """
        Generates dataset, allowing get_train() get_test() to be called
        Also generates inds for training / test set.
        """
        self.generate_inds()

    def generate_inds(self):
        """ Generates inds for training / test set split. """
        if self.X[0].shape[0] < 1:
            raise Exception('Dataset not generated')
        is_classification = self.y[0].dtype.kind == 'u'
        for ind_task in range(self.T):
            m_data = self.X[ind_task].shape[0]
            if self.permutation:
                self.inds[ind_task] = np.random.permutation(m_data)
            self.inds[ind_task] = list(range(m_data))
            m_val = int(np.min([self.m[ind_task], m_data]))
            if is_classification and m_val < m_data:
                inds_c0 = np.where(self.y[ind_task] == 0)
                inds_c1 = np.where(self.y[ind_task] == 1)
                perm_0 = np.random.permutation(inds_c0[0])
                perm_1 = np.random.permutation(inds_c1[0])
                len_0 = len(inds_c0[0])
                len_1 = len(inds_c1[0])
                half = round(m_val / 2)
                if half <= np.min((len_0, len_1)):
                    i0 = perm_0[:half]
                    i1 = perm_1[:half]
                else:
                    if len_0 < len_1:
                        tot = m_val - len_0
                        i0 = perm_0[:]
                        i1 = perm_1[:tot]
                    else:
                        tot = m_val - len_1
                        i0 = perm_0[:tot]
                        i1 = perm_1[:]
                self.inds_tr[ind_task] = np.concatenate((i0, i1))
                self.inds_te[ind_task] = np.setdiff1d(self.inds[ind_task],
                                                      self.inds_tr[ind_task])
            else:
                end_tr = round(self.split * m_val)
                self.inds_tr[ind_task] = self.inds[ind_task][:end_tr]
                self.inds_te[ind_task] = self.inds[ind_task][end_tr:]

    def get_data(self, t, inds):
        """
            Returns X, y data of all tasks.

        :param t: task index
        :param inds: inds sets of all tasks
        :returns: ret
        :rtype: dict
        """
        if self.X is None:
            raise Exception(
                "You need to generate the dataset first. Call generate().")

        ret = dict()
        if t is not None:
            ret['X'] = self.X[t][inds[t]]
            ret['y'] = self.y[t][inds[t]]
        else:
            ret['X'] = [[] for task in range(self.T)]
            ret['y'] = [[] for task in range(self.T)]
            for task in range(self.T):
                ret['X'][task] = self.X[task][inds[task]]
                ret['y'][task] = self.y[task][inds[task]]
        return ret

    def get_train(self, t=None):
        """
        Returns X, y of training set.

        :param t: optional, task index.
        :returns: ret
        :rtype: dict
        """
        inds = self.inds_tr
        return self.get_data(t=t, inds=inds)

    def get_test(self, t=None):
        """
        Returns X, y of training set.

        :param t: optional, task index.
        :returns: ret
        :rtype: dict
        """
        inds = self.inds_te
        return self.get_data(t=t, inds=inds)

    def get_params(self):
        """
        Returns object parameters used to generate data.
        Contains:
            ret (dict): keys:
                'm' - #rows design matrix
                'n' - #cols design matrix
                'split' - how to split data
                'T' - number of tasks
        :returns: ret
        :rtype: dict
        """
        return {'m': self.m, 'n': self.n, 'T': self.T}

    def set_params(self, m, n, T):
        """
        Sets object parameters used to generate data.

        :param m: number of samples for each task
        :param n: number of attributes
        :param T: number of tasks
        """
        if isinstance(m, (int, float)) or (len(m) == 1 and np.isinf(m)):
            self.m = [m for i in range(T)]
        else:
            self.m = m
        assert T > 0
        self.T = T
        assert n > 0
        self.n = n

    def get_metadata(self):
        """
        Prints metadata from object and returns a Dataframe with information
        of train / test split.
        """
        train = self.get_train()
        X = train['X']
        y = train['y']
        test = self.get_test()
        Xte = test['X']
        yte = test['y']
        lenX = []
        lenXte = []
        leny = []
        lenyte = []
        len_inds = []
        is_classification = self.y[0].dtype.kind == 'u'
        if is_classification:
            tot0 = []
            tot1 = []
            tot0te = []
            tot1te = []
        for t in range(len(X)):
            if is_classification:
                tot0.append(len(np.where(y[t] == 0)[0]))
                tot1.append(len(np.where(y[t] == 1)[0]))
                tot0te.append(len(np.where(yte[t] == 0)[0]))
                tot1te.append(len(np.where(yte[t] == 1)[0]))
            lenX.append(X[t].shape[0])
            lenXte.append(Xte[t].shape[0])
            leny.append(y[t].shape[0])
            lenyte.append(yte[t].shape[0])
            len_inds.append(len(self.inds[t]))
        if is_classification:
            di = {
                'tam Xtr': lenX,
                'tam ytr': leny,
                'tam Xte': lenXte,
                'tam yte': lenyte,
                'len inds': len_inds,
                'class 0 tr': tot0,
                'class 1 tr': tot1,
                'class 0 te': tot0te,
                'class 1 te': tot1te
            }
        else:
            di = {
                'tam Xtr': lenX,
                'tam ytr': leny,
                'tam Xte': lenXte,
                'tam yte': lenyte,
                'len inds': len_inds
            }
        return pd.DataFrame(di)

    def __str__(self):
        """
        Briefly description of object.

        :returns: ret
        :rtype: str
        """
        ret = "Dataset {}\n{} tasks with {} attributes.\n{}"
        ret = ret.format(self.name, self.T, self.n, self.get_metadata())
        return ret


class ArtificialDataset(Dataset):
    """ Represents an Artificial Dataset for multi-task learning methods.

    Args:
        name (str): name of the dataset.
        m (int): number of samples
        n (int): number of attributes
        T (int): number of tasks
        split (float): pct of samples to be in the training set.
        W (np.array n x T): parameter matrix
        groups (np.array 3 x |G|): groups of attributes.
            [[start g1, start g2, ..., start gG]
            [end g1, end g2, ..., end gG]
            [weight g1, weight g2, ... weight gG]]
            Each column of this matrix should contain the start index, stop index, and group weight.
            Recommended value for weight_g =  np.sqrt(end g - start g).

    Attributes:
        W (np.array n x T): parameter matrix
        X (list of (m x n)): samples
        y (list of (m)): labels
        groups

    Methods:
        get_params
        set_params
    """
    __metaclass__ = ABCMeta

    def __init__(self, name, m, n, T, split, W, groups):
        super().__init__(name, m, n, T, split)
        self.W = W
        self.groups = groups
        self.noise = None

    def get_params(self):
        """
        Returns object parameters used to generate data.
        Contains:
            ret (dict): keys:
                'm' - #rows design matrix
                'n' - #cols design matrix
                'split' - how to split data
                'T' - number of tasks
        :returns: ret
        :rtype: dict
        """
        return {'m': self.m, 'n': self.n, 'T': self.T,
                'split': self.split}

    def set_params(self, m, n, T, split):
        """
        Sets object parameters used to generate data.

        :param m: number of samples for each task
        :param n: number of attributes
        :param T: number of tasks
        """
        if isinstance(m, (int, float)) or (len(m) == 1 and np.isinf(m)):
            self.m = [m for i in range(T)]
        else:
            self.m = m
        assert T > 0
        self.T = T
        assert n > 0
        self.n = n
        assert split <= 1
        assert split > 0
        self.split = split


class Method:
    """ Method representation for MTL.
    Args:
        name (str): name of the method.
        label (str): label of the method.
        normalize (bool): standardize data.
        bias (bool): add bias term.

    Attributes:
        W (np.array n x T): parameter matrix
        _col_means (list of np.array n)
        _col_stds (list of np.array n)
        _y_means (list of np.array n)
        _y_stds (list of np.array n)

    Methods:
        add_bias
        normalize_data
        normalize_in
        fit (abstract)
        predict (abstract)
        set_params (abstract)
        get_params (abstract)
        get_resul (abstract)
    """
    __metaclass__ = ABCMeta

    def __init__(self, name, label, normalize, bias):
        self.name = name
        self.label = label
        self.W = None
        self.normalize = normalize
        self.bias = bias
        self._col_means = None
        self._col_stds = None
        self._y_means = None
        self._y_stds = None

    def add_bias(self, X):
        """ Adds bias term at the last column of X on all tasks."""
        if self.bias:
            X = X.copy()
            for t in range(len(X)):
                X[t] = np.append(X[t], np.ones((X[t].shape[0], 1)), axis=1)
        return X

    def normalize_data(self, X):
        """
        Standardize data and stores means / stds.

        :param X: samples
        """
        if self.normalize:
            X = X.copy()
            tasks = list(range(len(X)))
            self._col_means = [[] for i in tasks]
            self._col_stds = [[] for i in tasks]
            for t in range(len(X)):
                self._col_means[t] = np.mean(X[t], axis=0)
                var = np.std(X[t], axis=0)
                var = np.where(var > 1e-4, var, 1)
                self._col_stds[t] = var
                X[t] = (X[t] - self._col_means[t]) / self._col_stds[t]
        return X

    def normalize_in(self, X):
        """
        Normalize X data.

        :param X: samples
        :param t: task index
        """
        X = X.copy()
        if self.normalize and self._col_means:
            for t in range(len(X)):
                if self.bias:
                    X[t][:, :-1] = (X[t][:, :-1] - self._col_means[t]) / self._col_stds[t]
                else:
                    X[t] = (X[t] - self._col_means[t]) / self._col_stds[t]
        return X

    @abstractmethod
    def fit(self, X, y):
        """
        Runs learning routine.

        :param X: list of np.array (m_t x n) samples
        :param y: list of np.array (m_t) labels

        """
    @abstractmethod
    def predict(self, X):
        """ Predicts output for samples in X if fit already called.
        Args:
        :param X: input data
        """
    @abstractmethod
    def set_params(self):
        """ Sets method parameters. """
    @abstractmethod
    def get_params(self):
        """ Returns method parameters. """

    @abstractmethod
    def get_resul(self):
        """ Returns estimated results. """

class MethodRegression(Method):
    """ Implements specificities for Regression methods.

    Methods:
        predict
        fit (abstract)
    """
    __metaclass__ = ABCMeta

    def predict(self, X):
        """ Predicts output for samples in X if fit already called.
        Args:
        :param X: input data
        """
        assert isinstance(X, list)
        assert len(X) == self.W.shape[1]
        X = self.add_bias(X)
        X = self.normalize_in(X)
        y = []
        for t in range(len(X)):
            y_temp = np.dot(X[t], self.W[:, t])
            y.append(y_temp)
        return y

    @abstractmethod
    def fit(self, X, y):
        """
        Runs learning routine.

        Args:
            :param X: list of np.array (m_t x n) samples
            :param y: list of np.array (m_t) labels
        """


class MethodClassification(Method):
    """ Implements specificities for Classification methods.

    Methods:
        predict
        fit (abstract)
    """

    __metaclass__ = ABCMeta

    def __init__(self, name, label, normalize=False, bias=False,
                 threshold=0.5):
        super().__init__(name, label, normalize, bias)
        assert isinstance(threshold, float)
        self.threshold = threshold

    def predict(self, X):
        """ Predicts output for samples in X if fit already called.
        Args:
        :param X: input data
        """
        assert isinstance(X, list)
        assert len(X) == self.W.shape[1]
        X = self.add_bias(X)
        X = self.normalize_in(X)
        y = []
        for t in range(len(X)):
            z = np.dot(X[t], self.W[:, t])
            h = scipy.special.expit(z)
            y_temp = (h >= self.threshold).astype('int')
            y.append(y_temp)
        return y

    @abstractmethod
    def fit(self, X, y):
        """
        Runs learning routine.

        Args:
            :param X: list of np.array (m_t x n) samples
            :param y: list of np.array (m_t) labels
        """


class HyperParameterization:
    """ HyperParameterization procedure representation.

    Args:
        dataset (ArtificialDataset, RealDataset): dataset to be
            used.
        params (dict): like method.params, but each param value is a
            list.
        metric (name, func): metric to be used.
        bb (bool): bigger is better flag.

    Methods:
        fit_param
        fit (abstract)
    """
    __metaclass__ = ABCMeta

    def __init__(self, dataset, metric, bb=True):
        assert isinstance(dataset, Dataset)
        self.dataset = dataset
        assert isinstance(metric[0], str)
        self.metric_name = metric[0]
        assert callable(metric[1])
        self.metric = metric[1]
        assert isinstance(bb, bool)
        self.bb = bb

    def _fit_param(self, method_ref, init_params, param):
        """Fits method_ref initialized with init_params, using params.

        Args:
        :param method_ref: method to be used
        :param init_params: initianization params of method
        :param param: parameters for method

        Returns:
        :returns: res
        :rtype: set
        """
        train = self.get_train()
        Xtr = train['X']
        ytr = train['y']
        val = self.get_val()
        Xval = val['X']
        yval = val['y']
        method = method_ref(**init_params)
        meth_params = method.get_params()
        temp_params = {**meth_params, **param}
        method.set_params(**temp_params)
        res = pd.Series(data=param)
        res['metric'] = self.metric_name
        method.fit(Xtr, ytr)
        ytr_pred = method.predict(Xtr)
        yval_pred = method.predict(Xval)
        res['tr'] = self.metric(ytr, ytr_pred)
        res['val'] = self.metric(yval, yval_pred)
        return (res, param)

    @abstractmethod
    def fit(self, method_ref, init_params, params, n_jobs):
        """
        Runs hyper parameterization procedure.

        Args:
            :param X: list of np.array (m_t x n) samples
            :param y: list of np.array (m_t) labels

        """


class Strategy:
    """ Strategy for HyperParameterization.
    Encapsulates hyper parameterization with methods.

    Args:
        method (Method): method
        init_params (dict): method initialization parameters
        hp_params (dict): hyper parameterization parameters

    Attributes:
        name (str): name of the procedure.

    Methods:
        components
    """
    def __init__(self, method, init_params, hp_params):
        self.method = method
        self.init_params = init_params
        self.hp_params = hp_params
        self.name = method.__name__
        if 'name' in init_params and init_params['name'] == 'AMTL2':
            self.name = 'AMTL2'

    def components(self):
        """ Returns all data from Strategy. """
        return self.method, self.init_params, self.hp_params


class Strategies:
    """
    List of strategies that will run in a Experiment.

    Attributes
        strategies (list): list of Strategy

    Methods:
        add
        get_list
    """
    def __init__(self):
        self.strategies = list()

    def add(self, method, init_params, hp_params):
        """Adds a Strategy to the list.

        :param method: method
        :param init_params: initialization parameters
        :param hp_params: hyper parameterization parameters
        """
        assert issubclass(method, Method)
        assert isinstance(init_params, dict)
        assert isinstance(hp_params, list)
        for items in hp_params:
            for _, val in items.items():
                isint = isinstance(val, int)
                isfloat = isinstance(val, float)
                islist = isinstance(val, list)
                isbol = isinstance(val, bool)
                if not (isint or isfloat or islist or isbol):
                    raise ValueError('Unsupported data type')
        self.strategies.append(Strategy(method, init_params, hp_params))

    def get_list(self):
        """Returns list of Strategy objects that will be used in Experiment.

        :returns: strategies
        :rtype: list
        """
        return self.strategies


class ExperimentMTL:
    """
    Class that defines an MTL Experiment and contains some helper methods.
    The results are saved with several plots and tables in mind.
    Stick with it whenever possible!

    Args:
        name (str): name of experiment
        filename (str): name of experiment results file

    Attributes:
        runs
        dataset
        strategies
        metrics
        hyper_parameterization
        task_metrics

    Methods:
        execute (abstract)
        delete (abstract)
        _check_resul
        _run_method_pipeline
        _one_run
        __train
        __metric_evaluation
        __task_metrics_evaluation
        save

    """
    __metaclass__ = ABCMeta

    def __init__(self, name, filename):
        assert isinstance(name, str)
        self.name = name
        assert isinstance(filename, str)
        if not os.path.exists('results'):
            os.makedirs('results')
        self.filename = 'results/{}'.format(filename)
        self.resul = dict()
        self.resul['dataset'] = None
        self.resul['hp'] = None
        self.resul['metrics'] = pd.DataFrame()
        self.resul['task_metrics'] = pd.DataFrame()
        self.hp_metric = ('macc', macc)
        self.hp_bb = True
        self.done = list()
        self._runs = None  # número padrão de execuções.
        self._dataset = None
        self._strategies = None
        self._metrics = None
        self._task_metrics = None
        self._hyper_parameterization = None

    @property
    def runs(self):
        """ property """
        return self._runs

    @runs.setter
    def runs(self, runs):
        """ Define number of runs

        :param runs: number of runs.
        """
        assert isinstance(runs, int)
        assert runs >= 0
        self._runs = runs

    @property
    def dataset(self):
        """ property """
        return self._dataset

    @dataset.setter
    def dataset(self, dataset):
        """ Define experiment dataset.

        :param dataset: dataset
        """
        assert issubclass(dataset, Dataset)
        self._dataset = dataset

    @property
    def strategies(self):
        """ property """
        return self._strategies

    @strategies.setter
    def strategies(self, strategies):
        """ Define methods that will run in this experiment.

        :param strategies: strategies
        """
        assert isinstance(strategies, Strategies)
        for strategy in strategies.get_list():
            method, init_params, hp_params = strategy.components()
            assert issubclass(method, Method)
            assert isinstance(init_params, dict)
            assert isinstance(hp_params, list)
        self._strategies = strategies

    @property
    def metrics(self):
        """ property """
        return self._metrics

    @metrics.setter
    def metrics(self, metrics):
        """ Define overall metrics of experiment.

        :param metrics: (name, func)
        """
        assert isinstance(metrics, list)
        assert len(metrics) > 0
        for metric in metrics:
            assert 'name' in metric.keys()
            assert 'func' in metric.keys()
            assert callable(metric['func'])
        self._metrics = metrics

    @property
    def hyper_parameterization(self):
        """ property """
        return self._hyper_parameterization

    @hyper_parameterization.setter
    def hyper_parameterization(self, hyper_parameterization):
        """How hyper parameters will be estimated.

        :param hyper_parameterization:
        """
        assert issubclass(hyper_parameterization, HyperParameterization)
        self._hyper_parameterization = hyper_parameterization

    @property
    def task_metrics(self):
        """ property """
        return self._task_metrics

    @task_metrics.setter
    def task_metrics(self, task_metrics):
        """ Metrics that will be computed per task.

        :param task_metrics:
        """
        assert isinstance(task_metrics, list)
        assert len(task_metrics) > 0
        for tm in task_metrics:
            assert 'name' in tm.keys()
            assert 'func' in tm.keys()
            assert callable(tm['func'])
        self._task_metrics = task_metrics

    @abstractmethod
    def execute(self):
        """Each subclass must define it's own way of executing. Some helper
        methods are available below."""

    @abstractmethod
    def delete(self, method):
        """Each subclass must define it's own way of deleting a method from
        its results."""

    def _check_resul(self):
        """ Deletes method from the results

        :param method_name: name of method in Strategies
        """
        if self.filename and os.path.isfile(self.filename):
            print(
                """There is an experiment with this filename.\n
                - [s]top right now
                - [c]ontinue from last saved results
                - [r]estart?(C/r)""")
            aqui = input()
            if aqui in ('r', 'R'):
                os.remove(self.filename)
            elif aqui in ('c', 'C'):
                with open(self.filename, 'rb') as arq:
                    temp = pkl.load(arq)
                    self.resul = temp.resul
                    self.done = temp.done
            else:
                os._exit(1)

    def _run_method_pipeline(self, dataset, hp, strategy, n_jobs=1):
        """ Default method pipeline.

        Args:
            :param dataset: Dataset
            :param hp: HyperParameterization
            :param strategy: Strategies
            :param n_jobs: number of cores to use. -1 will use all possible.

        Returns:
            :returns: ret
            :rtype: dict
            :returns: method_name
            :rtype: str
            :returns: metrics
            :rtype: pd.DataFrame
            :returns: task_metrics
            :rtype: pd.DataFrame
        """
        method_ref, init_params, params_grid = strategy.components()
        ret = dict()
        start_time = time.time()
        params, res = hp.fit(method_ref, init_params, params_grid, n_jobs)
        cv_time = time.time() - start_time
        print('Hyper Parameterization took {:.3f} seconds...'.format(cv_time))
        cv_time = time.time()
        ret['hyper_params'] = res
        ret['best_params'] = params
        ret['resul'] = list()
        ret['f'] = list()
        ret['time'] = list()
        metrics = pd.DataFrame()
        task_metrics = pd.DataFrame()
        results = list()
        start_time = time.time()
        if n_jobs == 1:
            for run in range(self.runs):
                results.append(
                    self._one_run(dataset, method_ref, init_params, params,
                                  run))
        else:
            results = Parallel(n_jobs=n_jobs)(delayed(self._one_run)(
                dataset, method_ref, init_params, params, run)
                                              for run in range(self.runs))
        run_time = time.time() - start_time
        print('RUN took {:.3f} seconds...'.format(run_time))
        start_time = time.time()
        for result in results:
            ret_run, df_metric, df_task_metric, method_name = result
            ret['f'].append(ret_run['f'])
            ret['time'].append(ret_run['time'])
            ret['resul'].append(ret_run['resul'])
            metrics = metrics.append(df_metric)
            task_metrics = task_metrics.append(df_task_metric)
        met_time = time.time() - start_time
        print('Metric evaluation took {:.3f} seconds...'.format(met_time))
        return ret, method_name, metrics, task_metrics

    def _one_run(self, dataset, method_ref, init_params, params, run):
        """Executes one run of a method in a dataset.

        Args:
            :param dataset: Dataset
            :param method_ref: method to run
            :param init_params: initialization parameters
            :param params: parameters of method execution
            :param run: number of run

        Returns:
            :returns: ret
            :rtype: dict
            :returns: df_metric
            :rtype: pd.DataFrame
            :returns: df_task_metric
            :rtype: pd.DataFrame
            :returns: method_name
            :rtype: str

        """
        method_name = None
        method = method_ref(**init_params)
        if method_name is None:
            method_name = method.name
        method.set_params(**params)
        ret = dict()
        res, resul = self.__train(method, dataset)
        ret['f'] = res['f']
        ret['time'] = res['time']
        ret['resul'] = resul
        df_metric = self.__metric_evaluation(method, dataset)
        df_metric['run'] = run
        df_metric['method'] = method_name
        df_task_metric = self.__task_metrics_evaluation(method, dataset)
        df_task_metric['run'] = run
        df_task_metric['method'] = method_name
        return ret, df_metric, df_task_metric, method_name

    def __train(self, method, dataset):
        """
        Trains model in dataset.
        Args:
            run (int):
            method (Method): already instantiated.
            dataset (Dataset): already generated.
        Returns
            :returns: ret
            :rtype: dict
                ['time']: int
                ['f']: list (max len == 100)
                ['resul']: dict
            :returns: method_resul
            :rtype: dict
        """
        f = []
        tempo = 0
        res = dict()
        train = dataset.get_train()
        Xtr = train['X']
        ytr = train['y']
        _, f, tempo = method.fit(Xtr, ytr)
        if isinstance(f, float):
            f = np.array([f])
        f_cor = np.array([])
        if len(f) < 100:
            f_cor = np.lib.pad(
                f.flatten(), (0, 100 - len(f)), 'constant', constant_values=0)
        else:
            f_cor = f.flatten()[:100]
        res['f'] = f_cor
        res['time'] = tempo
        return res, method.get_resul()

    def __metric_evaluation(self, method, dataset):
        """
        Evaluates general metrics with the predictions of method in
        dataset.
        The result is stored in self.resul using pos and run.

        Args:
            method (Method):
            dataset (Dataset):

        Returns:
            :returns: df_ret
            :rtype: pd.DataFrame
        """
        df_ret = pd.DataFrame(columns=['metric', 'tr', 'te'])
        for metric in self.metrics:
            train = dataset.get_train()
            Xtr = train['X']
            ytr = train['y']
            ytr_pred = method.predict(Xtr)
            test = dataset.get_test()
            Xte = test['X']
            yte = test['y']
            yte_pred = method.predict(Xte)
            res = pd.Series()
            res['metric'] = metric['name']
            res['tr'] = metric['func'](ytr, ytr_pred)
            res['te'] = metric['func'](yte, yte_pred)
            df_ret = df_ret.append(res, ignore_index=True)
        return df_ret

    def __task_metrics_evaluation(self, method, dataset):
        """
        Evaluates metrics per task with the predictions of method in dataset.
        The result is stored in self.resul using pos and run.

        Args:
            :param method: method
            :param dataset: dataset

        Returns:
            :returns: df_ret
            :rtype: pd.DataFrame
        """
        df_ret = pd.DataFrame(columns=['metric', 'task', 'tr', 'te'])
        for metric in self.task_metrics:
            train = dataset.get_train()
            Xtr = train['X']
            ytr = train['y']
            ytr_pred = method.predict(Xtr)
            test = dataset.get_test()
            Xte = test['X']
            yte = test['y']
            yte_pred = method.predict(Xte)
            for t in range(dataset.T):
                ret = pd.Series()
                ret['metric'] = metric['name']
                ret['task'] = t
                ret['tr'] = metric['func'](ytr[t], ytr_pred[t])
                ret['te'] = metric['func'](yte[t], yte_pred[t])
                df_ret = df_ret.append(ret, ignore_index=True)
        return df_ret

    def save(self):
        """
        Pickles Experiment object.
        I'm reading only the attributes resul and .
        The remainning attributes are still saved, but the intention is to have
        if set before loading. This save/load option is here only to recover
        from any interruption in the experiment execution.
        """
        with open(self.filename, 'wb') as arq:
            copia = copy.deepcopy(self)
            pkl.dump(copia, arq)
