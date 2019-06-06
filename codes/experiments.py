#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy

import pandas as pd

from codes.design import ExperimentMTL
from codes.metrics import nmse


class ExperimentVarParam(ExperimentMTL):
    """
    Experiment that varies a parameter in the Dataset object.

    Args:
        name (str): name
        filename (str): filename
        param_name (str): name of parameter that will vary
        param_vals (list): values of parameter

    Attributes:
        msg (str): Human readable string describing the exception
        runs (int): number of runs (default 1)
        dataset (ArtificialDataset): experiment dataset
        resul (dict): experiment results. A dict organized mess.
        hp_metric (name, func): metric to be used in the hyper-parameterization procedure
        hp_bb (bool): bigger is best when considering metric in hyper-parameterization procedure
        done (list of lists): status of experiment

    Methods:
        execute
        delete
    """
    def __init__(self, name, filename, param_name, param_vals):
        super().__init__(name, filename)
        assert isinstance(param_vals, list)
        self.dataset_params = param_vals
        assert param_name is not None
        self.dataset_param_name = param_name
        self.resul = dict()
        lists = list()
        for _ in param_vals:
            temp = dict()
            temp['dataset'] = None
            temp['hp'] = None
            lists.append(temp)
        self.resul['objs'] = lists
        self.resul['metrics'] = pd.DataFrame()
        self.resul['task_metrics'] = pd.DataFrame()
        self.resul['data_ref'] = None
        self.hp_metric = ('nmse', nmse)
        self.hp_bb = False
        self.done = [[] for i in range(len(param_vals))]

    def execute(self, n_jobs=1):
        """Executes Experiment using actual configuration.

        :param n_jobs: number of cores to use. Default: -1, which will use
            all possible cores.
        """
        print('------------------------------------------------------------')
        print('|Experiment name: {:40}|'.format(self.name))
        print('|Results file: {:44}|'.format(self.filename))
        print('------------------------------------------------------------')
        self._check_resul()
        print('\t\t\tPREPARING DATASET')
        dataset = None
        if self.resul['data_ref']:
            dataset = self.resul['data_ref']
        else:
            dataset = self.dataset()
            dataset.generate()
            self.resul['data_ref'] = dataset
        print('\t\t\tDONE')
        for pos, dataset_param in enumerate(self.dataset_params):
            temp = dataset.get_params()
            temp.update({self.dataset_param_name: dataset_param})
            dataset.set_params(**temp)
            dataset.generate_inds()
            self.resul['objs'][pos]['dataset'] = copy.deepcopy(dataset)
            if self.resul['objs'][pos]['hp'] is not None:
                hp = self.resul['objs'][pos]['hp']
            else:
                hp = self.hyper_parameterization(dataset=dataset,
                                                 metric=self.hp_metric,
                                                 bb=self.hp_bb,
                                                 split=0.7)
                self.resul['objs'][pos]['hp'] = hp
            list_methods = [strategy for strategy in self.strategies.get_list()
                            if strategy.name not in self.done[pos]]
            for strategy in list_methods:
                print('FOR {}= {}'.format(self.dataset_param_name,
                                          dataset_param))
                res, method_name, metrics, task_metrics = \
                    self._run_method_pipeline(dataset, hp, strategy,
                                              n_jobs)
                metrics['dataset_param'] = dataset_param
                task_metrics['dataset_param'] = dataset_param
                self.resul['objs'][pos][method_name] = res
                self.resul['metrics'] = self.resul['metrics'].append(metrics)
                self.resul['task_metrics'] = \
                    self.resul['task_metrics'].append(task_metrics)
                self.done[pos].append(strategy.name)
                print('done 1')
                self.save()
        print('The End')

    def delete(self, method_name):
        """Removes a method from the results, being able to run it again.

        :param method_name: method name
        """
        self.resul['metrics'] = \
            self.resul['metrics'][self.resul['metrics']['method'] != method_name]
        self.resul['task_metrics'] = \
            self.resul['task_metrics'][self.resul['task_metrics']['method'] != method_name]
        for pos in range(len(self.resul['objs'])):
            if method_name in self.resul['objs'][pos]:
                del self.resul['objs'][pos][method_name]
            if method_name in self.done[pos]:
                self.done[pos].remove(method_name)


class Experiment(ExperimentMTL):
    """
    Experiment that uses Cross-Validation as hyper parameterization procedure in a common pipeline.

    Args:
        name (str): name
        filename (str): filename

    Methods:
        execute
        delete
    """

    def __init__(self, name, filename):
        super().__init__(name, filename)
        self.hp_permutation = False

    def execute(self, n_jobs=1):
        """Executes Experiment using actual configuration.

        :param n_jobs: number of cores to use. Default: -1, which will use
            all possible cores.
        """
        print('------------------------------------------------------------')
        print('|Experiment name: {:40}|'.format(self.name))
        print('|Results file: {:44}|'.format(self.filename))
        print('------------------------------------------------------------')
        self._check_resul()
        dataset = None
        if self.resul['dataset'] is not None:
            dataset = self.resul['dataset']
        else:
            dataset = self.dataset()
            dataset.generate()
            self.resul['dataset'] = copy.deepcopy(dataset)
        if self.resul['hp'] is not None:
            hp = self.resul['hp']
        else:
            hp = self.hyper_parameterization(dataset=dataset,
                                             metric=self.hp_metric,
                                             bb=self.hp_bb,
                                             permutation=self.hp_permutation)
            self.resul['hp'] = hp
        list_methods = [strategy for strategy in self.strategies.get_list() \
                        if strategy.name not in self.done]
        for strategy in list_methods:
            res, method_name, metrics, task_metrics = \
                    self._run_method_pipeline(dataset, hp, strategy, n_jobs)
            self.resul[method_name] = res
            self.resul['metrics'] = self.resul['metrics'].append(metrics)
            self.resul['task_metrics'] = \
                self.resul['task_metrics'].append(task_metrics)
            self.done.append(strategy.name)
            self.save()
        print('The End')

    def delete(self, method_name):
        """Removes a method from the results, being able to run it again.

        :param method_name: method name
        """
        self.resul['metrics'] = \
            self.resul['metrics'][self.resul['metrics']['method'] != method_name]
        self.resul['task_metrics'] = \
            self.resul['task_metrics'][self.resul['task_metrics']['method'] != method_name]
        del self.resul[method_name]
