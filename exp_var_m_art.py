#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint:disable=invalid-name,missing-docstring
"""
Executes experiment pipeline varying the number of samples on Art1.
"""
import numpy as np
from sklearn.metrics import mean_squared_error as mse
from codes.design import Strategies
from codes.metrics import nmse
from codes.datasets import ArtGAMTLRegression
from codes.regression import GroupAMTL, GroupAMTLnr
from codes.stl_methods import LassoRegression, GroupLassoRegression
import exp_var_m


# cross validation parameters
def lasso_params():
    lambda_1 = np.logspace(-3, -1, 10)
    res = []
    for l1 in lambda_1:
        res.append({'lambda_1': l1})
    return res


def glasso_params():
    lambda_1 = np.linspace(0.0001, 15, 30)
    res = list()
    for l1 in lambda_1:
        res.append({'lambda_1': l1})
    return res


def gamtl_params():
    lambda_1 = np.linspace(0.001, 3, 10)
    lambda_2 = np.linspace(0.001, 0.5, 4)
    lambda_3 = np.linspace(0.005, 0.1, 4)
    res = list()
    for l1 in lambda_1:
        for l2 in lambda_2:
            for l3 in lambda_3:
                res.append({'lambda_1': l1,
                            'lambda_2': l2,
                            'lambda_3': l3})
    return res


def myamtl2_params():
    lambda_1 = np.linspace(0.001, 1, 10)
    lambda_2 = np.linspace(0.01, 1, 10)
    lambda_3 = [0]
    res = list()
    for l1 in lambda_1:
        for l2 in lambda_2:
            for l3 in lambda_3:
                res.append({'lambda_1': l1,
                            'lambda_2': l2,
                            'lambda_3': l3})
    return res


if __name__ == "__main__":
    strategies = Strategies()
    # METHODS GROUP
    groups = np.array([[0, 25],
                       [25, 50],
                       [np.sqrt(25), np.sqrt(25)]])
    init_params = {'name': 'GroupLasso',
                   'label': 'glasso_',
                   'groups': groups}
    strategies.add(GroupLassoRegression, init_params, glasso_params())

    # GAMTL
    init_params = {'name': 'GroupAMTLr',
                   'label': 'gamtlr',
                   'groups': groups}
    strategies.add(GroupAMTL, init_params, gamtl_params())

    init_params = {'name': 'GroupAMTL',
                   'label': 'gamtl',
                   'groups': groups}
    strategies.add(GroupAMTLnr, init_params, gamtl_params())

    # METHODS WITHOUT GROUP
    strategies.add(LassoRegression, {}, lasso_params())
    groups = np.array([[0],
                       [50],
                       [np.sqrt(50)]])
    strategies.add(GroupAMTL, {'name': 'AMTL2',
                               'label': 'amtl2',
                               'groups': groups},
                   myamtl2_params())

    vals = np.linspace(30, 201, 10, dtype=int).tolist()
    hp_metric = ('nmse', nmse)
    hp_bb = False
    metrics = [{'name': 'nmse', 'func': nmse}]
    task_metrics = [{'name': 'mse', 'func': mse}]
    exp_var_m.exp_base(ArtGAMTLRegression, 'art1_vary_m.pkl', vals,
                       strategies, hp_metric, hp_bb, metrics,
                       task_metrics, runs=30)
