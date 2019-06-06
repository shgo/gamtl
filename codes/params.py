#!/usr/bin/env python
# -*- coding: utf-8 -*-
#pylint:disable=invalid-name
"""
Set of functions that generates parameters for the pair (method x dataset) to be
used by an design.HyperParameterization class.
"""
import numpy as np

VALID_DATASETS = ('art1', 'landmine', 'adni')
def mtsgl(dataset):
    assert dataset in VALID_DATASETS
    return [{'r': i} for i in np.linspace(0.0001, 1, 20)]

def gsmurfs(dataset):
    assert dataset in VALID_DATASETS
    r1s = np.linspace(0.001, 5, 10).tolist()
    r2s = np.linspace(0.001, 5, 10).tolist()
    returns = []
    for r1 in r1s:
        for r2 in r2s:
            returns.append({'r1': r1, 'r2': r2})
    return returns

def gamtl(dataset):
    """ Generaters parameters for cross validation. """
    assert dataset in VALID_DATASETS
    res = list()
    init_params = dict()
    if dataset == 'art1':
        init_params['groups'] = [np.array([[0, 25],
                                 [25, 50],
                                 [np.sqrt(25), np.sqrt(25)]])]
        lambda_1 = np.linspace(0.0001, 0.001, 3)
        lambda_2 = np.linspace(0.0001, 0.001, 3)
        lambda_3 = np.linspace(0.01, 0.001, 2)
    elif dataset == 'landmine':
        init_params['groups'] = [np.array([[0],
                                 [9],
                                 [3]])]
        init_params['groups'] =  np.array([np.arange(9).tolist(),
                                           np.arange(1, 10).tolist(),
                                           [1 for i in range(9)]])
        lambda_1 = np.linspace(0.001, 0.01, 3)
        lambda_2 = np.linspace(0.001, 0.01, 3)
        lambda_3 = np.array([0.01, 0.02])
    #creating list and returning
    for l1 in lambda_1:
        for l2 in lambda_2:
            for l3 in lambda_3:
                    res.append({'lambda_1': l1,
                                'lambda_2': l2,
                                'lambda_3': l3})
    return init_params, res

def gamtl_class(dataset):
    """ Generaters parameters for cross validation. """
    assert dataset in VALID_DATASETS
    res = list()
    if dataset == 'landmine':
        groups = [np.array([[0],
                            [9],
                            [3]]),
                  np.array([np.arange(9).tolist(),
                            np.arange(1, 10).tolist(),
                            [1 for i in range(9)]])]
        lambda_1 = np.linspace(0.001, 0.01, 4)
        lambda_2 = np.linspace(0.001, 0.01, 4)
        lambda_3 = np.linspace(0.001, 0.01, 4)
    for l1 in lambda_1:
        for l2 in lambda_2:
            for l3 in lambda_3:
                for group in groups:
                    res.append({'lambda_1': l1,
                                'lambda_2': l2,
                                'lambda_3': l3,
                                'groups': group})
    return res

def go(dataset):
    """ Generaters parameters for cross validation. """
    assert dataset in VALID_DATASETS
    res = list()
    if dataset == 'art1':
        nb_latvars = [3]
        rho_1 = [0.001, 0.01]
        rho_2 = [0.001]
    elif dataset == 'landmine':
        nb_latvars = [3]
        rho_1 = [0.001, 0.01]
        rho_2 = [0.001]
    for nb_val in nb_latvars:
        for r1_val in rho_1:
            for r2_val in rho_2:
                res.append({'nb_latvars': nb_val,
                            'rho_1': r1_val,
                            'rho_2': r2_val})
    return res

def mssl(dataset):
    """ Generaters parameters for cross validation. """
    assert dataset in VALID_DATASETS
    res = list()
    if dataset == 'art1':
        rho_1 = [4, 6, 8] #estimados pra art1
        rho_2 = [0.1, 0.5, 1]
    elif dataset == 'landmine':
        rho_1 = [2, 4, 6] #estimados pra art1
        rho_2 = [0.1, 0.5, 1]
    for r1_val in rho_1:
        for r2_val in rho_2:
            res.append({'rho_1': r1_val,
                        'rho_2': r2_val})
    return res

def mtrl(dataset):
    """ Generaters parameters for cross validation. """
    assert dataset in VALID_DATASETS
    res = list()
    if dataset == 'art1':
        rho_1 = np.linspace(0.001, 0.01, 4)
        rho_2 = np.linspace(0.001, 0.01, 4)
    elif dataset == 'landmine':
        rho_1 = [0.0001, 0.001]
        rho_2 = [0.001, 0.003, 0.005, 0.006]
    for r1_val in rho_1:
        for r2_val in rho_2:
            res.append({'rho_1': r1_val,
                        'rho_2': r2_val})
    return res

def group_mtl(dataset):
    """ Generaters parameters for cross validation. """
    assert dataset in VALID_DATASETS
    res = list()
    if dataset == 'art1':
        nb_groups = np.arange(2, 4, step=3)
        rho_1 = [0.001, 0.01, 0.05, 0.1]
        rho_2 = [0.001, 0.01, 0.05, 0.1]
    elif dataset == 'landmine':
        nb_groups = np.arange(2, 4, step=3)
        rho_1 = [0.001, 0.01, 0.05, 0.1]
        rho_2 = [0.001, 0.01, 0.05, 0.1]
    for r0_val in nb_groups:
        for r1_val in rho_1:
            for r2_val in rho_2:
                res.append({'nb_groups': float(r0_val),
                            'rho_1': r1_val,
                            'rho_2': r2_val})
    return res

def amtl(dataset):
    """ Generaters parameters for cross validation. """
    assert dataset in VALID_DATASETS
    res = list()
    if dataset == 'art1':
        mu = [0.001, 0.01, 0.1]
        lamb = [0.001, 0.01]
    elif dataset == 'landmine':
        mu = np.linspace(0.0001, 0.01, 5)
        lamb = np.linspace(0.0001, 0.01, 5)
    elif dataset == 'adni':
        mu = np.linspace(1e-3, 1, 10)
        lamb = np.linspace(1e-3, 1, 10)
    for m1 in mu:
        for l1 in lamb:
            res.append({'mu': m1, 'lamb': l1})
    return res

def group_lasso(dataset):
    """ Generaters parameters for cross validation. """
    assert dataset in VALID_DATASETS
    res = list()
    lambda_1 = np.linspace(0.001, 0.1, 6)
    if dataset == 'art1':
        groups = [np.array([[0, 25],
                            [25, 50],
                            [np.sqrt(25), np.sqrt(25)]])]
    elif dataset == 'landmine':
        groups = [np.array([[0],
                            [9],
                            [3]]),
                  np.array([[i for i in range(9)],
                            [i for i in range(1, 10)],
                            [1 for i in range(9)]])]
    for l_val in lambda_1:
        for group in groups:
            res.append({'lambda_1': l_val, 'groups': group})
    return res

def lasso(dataset):
    """ Generaters parameters for cross validation. """
    assert dataset in VALID_DATASETS
    res = list()
    if dataset == 'art1':
        lambda_1 = np.logspace(-3, 2, 5)
        #lambda_1 = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 1.5]
    elif dataset == 'landmine':
        lambda_1 = np.logspace(-3, 1, 5)
    for val in lambda_1:
        res.append({'lambda_1': val})
    return res
