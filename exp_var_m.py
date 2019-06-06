#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Função que executa experimento variando parametros. Seu nome é prefixo no nome
dos scripts que a usam.
"""
from codes.experiments import ExperimentVarParam
from codes.hyper_params import CrossValidation
def exp_base(dataset, filename, vals, strategies, hp_metric, hp_bb, metrics,
             task_metrics):
    """
        Args:
            dataset (codes.design.RealDatasetMTL): dataset to run experiment.
            filename (str): filename of the results file.
    """
    exp = ExperimentVarParam('Varying m on {}'.format(dataset), filename,
                             param_name='m', param_vals=vals)
    exp.hp_metric = hp_metric
    exp.hp_bb = hp_bb
    exp.dataset = dataset
    exp.runs = 2#30
    exp.metrics = metrics
    exp.task_metrics = task_metrics
    exp.hyper_parameterization = CrossValidation
    exp.strategies = strategies
    exp.execute(n_jobs=-1)
