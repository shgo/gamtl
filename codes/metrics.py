#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Metrics module.
"""
import numpy as np

## metrics for overall performace
def nmse(y_true, y_pred):
    """ Normalized mean squared error for MTL.

    Args:
        :param y_true:
        :param y_pred:

    Returns:
        :returns: value
        :rtype: float
    """
    num = 0
    den = 0
    for t, y_true_t in enumerate(y_true):
        num += np.linalg.norm(y_pred[t].ravel() - y_true_t.ravel())**2/np.std(y_true_t.ravel())
        den += len(y_true_t)
    return num/den

def macc(y_true, y_pred):
    """ Mean of accuracy for MTL.

    Args:
        :param y_true:
        :param y_pred:

    Returns:
        :returns: value
        :rtype: float
    """
    acc_tot = 0
    for y_t, yp_t in zip(y_true, y_pred):
        acc_tot += np.sum(y_t == yp_t)/len(y_t)
    acc_tot = acc_tot / len(y_true)
    return acc_tot

def cc(y_true, y_pred):
    """ CC for MTL.

    Args:
        :param y_true:
        :param y_pred:

    Returns:
        :returns: value
        :rtype: float
    """
    acum = np.zeros(len(y_true))
    for t, y_true_t in enumerate(y_true):
        acum[t] = np.corrcoef(y_true_t, y_pred[t])[0, 1]
    return acum.mean()
