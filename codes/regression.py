# -*- coding: utf-8 -*-
#pylint:disable=arguments-differ,invalid-name,no-name-in-module,too-many-arguments
"""
Regression methods for Multi-task Learning.
"""
from codes.design import MethodRegression
from codes.methods.amtl import MyAMTLBase
from codes.methods.gamtl import GroupAMTLBase


class GroupAMTL(GroupAMTLBase, MethodRegression):
    """
    GAMTL Implemented for linear regression.

    Args:
        name (str): name that will be used in Experiment results
        label (str): label that will be used in Experiment results
        normalize (bool): standardize data
        bias (bool): add bias term
        groups
        lambda_1 (float): regularization parameter
        lambda_2 (float): regularization parameter
        lambda_3 (float): regularization parameter

    Attributes:
        W (np.array) = parameter matrix W
        Bs (np.array) = relationship matrices
    """
    def __init__(self,
                 name='GroupAMTL',
                 label='gamtl',
                 normalize=False,
                 bias=False,
                 groups=None,
                 lambda_1=0.01,
                 lambda_2=0.01,
                 lambda_3=0.01):
        super().__init__(name=name,
                         label=label,
                         normalize=normalize,
                         bias=bias,
                         groups=groups,
                         lambda_1=lambda_1,
                         lambda_2=lambda_2,
                         lambda_3=lambda_3,
                         glm="Gaussian",
                         consider_loss=True,
                         consider_B_restriction=True)


class GroupAMTLnl(GroupAMTLBase, MethodRegression):
    """ GroupAMTL No Loss for Regression """
    def __init__(self,
                 name='GroupAMTLnl',
                 label='gamtl_nl',
                 normalize=False,
                 bias=False,
                 groups=None,
                 lambda_1=0.01,
                 lambda_2=0.01,
                 lambda_3=0.01):
        super().__init__(name=name,
                         label=label,
                         normalize=normalize,
                         bias=bias,
                         groups=groups,
                         lambda_1=lambda_1,
                         lambda_2=lambda_2,
                         lambda_3=lambda_3,
                         glm="Gaussian",
                         consider_loss=False,
                         consider_B_restriction=True)


class GroupAMTLnr(GroupAMTLBase, MethodRegression):
    """ GroupAMTL No Restriction for Regression """
    def __init__(self,
                 name='GroupAMTLnr',
                 label='gamtl_nr',
                 normalize=False,
                 bias=False,
                 groups=None,
                 lambda_1=0.01,
                 lambda_2=0.01,
                 lambda_3=0.01):
        super().__init__(name=name,
                         label=label,
                         normalize=normalize,
                         bias=bias,
                         groups=groups,
                         lambda_1=lambda_1,
                         lambda_2=lambda_2,
                         lambda_3=lambda_3,
                         glm="Gaussian",
                         consider_loss=True,
                         consider_B_restriction=False)


class GroupAMTLnlnr(GroupAMTLBase, MethodRegression):
    """ GroupAMTL No Loss No Restriction for Regression """
    def __init__(self,
                 name='GroupAMTLnlnr',
                 label='gamtl_nlnr',
                 normalize=False,
                 bias=False,
                 groups=None,
                 lambda_1=0.01,
                 lambda_2=0.01,
                 lambda_3=0.01):
        super().__init__(name=name,
                         label=label,
                         normalize=normalize,
                         bias=bias,
                         groups=groups,
                         lambda_1=lambda_1,
                         lambda_2=lambda_2,
                         lambda_3=lambda_3,
                         glm="Gaussian",
                         consider_loss=False,
                         consider_B_restriction=False)


class MyAMTL(MyAMTLBase, MethodRegression):
    """ My implementation of AMTL for Regression """
    def __init__(self,
                 name='MyAMTL',
                 label='myamtl',
                 normalize=False,
                 bias=False,
                 mu=0.01,
                 lamb=0.01):
        super().__init__(name=name,
                         label=label,
                         normalize=normalize,
                         bias=bias,
                         mu=mu,
                         lamb=lamb,
                         glm="Gaussian",
                         consider_loss=True,
                         consider_B_restriction=True)

