# -*- coding: utf-8 -*-
#pylint:disable=arguments-differ,invalid-name,no-name-in-module,too-many-arguments
"""
Regression methods for Multi-task Learning.
"""
from codes.design import MethodRegression
from codes.methods.amtl import MyAMTLBase
from codes.methods.gamtl import GroupAMTLBase
from codes.methods.gamtl2 import GroupAMTLBase as GAMTLBase2
from codes.methods.mtsgl import MTSGLBase


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


class GroupAMTLcc(GroupAMTLBase, MethodRegression):
    """ GroupAMTL Custo Cruzado for Regression """
    def __init__(self,
                 name='GroupAMTLcc',
                 label='gamtl_cc',
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
                         consider_cross_loss=True,
                         consider_B_restriction=True)


class GroupAMTL2(GAMTLBase2, MethodRegression):
    """ GroupAMTL2 for Regression """
    def __init__(self,
                 name='GroupAMTL2',
                 label='gamtl2',
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


class GroupAMTL2nl(GAMTLBase2, MethodRegression):
    """ GroupAMTL2 No Loss for Regression """
    def __init__(self,
                 name='GroupAMTL2nl',
                 label='gamtl2_nl',
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


class GroupAMTL2nr(GAMTLBase2, MethodRegression):
    """ GroupAMTL2 No Restriction for Regression """
    def __init__(self,
                 name='GroupAMTL2nr',
                 label='gamtl2_nr',
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


class GroupAMTL2nlnr(GAMTLBase2, MethodRegression):
    """ GroupAMTL2 No Loss No Restriction for Regression """
    def __init__(self,
                 name='GroupAMTL2nlnr',
                 label='gamtl2_nlnr',
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


class GroupAMTL2cc(GAMTLBase2, MethodRegression):
    """ GroupAMTL2 Custo Cruzado for Regression """
    def __init__(self,
                 name='GroupAMTL2cc',
                 label='gamtl2_cc',
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
                         consider_cross_loss=True,
                         consider_B_restriction=True)


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


class MTSGL(MTSGLBase, MethodRegression):
    """
    MTSGL for Regression

    Args:
        name (str): name that will be used in Experiment results
        label (str): label that will be used in Experiment results
        normalize (bool): standardize data
        bias (bool): add bias term
        groups (np.array()): [[start g1, start g2, ..., start gG]
                              [end g1, end g2, ..., end gG]
                              [weight g1, weight g2, ... weight gG]]
                    Group representation. Each column of this matrix should
                    contain the start index, stop index, and group weight.
                    Recommended value = np.sqrt(end g - start g).
        r (float): regularization parameter
        opt_method (str): 'proximal_average'

    Reference:
        Liu, X.; Goncalves, A. R.; Cao, P.; Zhao, D.; and Banerjee,A.   2018.
        Modeling  alzheimer’s  disease  cognitive  scoresusing multi-task sparse group lasso.
        Computerized MedicalImaging and Graphics 66:100 – 114
    """
    def __init__(self,
                 name="MT-SGL",
                 label='mt-sgl',
                 normalize=False,
                 bias=False,
                 groups=None,
                 r=1,
                 opt_method='proximal_average'):
        super().__init__(name=name,
                         label=label,
                         normalize=normalize,
                         bias=bias,
                         groups=groups,
                         r=r,
                         glm="Gaussian",
                         opt_method=opt_method)
