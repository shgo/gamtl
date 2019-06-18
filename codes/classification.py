# -*- coding: utf-8 -*-
#pylint:disable=too-many-arguments
"""
Classification methods for Multi-Task Learning.

"""

from codes.design import MethodClassification
from codes.methods.amtl import MyAMTLBase
from codes.methods.gamtl import GroupAMTLBase


class GroupAMTL(GroupAMTLBase, MethodClassification):
    """ GroupAMTL """

    def __init__(self,
                 name='GroupAMTL',
                 label='gamtl',
                 normalize=False,
                 bias=False,
                 threshold=0.5,
                 groups=None,
                 lambda_1=0.01,
                 lambda_2=0.01,
                 lambda_3=0.01,
                 cache=True,
                 vary=False,
                 glm="Binomial"):
        super().__init__(
            name=name,
            label=label,
            normalize=normalize,
            bias=bias,
            groups=groups,
            lambda_1=lambda_1,
            lambda_2=lambda_2,
            lambda_3=lambda_3,
            glm=glm,
            cache=cache,
            vary=vary,
            consider_loss=True,
            consider_B_restriction=True)
        self.threshold = threshold


class GroupAMTLnl(GroupAMTLBase, MethodClassification):
    """ GroupAMTL No Loss """

    def __init__(self,
                 name='GroupAMTLnl',
                 label='gamtl_nl',
                 normalize=False,
                 bias=False,
                 threshold=0.5,
                 groups=None,
                 lambda_1=0.01,
                 lambda_2=0.01,
                 lambda_3=0.01,
                 cache=True,
                 vary=False,
                 glm="Binomial"):
        super().__init__(
            name=name,
            label=label,
            normalize=normalize,
            bias=bias,
            threshold=threshold,
            groups=groups,
            lambda_1=lambda_1,
            lambda_2=lambda_2,
            lambda_3=lambda_3,
            glm=glm,
            cache=cache,
            vary=vary,
            consider_loss=False,
            consider_B_restriction=True)


class GroupAMTLnr(GroupAMTLBase, MethodClassification):
    """ GroupAMTL No Restriction """

    def __init__(self,
                 name='GroupAMTLnr',
                 label='gamtl_nr',
                 normalize=False,
                 bias=False,
                 threshold=0.5,
                 groups=None,
                 lambda_1=0.01,
                 lambda_2=0.01,
                 lambda_3=0.01,
                 cache=True,
                 vary=False,
                 glm="Binomial"):
        super().__init__(
            name=name,
            label=label,
            normalize=normalize,
            bias=bias,
            threshold=threshold,
            groups=groups,
            lambda_1=lambda_1,
            lambda_2=lambda_2,
            lambda_3=lambda_3,
            glm=glm,
            cache=cache,
            vary=vary,
            consider_loss=True,
            consider_B_restriction=False)


class GroupAMTLnlnr(GroupAMTLBase, MethodClassification):
    """ GroupAMTL No Loss No Restriction """

    def __init__(self,
                 name='GroupAMTLnlnr',
                 label='gamtl_nlnr',
                 normalize=False,
                 bias=False,
                 threshold=0.5,
                 groups=None,
                 lambda_1=0.01,
                 lambda_2=0.01,
                 lambda_3=0.01,
                 cache=True,
                 vary=False,
                 glm="Binomial"):
        super().__init__(
            name=name,
            label=label,
            normalize=normalize,
            bias=bias,
            threshold=threshold,
            groups=groups,
            lambda_1=lambda_1,
            lambda_2=lambda_2,
            lambda_3=lambda_3,
            glm=glm,
            cache=cache,
            vary=vary,
            consider_loss=False,
            consider_B_restriction=False)


class MyAMTL(MyAMTLBase, MethodClassification):
    """
    AMTL
        Reference:
        G. Lee, E. Yang, S. Hwang. Asymmetric multi-task learning based on task
        relatedness and loss, In Proceedings of Machine Learning Research (PMLR),
        vol. 48, 2016.
    """

    def __init__(self,
                 name='MyAMTL',
                 label='myamtl',
                 normalize=False,
                 bias=False,
                 threshold=0.5,
                 mu=0.01,
                 lamb=0.01,
                 glm='Binomial'):
        super().__init__(
            name=name,
            label=label,
            normalize=normalize,
            bias=bias,
            mu=mu,
            lamb=lamb,
            glm=glm,
            consider_loss=True,
            consider_B_restriction=True)
        assert isinstance(threshold, float)
        self.threshold = threshold
