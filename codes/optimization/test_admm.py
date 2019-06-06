#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 16:03:23 2018

@author: goncalves1
"""
import numpy as np
from admm import ADMM
from admm import ADMM_Lasso


m  = 5000 # number of examples
n  = 20     # number of features

#x0 = np.maximum(np.random.randn(n, 1), 0)
x0 = np.random.randn(n, 1)
x0[int(0.2*n):] = 0
A = np.random.randn(m, n)
b = np.dot(A, x0) + 0.1*np.random.randn(m,1)

#obj = ADMM(lamb=1)
obj = ADMM_Lasso(lamb=0.3, rho=1.0, alpha=1.0)
w_hat = obj.fit(A, b)

print('real | estimated :')
for (i, j) in zip(x0, w_hat):
    print('%.5f | %.5f' % (i, j))


print('norm of diff: {}'.format(np.linalg.norm(w_hat - x0)))
#lambda_max = norm(A'*b, 'inf')
#lambda_reg = 0.1*lambda_max;
