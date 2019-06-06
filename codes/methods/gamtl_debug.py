#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 11:06:54 2018

@author: goncalves1
"""

import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
from gamtl import GroupAMTL

T = 7
m = [np.random.randint(80, 120) for t in range(T)]
split = (0.6, 0.2, 0.2)
n = 24

W = np.zeros((n, T))
groups = np.array([[0, 12],
                   [12, 24],
                   [np.sqrt(12), np.sqrt(12)]])
noise = 1

grupo_1 = np.arange(int(groups[0, 0]), int(groups[1, 0]))
grupo_2 = np.arange(int(groups[0, 1]), int(groups[1, 1]))

# gerando seed 3
W[:, 4] = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                    12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
# gerando seeds 1 e 2
W[grupo_1, 0] = np.random.randn(12) * 5
W[grupo_2, 2] = np.random.randn(12) * 5

# dificeis 1, 2
W[grupo_1, 1] = np.random.randn(12) * 25
W[grupo_2, 3] = np.random.randn(12) * 25

# dificil 3
W[:, 5] = W[:, 4] + np.random.randn(n) + 5

# tarefa mais chata
W[:, 6] = W[:, 1] + W[:, 3] + np.random.randn(n) * 5


X = []
y = []
for t in range(T):
    X.append(np.random.randn(m[t], n))
    print(X[t].shape)
    temp_y = np.dot(X[t], W[:, t]) + np.random.randn(X[t].shape[0]) * noise
    y.append(temp_y)
    print(y[t].shape)
#X = np.array(X)
#y = np.array(y)
#print(X.shape)
#print(y.shape)
lambda_1 = 0.01
lambda_2 = 0.001
lambda_3 = 0.01
gmtl = GroupAMTL()
gmtl.set_params(lambda_1, lambda_2, lambda_3, groups, 'Gaussian')
gmtl.fit(X, y)
print(gmtl.Bs.shape)

for t in range(T):
    lr = lm.LinearRegression()
    lr.fit(X[t], y[t])
    print('{}'.format(np.vstack((lr.coef_, gmtl.W[:, t], W[:, t])).T))
    print('Norm: {}'.format(np.linalg.norm(lr.coef_- W[:, t])))
    print('Norm: {}'.format(np.linalg.norm(gmtl.W[:, t]- W[:, t])))
    print('\n')



cov_g1 = np.corrcoef(gmtl.W[grupo_1, :], rowvar=False) #, bias=True)
cov_g2 = np.corrcoef(gmtl.W[grupo_2, :], rowvar=False) #, bias=True)

cov_g1 = cov_g1 - np.eye(cov_g1.shape[0])
cov_g2 = cov_g2 - np.eye(cov_g1.shape[0])

cmap = plt.cm.OrRd
cmap.set_bad(color='blue')


axs = [None]*3*len(gmtl.Bs)
nplots = len(axs)

fig = plt.figure(figsize=(10,10))
axs[0] = fig.add_subplot(nplots, 1, 1)
cax = axs[0].matshow(W, interpolation='nearest', cmap = cmap)
#fig.colorbar(cax)

axs[1] = fig.add_subplot(nplots, 1, 2)
cax = axs[1].matshow(cov_g1, interpolation='nearest',  cmap = cmap)
fig.colorbar(cax)

axs[2] = fig.add_subplot(nplots, 1, 3)
cax = axs[2].matshow(cov_g2, interpolation='nearest',  cmap = cmap)
fig.colorbar(cax)



for i, B in enumerate(gmtl.Bs):
    axs[i+3] = fig.add_subplot(nplots, 1, i+4)
    cax = axs[i+3].matshow(B, interpolation='nearest',  cmap = cmap)
    fig.colorbar(cax)
plt.show()
