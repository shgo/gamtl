

import numpy as np
import codes.optimization.proximal_operator as prxopt

MAX_ITER = 10

def SolveFISTA( xk, A, b, ind, gamma, glms, method, maxIter=MAX_ITER):

    lambda_reg = 1e-6
    STOPPING_GROUND_TRUTH = -1
    STOPPING_SUBGRADIENT = 4
    STOPPING_DEFAULT = STOPPING_GROUND_TRUTH

    ntasks = len(b)
    ndim = int(len(xk)/ntasks)

    stoppingCriterion = STOPPING_DEFAULT
    tolerance = 1e-3

    # Initializing optimization variables
    t_k = 1
    t_km1 = 1
    nIter = 0

    xt_y = np.zeros( ntasks * A.shape[1] )
    for k in range(ntasks):
        xt_y[int(k*ndim):int((k+1)*ndim)] = np.absolute( np.dot(A.T,b[k]) )

    lambda1_max = np.max(xt_y)
    
    lambda1 = gamma*lambda1_max
    larg_l1 = np.maximum( xt_y-lambda1, 0 )

    lambda2_max = prxopt.compute_largest_group_norm( larg_l1, ind, int(ndim), ntasks )
    lambda2 = gamma*lambda2_max

    eta = 0.75
    L = 1
    beta = 2

    keep_going = True
    xkm1 = xk
    ind_work = np.zeros( (ind.shape[0],ind.shape[1]+1) )

    ind_work[:2,:] = np.column_stack( (-1*np.ones( (2,1) ), ind[:2,:] ) )

    hist = list()
    while keep_going and ( nIter < maxIter ):

        hist.append( cost_function( xk, A, b, ind, glms, lambda1, lambda2 ) )

        nIter = nIter + 1
        yk = xk + ((t_km1-1)/float(t_k)) * (xk-xkm1)
        stop_backtrack = 0

        temp = task_cost_function_derivative( yk, A, b, glms )
        
        while not stop_backtrack:

            gk = yk - (1.0/L)*temp
            ind_work[2,:] = np.concatenate( ( np.array( (lambda1/float(L), ) ), ind[2,:] * (lambda2/float(L)) ) )
            if method == 'proximal_composition':
                xkp1 = prxopt.proximal_composition( gk, ind_work, ndim, ntasks )
            elif method =='proximal_average':
                xkp1 = prxopt.proximal_average( gk, ind_work, ndim, ntasks ) 

            temp1 = task_cost_function(xkp1, A, b, glms )
            temp2 = task_cost_function( yk, A, b, glms )
            temp2 += np.dot( (xkp1-yk).T, temp ) + (L/2.0) * np.square( np.linalg.norm(xkp1-yk) )

            if temp1 <= temp2:
                stop_backtrack = True
            else:
                L = L*beta

        if stoppingCriterion == STOPPING_GROUND_TRUTH:
            keep_going = np.linalg.norm(xk-xkp1)>tolerance
        elif stoppingCriterion == STOPPING_SUBGRADIENT:
            sk = L*(yk-xkp1) + np.dot(A,(xkp1-yk))
            keep_going = np.linalg.norm(sk) > tolerance*L*np.maximum(1,np.linalg.norm(xkp1))

        lambda1 = np.maximum(eta*lambda1,lambda_reg)
        lambda2 = np.maximum(eta*lambda2,lambda_reg)

        t_kp1 = 0.5*(1+np.sqrt(1+4*t_k*t_k))

        t_km1 = t_k
        t_k = t_kp1
        xkm1 = xk
        xk = xkp1

    return xk

def cost_function( w, x, y, ind, glms, alpha1, alpha2 ):

    dimension = x.shape[1]
    ntasks = len(y)
    W = np.reshape( w,( dimension, ntasks ), order="F" )
    # f = 0
    f = task_cost_function( w, x, y, glms )

    # L2,1 penalization term
    f21 = np.array([np.linalg.norm(l,ord=2) for l in W]).sum() # L2,1 norm

    # Group Lasso term
    fgl = 0 # group lasso cost
    ngroups = ind.shape[1] # number of groups
    for t in range(ntasks): # group lasso is applied to all tasks independently
        for i in range(ngroups):
            ids = np.arange(ind[0,i],ind[1,i], dtype=np.int)
            fgl += np.sqrt( np.dot(W[ids,t],W[ids,t]) )

    # entire cost function: tasks_loss_function + l21_penalty + group_lasso_penalty
    f = f + alpha1*f21 + alpha2*fgl

    return f



def task_cost_function( w, x, y, glms ):

    ndim = x.shape[1]

    f = 0
    for k in range(len(y)):
        wk = w[k*ndim:(k+1)*ndim]
        loglik = 0.
        if glms[k]['glm'] == 'Gaussian':
            tt =  y[k] - np.dot( x, wk )
            loglik = 0.5*np.dot( tt.T, tt )

        elif glms[k]['glm'] == 'Poisson':
            xb = np.maximum( np.minimum( np.dot(x,wk), 100 ), -100 )   ## avoid overflow
            loglik =  -(y[k] * xb - np.exp( xb ) ).sum()

        elif glms[k]['glm'] == 'Gamma':
            loglik = 0
            for i in range(x.shape[0]):
                loglik = scipy.stats.gamma.logpdf( y[k][i], 1.0/np.dot(x[i,:],wk) )

        if not np.isnan( loglik ):
            f += loglik
        else:
            print("****** WARNING: loglik is nan.")

    return f



def task_cost_function_derivative( w,x,y,glms ):

    ndim = x.shape[1]
    ntasks = len(y)
    tmp = np.zeros(w.shape)
    kappa = 0.5

    # compute gradient for the corresponding task
    for k in range(ntasks):

        wk = w[k*ndim:(k+1)*ndim]
        temp_k = np.zeros(wk.shape)

        if glms[k]['glm'] == 'Gaussian':
            temp_k = -np.dot( (y[k]-np.dot(x,wk)).T, x )

        elif glms[k]['glm'] == 'Poisson':
            xb = np.maximum( np.minimum( np.dot(x,wk), 200 ), -200 )  # avoid overflow
            temp_k = -np.dot( x.T, y[k]-np.exp(xb) )

        elif glms[k]['glm'] == 'Gamma':
            temp_k = kappa * np.dot( x.T, np.reciprocal( np.dot(x,wk.T) - y[k] ) )

        tmp[k*ndim:(k+1)*ndim] = temp_k

    return tmp
