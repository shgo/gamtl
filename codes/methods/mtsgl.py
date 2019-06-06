
import time
from abc import ABCMeta
import numpy as np
import numpy.linalg as nplin
from codes.design import Method
from codes.optimization import solve_fista as fista_new

_EPS_W = 1e-3
MAX_ITER = 100

class MTSGLBase(Method):
    """
    Multitask Sparse Group Lasso.
    """
    __metaclass__ = ABCMeta
    def __init__(self,
                 name="MT-SGL",
                 label='mt-sgl',
                 normalize=False,
                 bias=False,
                 groups=None,
                 r=1,
                 threshold=0.5,
                 glm="Gaussian",
                 opt_method='proximal_average'):
        super().__init__(name, label, normalize, bias)
        assert groups is not None
        assert groups.shape[0] == 3
        assert groups.shape[1] > 0
        self.groups = groups
        assert isinstance(glm, str)
        self.glm = glm
        assert opt_method in ('proximal_average', 'proximal_composition')
        self.opt_method = opt_method
        assert r >= 0
        self.r = r
        assert threshold >= 0.
        self.threshold = threshold
        self.W = None

    def fit( self, X, y, max_iter=MAX_ITER):
        X = self.normalize_data(X)
        X = self.add_bias(X)
        ntasks = len(y)
        X = X[0]
        glms = [{'glm': self.glm, 'offset': 0} for t in range(ntasks)]
        dimension = X.shape[1]
        Wvec = np.random.randn( dimension * ntasks ) * 1e-4
        # apply offset in y's if needed    
        #for k in xrange(0,self.ntasks):
        #    if np.abs(self.glms[k]['offset']) > 0:
        #        ytr[k] = self.glms[k]['offset'] - ytr[k]
        cont = 1
        start = time.time()
        while cont <= max_iter:
            W_old = Wvec.copy()
            Wvec= fista_new.SolveFISTA( Wvec, X, y, self.groups, self.r, glms, self.opt_method)
            diff_W = nplin.norm(Wvec - W_old)
            if diff_W < _EPS_W:
                break
            cont = cont + 1

        # print 'cont: %d'%(cont)
        self.W = np.reshape( Wvec,( dimension, ntasks ), order="F" )
        return (self.W, np.zeros(100), time.time() - start)

    def set_params(self, r=1):
        """
        Configura os parâmetros de execução do metodo.
        Args:
            mu (float): mu parameter.
                Default: 0.01.
            lamb (float): lambda parameter.
                Default: 0.1.
        """
        assert r >= 0
        self.r = r

    def get_params(self):
        """
        Retorna os parâmetros de execução do metodo.
        Return
            params (dict):
        """
        ret = {'r': self.r}
        return ret

    def get_resul(self):
        res = {'W': self.W}
        return res
