import numpy as np
from ..utils.extmath import fast_dot
from ..utils import check_random_state
from ..utils.validation import check_array
from ._base import _BasePCA

def _sym_decorrelation(W):
    """ Symmetric decorrelation
    i.e. W <- (W * W.T) ^{-1/2} * W
    """
    K = fast_dot(W, W.T)
    s, u = np.linalg.eigh(K)
    return fast_dot(np.dot(u * (1. / np.sqrt(s)), u.T), W)

def _ica_def(X, tol, g, fun_args, max_iter, w_init):
    """ Deflationary FastICA using the Hyvarinen approach

    Used under a PSF-derived license, compatible with BSD.
    """
    W = np.zeros((X.shape[1], X.shape[1]), dtype=X.dtype)

    for j in range(X.shape[1]):
        w = w_init[j, :].copy()
        w /= np.sqrt((w ** 2).sum())

        for _ in range(max_iter):
            wtx = np.dot(w, X.T)
            gwtx, g_wtx = g(wtx, **fun_args)
            w1 = (X * gwtx).mean(axis=0) - g_wtx.mean() * w

            # Decorrelate
            w1 -= np.dot(np.dot(w1, W[:j].T), W[:j])
            w1 /= np.sqrt((w1 ** 2).sum())

            if np.abs(np.abs((w1 * w).sum()) - 1) < tol:
                break

            w = w1

        W[j, :] = w

    return W

class FastICA(_BasePCA):
    """ FastICA: a fast algorithm for Independent Component Analysis.
    """

    def __init__(self, n_components=None, algorithm="parallel", whiten=True,
                 fun="logcosh", fun_args=None, max_iter=200, tol=1e-4,
                 w_init=None, random_state=None):
        self.n_components = n_components
        self.algorithm = algorithm
        self.whiten = whiten
        self.fun = fun
        self.fun_args = fun_args
        self.max_iter = max_iter
        self.tol = tol
        self.w_init = w_init
        self.random_state = random_state

    def _fit(self, X, compute_sources=False):
        """ Fit the model
        """
        X = check_array(X, dtype=[np.float64, np.float32], ensure_min_samples=2)

        random_state = check_random_state(self.random_state)

        fun_args = {} if self.fun_args is None else self.fun_args

        if self.fun == "logcosh":
            g = self._logcosh
        elif self.fun == "exp":
            g = self._exp
        elif self.fun == "cube":
            g = self._cube
        else:
            raise ValueError('Unknown function %s' % self.fun)

        alpha = fun_args.get('alpha', 1.0)
        if alpha <= 1 or alpha >= 2:
            raise ValueError('alpha must be in [1, 2]')

        n_components = self.n_components
        if n_components is None:
            n_components = X.shape[1]
        elif not 1 <= n_components <= X.shape[1]:
            raise ValueError("n_components=%d must be between 1 and "
                             "n_features=%d" % (n_components, X.shape[1]))

        X_mean = X.mean(axis=-1)
        X -= X_mean

        if self.whiten:
            # Centering the data before whitening is mandatory!
            X = np.dot(X, self.components_.T)

        W = self._fit_ica(X, g, fun_args, self.max_iter, self.tol, w_init=self.w_init,
                          random_state=random_state)
        return W

    def _fit_ica(self, X, g, fun_args, max_iter, tol, w_init, random_state):
        """ FastICA main loop.
        """
        W = np.zeros((self.n_components, X.shape[1]), dtype=X.dtype)

        for i in range(self.n_components):
            w = random_state.normal(size=(X.shape[1],))
            w /= np.sqrt((w ** 2).sum())

            for j in range(max_iter):
                wtx = np.dot(w, X.T)
                gwtx, g_wtx = g(wtx, **fun_args)
                w1 = (X * gwtx).mean(axis=0) - g_wtx.mean() * w

                # Decorrelate
                w1 -= np.dot(np.dot(w1, W[:i].T), W[:i])
                w1 /= np.sqrt((w1 ** 2).sum())

                if np.abs(np.abs(np.dot(w1, w)) - 1) < tol:
                    break

                w = w1

            W[i, :] = w

        return W
