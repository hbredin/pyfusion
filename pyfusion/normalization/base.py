import numpy as np
import scipy.stats
from sklearn.base import BaseEstimator, TransformerMixin

class ZNorm(BaseEstimator, TransformerMixin):
    """
    Z-normalization
    
    Parameters
    ----------
    copy : boolean, optional
        If False, .transform() is applied inplace.
    """
    
    def __init__(self, copy=True):
        super(ZNorm, self).__init__()
        self.copy = copy
    
    def fit(self, X, y=None):
        
        # make X.shape == (n, d)
        X = np.atleast_2d(X)
        N, D = X.shape
        
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        
        return self
    
    def transform(self, X):
        
        X = np.atleast_2d(X)
        nX, D = X.shape
        if D != len(self.mean_):
            raise ValueError('X shape mismatch')
        
        if self.copy:
            X = X.copy()
        
        return (X - self.mean_) / self.std_


class Sigmoid(ZNorm):
    """
    Sigmoid normalization
    
    1/2 (1 + tanh( sigma * (x-mean)/std ))
    
    Parameters
    ----------
    copy : boolean, optional
        If False, .transform() is applied inplace.
    """
    def __init__(self, sigma=0.01, copy=True):
        super(Sigmoid, self).__init__()
        self.copy = copy
        self.sigma = sigma
        
    def transform(self, X):
        return .5*(1+np.tanh(sigma*super(Sigmoid, self).transform(X)))