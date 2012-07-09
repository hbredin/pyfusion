#!/usr/bin/env python
# encoding: utf-8

# Copyright 2012 Herve BREDIN (bredin@limsi.fr)

# This file is part of PyFusion.
# 
#     PyFusion is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
# 
#     PyFusion is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
# 
#     You should have received a copy of the GNU General Public License
#     along with PyFusion.  If not, see <http://www.gnu.org/licenses/>.

"""
Bayesian probability estimation
"""

import numpy as np
import scipy.stats
import multiprocessing
from sklearn.base import BaseEstimator, TransformerMixin

def _log_likelihood_ratio((llh_ratio, x, d)):
    """
    Helper function for multi-threaded computation of log-likelihood ratio
    
    Function
    
    Parameters
    ----------
    llh_ratio : LogLikelihoodRatio
    x : array (num_samples, )
    d : int
    
    Returns
    -------
    x : 
    
    """
    
    # compute positive & negative likelihoods
    neg_kde, pos_kde = llh_ratio.kde_[d]
    pos_llh = pos_kde(x)
    neg_llh = neg_kde(x)
    
    # samples where both likelihoods are strictly positive
    known = np.where((pos_llh > 0) * (neg_llh > 0))
    x[known] = np.log(pos_llh[known]) - np.log(neg_llh[known])
    
    # samples where both likelihoods are null
    # likelihood ratio == 1 ==> log likelihood ratio == 0
    unknown = np.where((pos_llh == 0) * (neg_llh == 0))
    x[unknown] = 0.
    
    # samples where positive likelihood is strictly positive
    # and negative likelihood is null ==> infinity likelihood ratio
    good = np.where((neg_llh == 0) * (pos_llh > 0))
    x[good] = np.inf
    
    # samples where negative likelihood is strictly positive
    # and positive likelihood is null ==> zero likelihood ratio
    bad = np.where((neg_llh > 0) * (pos_llh == 0))
    x[bad] = -np.inf
    
    return x

class LogLikelihoodRatio(BaseEstimator, TransformerMixin):
    """
    Log-likelihood ratio estimator log f(x | H) - log f(x | not H)
    
    ... based on non-parametric Gaussian-kernel density estimator
    
    Parameters
    ----------
    pos_label : int, optional
        Label of positive samples. Defaults to 1.
    neg_label : int, optional
        Label of negative samples.
        If it is not provided, all samples that are not positive samples
        are considered as negative samples (this is the default behavior).
    """
    
    def __init__(self, pos_label=1, neg_label=None, parallel=False):
        super(LogLikelihoodRatio, self).__init__()
        self.pos_label = pos_label
        self.neg_label = neg_label
        self.parallel = parallel
    
    def fit(self, X, y=None):
        
        # make X.shape == (n, d) and y.shape == (n, 1)
        X = np.atleast_2d(X)
        y = np.atleast_2d(y)
        N, D = X.shape
        n, d = y.shape
        if N != n or d != 1:
            raise ValueError('X-y shape mismatch (%d, %d) vs. (%d, %d)' % 
                             (N, D, n, d))
        
        # prior probability
        pos_indices = np.where(y[:, 0] == self.pos_label)
        if self.neg_label is None:
            neg_indices = np.where(y[:, 0] != self.pos_label)
        else:
            neg_indices = np.where(y[:, 0] == self.neg_label)
        Np = len(pos_indices[0])
        Nn = len(neg_indices[0])
        N = Np + Nn
        self.prior_ = (1.*Nn/N, 1.*Np/N)
        
        # gaussian kernel density estimation
        self.kde_ = []
        
        for d in range(D):
            
            # extract negative and positive samples
            neg_samples = X[neg_indices, d]
            pos_samples = X[pos_indices, d]
            
            # estimate probability density function
            neg_kde = scipy.stats.kde.gaussian_kde(neg_samples)
            pos_kde = scipy.stats.kde.gaussian_kde(pos_samples)
            self.kde_.append((neg_kde, pos_kde))
        
        return self
    
    def transform(self, X):
        
        X = np.atleast_2d(X)
        nX, D = X.shape
        if D != len(self.kde_):
            raise ValueError('X shape mismatch')
        
        X = X.copy()
        
        if self.parallel:
            pool = multiprocessing.Pool(processes=None)
            tX = pool.map(_log_likelihood_ratio, [(self, X[:, d], d) 
                                                  for d in range(D)])
            pool.close()
            for d, tx in enumerate(tX):
                X[:, d] = tx
        else:
            for d in range(D):
                X[:, d] = _log_likelihood_ratio((self, X[:, d], d))
        
        return X


class Posterior(LogLikelihoodRatio):
    """
    Posterior probabiliy estimator
    
    ... based on non-parametric Gaussian-kernel density estimator
    
    Parameters
    ----------
    pos_label : int, optional
        Label of positive samples. Defaults to 1.
    neg_label : int, optional
        Label of negative samples.
        If it is not provided, all samples that are not positive samples
        are considered as negative samples (this is the default behavior).
    """
    
    def __init__(self, pos_label=1, neg_label=None, parallel=False):
        super(Posterior, self).__init__(pos_label=pos_label,
                                        neg_label=neg_label, 
                                        parallel=parallel)
    
    def transform(self, X):
        """
        Compute posterior probabilities
        
        Parameters
        ----------
        X : array-like (num_samples, num_systems)
            Sample relevance scores for each system
        
        Returns
        -------
        posterior : array (num_samples, num_systems)
            Posterior probabilities for each system
        
        """
        
        lr = np.exp(-super(Posterior, self).transform(X))
        neg_prior, pos_prior = self.prior_
        return 1./(1 + lr*neg_prior/pos_prior)

if __name__ == "__main__":
    import doctest
    doctest.testmod()
