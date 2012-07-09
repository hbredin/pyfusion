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

import numpy as np
import scipy.stats
from sklearn.base import BaseEstimator, TransformerMixin

class LogLikelihoodRatio(BaseEstimator, TransformerMixin):
    """
    Multivariate log-likelihood ratio estimator log f(X | H) - log f(X | not H)
    
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
    
    def __init__(self, pos_label=1, neg_label=None):
        super(LogLikelihoodRatio, self).__init__()
        self.pos_label = pos_label
        self.neg_label = neg_label
    
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
        pos_indices = np.where(y[:, 0] == self.pos_label)[0]
        if self.neg_label is None:
            neg_indices = np.where(y[:, 0] != self.pos_label)[0]
        else:
            neg_indices = np.where(y[:, 0] == self.neg_label)[0]
        Np = len(pos_indices)
        Nn = len(neg_indices)
        N = Np + Nn
        self.prior_ = (1.*Nn/N, 1.*Np/N)
        
        # extract negative and positive samples
        neg_samples = np.take(X, neg_indices, axis=0)
        pos_samples = np.take(X, pos_indices, axis=0)
        
        # estimate probability density function
        neg_kde = scipy.stats.kde.gaussian_kde(neg_samples)
        pos_kde = scipy.stats.kde.gaussian_kde(pos_samples)
        self.kde_ = (neg_kde, pos_kde)
        
        return self
    
    def transform(self, X):
        
        X = np.atleast_2d(X)
        nX, D = X.shape
        if D != len(self.kde_):
            raise ValueError('X shape mismatch')
        
        x = np.zeros((nX, 1))
        
        # compute positive & negative likelihoods
        neg_kde, pos_kde = self.kde_
        pos_llh = pos_kde(X)
        neg_llh = neg_kde(X)
        
        # samples where both likelihoods are strictly positive
        known = np.where((pos_llh > 0) * (neg_llh > 0))
        x[known] = np.log(pos_llh[known]) - np.log(neg_llh[known])
        
        # samples where both likelihoods are null
        # likelihood ratio == 1 ==> 
        unknown = np.where((neg_llh == 0) * (pos_llh == 0))
        x[unknown] = 0.
        
        # samples where positive likelihood is strictly positive
        # and negative likelihood is null ==> infinity likelihood ratio
        good = np.where((neg_llh == 0) * (pos_llh > 0))
        x[good] = np.inf
        
        # samples where negative likelihood is strictly positive
        # and positive likelihood is null ==> null likelihood ratio
        bad = np.where((neg_llh > 0) * (pos_llh == 0))
        x[bad] = -np.inf
        
        return x

class Posterior(LogLikelihoodRatio):
    """
    Multivariate posterior probabiliy estimator
    
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
    def __init__(self, pos_label=1, neg_label=None):
        super(Posterior, self).__init__(pos_label=pos_label,
                                        neg_label=neg_label)
    
    def transform(self, X):
        lr = np.exp(-super(Posterior, self).transform(X))
        neg_prior, pos_prior = self.prior_
        return 1./(1 + lr*neg_prior/pos_prior)
