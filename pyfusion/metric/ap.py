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

def average_precision(X, y, relevant_label=1):
    """
    Average precision (for fully annotated data)
    
    Parameters
    ----------
    X : array-like (num_samples, num_systems)
        Sample relevance scores for each system
    y : array-like (num_samples, )
        Relevance judgments
    relevant_label : int, optional
        Label of relevant samples. Defaults to 1.
        All other labels means irrelevant.
    
    Returns
    -------
    ap : array (num_systems, )
        Average precision for each system
    
    """
    
    X = np.atleast_2d(X)
    y = np.atleast_2d(y)
    N, D = X.shape
    n, d = y.shape
    if N != n or d != 1:
        raise ValueError('X-y shape mismatch (%d, %d) vs. (%d, %d)' % 
                         (N, D, n, d))
    
    # rank samples by relevance scores
    R = np.argsort(-X, axis=0)
    
    # total number of relevant samples
    N_rel = np.sum(y == relevant_label)
    
    relevant = np.empty((N, D), dtype=np.bool)
    for d in range(D):
        r = R[:, d]
        relevant[:, d] = np.ravel(np.take(y, r, axis=0) == relevant_label)
    
    # relevant_at = np.cumsum(relevant, axis=0)
    # precision_at = relevant_at / np.arange(1, N+1)
    # ap = np.sum(relevant * precision_at) / N_rel
    return np.sum((1. * relevant * np.cumsum(relevant, axis=0) / 
                  np.arange(1, N+1).reshape((-1, 1))), axis=0) / N_rel

def induced_average_precision(X, y, relevant_label=1, unknown_label=-1):
    """
    Induced average precision (for partially annotated data)
    
    Parameters
    ----------
    X : array-like (num_samples, num_systems)
        Sample relevance scores for each system
    y : array-like (num_samples, )
        Relevance judgments
    relevant_label : int, optional
        Label of relevant samples. Defaults to 1.
    unknown_label : int, optional
        Label of unjudged samples. Defaults to -1.
    
    Returns
    -------
    ap : array (num_systems, )
        Induced average precision for each system
    
    """
    
    X = np.atleast_2d(X)
    y = np.atleast_2d(y)
    N, D = X.shape
    n, d = y.shape
    if N != n or d != 1:
        raise ValueError('X-y shape mismatch (%d, %d) vs. (%d, %d)' % 
                         (N, D, n, d))
    
    known = np.where(y != unknown_label)[0]
    return average_precision(X[known, :], y[known, :],
                             relevant_label=relevant_label)


def inferred_average_precision(X, y, unknown_label=-1, relevant_label=1):
    """
    Inferred average precision (for partially annotated data)
    
    Parameters
    ----------
    X : array-like (num_samples, num_systems)
        Sample relevance scores for each system
    y : array-like (num_samples, )
        Relevance judgments
    relevant_label : int, optional
        Label of relevant samples. Defaults to 1.
    unknown_label : int, optional
        Label of unjudged samples. Defaults to -1.
    
    Returns
    -------
    ap : array (num_systems, )
        Inferred average precision for each system
    
    Reference
    ---------
        "Inferred AP : Estimating Average Precision with Incomplete Judgments"
        Emine Yilmaz and  Javed A. Aslam
    
    """
    
    X = np.atleast_2d(X)
    y = np.atleast_2d(y)
    N, D = X.shape
    n, d = y.shape
    if N != n or d != 1:
        raise ValueError('X-y shape mismatch (%d, %d) vs. (%d, %d)' % 
                         (N, D, n, d))
    
    p = 1. * (y != unknown_label)
    return extended_inferred_average_precision(X, y, p,
                                               unknown_label=unknown_label,
                                               relevant_label=relevant_label)
    
def extended_inferred_average_precision(X, y, p, unknown_label=-1,
                                                 relevant_label=1):
    """
    Extended inferred average precision 
    (for partially & stratified annotated data)
    
    Parameters
    ----------
    X : array-like (num_samples, num_systems)
        Sample relevance scores for each system
    y : array-like (num_samples, )
        Relevance judgments
    p : array-like (num_samples, )
        Sample pool. Identifies which stratum each sample comes from.
    
    relevant_label : int, optional
        Label of relevant samples. Defaults to 1.
    unknown_label : int, optional
        Label of unjudged samples. Defaults to -1.
    
    Returns
    -------
    ap : array (num_systems, )
        Extended inferred average precision for each system
    
    Reference
    ---------
        "A Simple and Efficient Sampling Method for Estimating AP and NDCG"
        Emine Yilmaz, Evangelos Kanoulas, and Javed A. Aslam
        Proceedings of the 31st annual international ACM SIGIR conference
        on Research and development in information retrieval (SIGIR 2008).
        (www.ccs.neu.edu/home/ekanou/research/papers/mypapers/sigir08b.pdf)
    
    """
    
    eps = 1e-10
    
    X = np.atleast_2d(X)
    y = np.atleast_2d(y)
    p = np.atleast_2d(p)
    N, D = X.shape
    n, d = y.shape
    if N != n or d != 1:
        raise ValueError('X-y shape mismatch (%d, %d) vs. (%d, %d)' % 
                         (N, D, n, d))
    n, d = p.shape
    if N != n or d != 1:
        raise ValueError('X-p shape mismatch (%d, %d) vs. (%d, %d)' % 
                         (N, D, n, d))
    
    # rank samples by relevance scores
    R = np.argsort(-X, axis=0)
    
    pools = [pool for pool in sorted(set(p[:,0])) if pool > 0]
    N_pools = len(pools)
    
    # pooled[p,r,d] == True means rth sample (using ranking from dth system)
    #                             belongs to pth pool
    pooled = np.empty((N_pools, N, D), dtype=np.bool)
    
    # judged[p,r,d] == True means rth sample (using ranking from dth system)
    #                             groundtruh is known (ie. was annotated)
    judged = np.empty((N_pools, N, D), dtype=np.bool)
    
    # relevant[p,r,d] == True means rth sample (using ranking from dth system)
    #                               is a relevant sample from pth pool
    relevant = np.empty((N_pools, N, D), dtype=np.bool)
    
    # Relevant[r,d] == True means rth sample (using ranking from dth system)
    #                             is a relevant sample
    Relevant = np.empty((N, D), dtype=np.bool)
    
    for d in range(D):
        r = R[:, d]
        # pool sorted by score
        p_ = np.take(p, r, axis=0)
        # label sorted by score
        y_ = np.take(y, r, axis=0)
        Relevant[:, d] = y_[:, 0] == relevant_label
        for i, pool in enumerate(pools):
            in_pool = np.ravel(p_ == pool)
            pooled[i,:,d] = in_pool
            relevant[i,:,d] = in_pool * Relevant[:, d]
            judged[i,:,d] = in_pool * (y_[:,0]!=unknown_label)
    
    # pooled_at[p,r,d] is the number of p-pooled samples above rank r
    #                  (using ranking from dth system)
    pooled_at = np.cumsum(pooled, axis=1)
    
    # Pooled_at[r,d] is the total number of pooled samples above rank r
    #                (using ranking from dth system)
    Pooled_at = np.sum(pooled_at, axis=0)
    
    # relevant_at[p,r,d] is the number of relevant p-pooled samples above rank r
    #                    (using ranking from dth system)
    relevant_at = np.cumsum(relevant, axis=1)
    
    # judged_at[p,r,d] is the number of judged p-pooled samples above rank r
    #                    (using ranking from dth system)
    judged_at = np.cumsum(judged, axis=1)
    
    # Expected precision above rank /k/:
    #    * Probability of choosing pool /p/ at rank /k/
    #      ==> pooled_at[p,k,:] / k
    #    * Expected precision above rank /k/ within pool /p/
    #      ==> relevant_at[p,k,:] / judged_at[p,k,:]
    at = np.arange(1, N+1).reshape((-1, 1))
    precision_above = np.sum(1.*pooled_at*(relevant_at+eps)/(judged_at+2*eps), axis=0) / at
    
    # Precision @ N
    precision_at = 1./at + (at-1.)/at * precision_above
    
    # Expected number of relevant samples
    N_rel = np.sum(1. * pooled_at[:,-1, 0] * (relevant_at[:,-1, 0]+eps) / 
                                             (judged_at[:, -1, 0]+2*eps))
    
    # Average precision @ relevant samples
    return np.sum(Relevant * precision_at, axis=0) / N_rel
    
