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
Agreement between information retrieval (IR) systems
"""

import numpy as np
import scipy.stats

def rank(a, higher_first=True):
    """
    Rank relevance scores
    
    Parameters
    ----------
    a : array-like (num_samples, num_systems)
        Relevance scores for each sample and system
    higher_first : bool, optional
        True means higher is better (and therefore lower rank)
        False means the opposite. Defaults to True.
        
    Returns
    -------
    r : array (num_samples, num_systems)
        Samples rank for each system
    
    """
    N, D = a.shape
    r = np.empty((N, D), dtype=np.int64)
    for d in range(D):
        if higher_first:
            r[:, d] = scipy.stats.rankdata(-a[:, d])
        else:
            r[:, d] = scipy.stats.rankdata(a[:, d])
    return r


def top_ranked(r, pool_depth=100):
    """
    Pool top ranked samples
    
    Parameters
    ----------
    r : array-like (num_samples, num_systems)
        Relevance scores ranks
    pool_depth : int, optional
        Pool depth. Defaults to 100.
        
    Returns
    -------
    in_pool : array (num_samples, )
        True if at least one system ranks the sample in its top `pool_depth`
        False otherwise.
    """
    return np.any(r <= pool_depth, axis=1)


def kendall_rank_correlation(X, pool_depth=0, higher_first=True):
    """
    Kendall's Rank Correlation coefficient
    
    Parameters
    ----------
    X : array-like (num_samples, num_systems)
        Sample relevance scores for each system
    higher_first : bool, optional
        True means higher score is more relevant.
        False means the opposite. Defaults to True.
    pool_depth : int, optional
        Fasten computation by focusing on top ranked samples
    
    Returns
    -------
    A : array (num_systems, num_systems)
        Kendall's Rank Correlation coefficient for each pair of systems
    
    """
    
    # rank samples (more relevant to less relevant)
    R = rank(X, higher_first=higher_first)
    
    # keep top ranked samples only
    if pool_depth > 0:
        R = R[top_ranked(R, pool_depth=pool_depth), :]
    
    # pool size & number of systems
    N, D = R.shape
    
    # initialize returned matrix
    A = np.zeros((D, D), dtype=np.float64)
    for d in range(D):
        
        # sort samples based on rank of dth dimension
        order = np.argsort(R[:, d], axis=0, kind='quicksort', order=None)
        
        # since Kendall's tau is symetric, only the lower part of the matrix
        # is computed -- the other part is copied just before returning
        r = R[order, :d]
        
        for i in range(1, N):
            
            # analyzing pairs (i, j) with (j < i)
            # (this makes a total of i pairs)
            
            # number of concordant pairs
            c = np.sum(r[:i, :] < r[i, :], axis=0)
            # number of discordant pairs is directly deduced: d = i - c
            # c - d = 2c - i
            A[d, :d] += 2*c-i
    
    # normalize by total number of pairs
    # and shift to [-1, 1] range
    A = 4*A/(N*(N-1))-1
    
    # fill the higher part of the matrix
    for d in range(D):
        A[d, d] = 1.
        for e in range(d):
            A[e, d] = A[d, e]
    
    return A

def average_precision_correlation(X, pool_depth=0, higher_first=True):
    """
    Average Precision Correlation coefficient
    
    Parameters
    ----------
    X : array-like (num_samples, num_systems)
        Sample relevance scores for each system
    higher_first : bool, optional
        True means higher score is more relevant.
        False means the opposite. Defaults to True.
    pool_depth : int, optional
        Fasten computation by focusing on top ranked samples
    
    Returns
    -------
    A : array (num_systems, num_systems)
        Average precision correlation coefficient for each pair of systems
    
    References
    ----------
        "A New Rank Correlation Coefficient for Information Retrieval"
        Emine Yilmaz, Javed A. Aslam and Stephen Robertson
        Proceedings of the 31st annual international ACM SIGIR conference
        on Research and development in information retrieval (SIGIR 2008).
    
    """
    
    # rank samples (more relevant to less relevant)
    R = rank(X, higher_first=higher_first)
    
    # keep top ranked samples only
    if pool_depth > 0:
        R = R[top_ranked(R, pool_depth=pool_depth), :]
    
    # pool size & number of systems
    N, D = R.shape
    
    # initialize returned matrix
    A = np.zeros((D, D), dtype=np.float64)
    
    # loop on each system
    for d in range(D):
        
        # sort samples based on rank of current (dth) system
        order = np.argsort(R[:, d], axis=0, kind='quicksort', order=None)
        r = R[order, :]
        
        # 'precision at i' 
        # (based on number of samples correctly ranked as more relevant)
        for i in range(1, N):
            A[d, :] += 1. * np.sum(r[:i, :] <= r[i, :], axis=0) / i
    
    # shift to [-1, 1] range
    return 2./(N-1)*A-1

def spearman_rank_correlation(X, pool_depth=0, higher_first=True):
    """
    Spearman's Rank Correlation coefficient
    
    Parameters
    ----------
    X : array-like (num_samples, num_systems)
        Sample relevance scores for each system
    higher_first : bool, optional
        True means higher score is more relevant.
        False means the opposite. Defaults to True.
    pool_depth : int, optional
        Fasten computation by focusing on top ranked samples
    
    Returns
    -------
    A : array (num_systems, num_systems)
        Spearman's rank correlation coefficient for each pair of systems
    
    """
    
    # keep top ranked samples only
    if pool_depth > 0:
        # rank samples (more relevant to less relevant)
        R = rank(X, higher_first=higher_first)
        X = X[top_ranked(R, pool_depth=pool_depth), :]
    
    # why reinvent the wheel? :)
    A, _ = scipy.stats.spearmanr(X, axis=0)
    
    # pool size & number of systems
    N, D = X.shape
    
    # when D == 2, spearmanr returns a scalar
    if D == 2:
        A = np.array([[1., A],[A, 1.]])
    
    return A

def pearson_correlation(X, pool_depth=0, higher_first=True):
    """
    Pearson Product-Moment Correlation coefficient
    
    Parameters
    ----------
    X : array-like (num_samples, num_systems)
        Sample relevance scores for each system
    higher_first : bool, optional
        True means higher score is more relevant.
        False means the opposite. Defaults to True.
    pool_depth : int, optional
        Fasten computation by focusing on top ranked samples
    
    Returns
    -------
    A : array (num_systems, num_systems)
        Pearson product-moment correlation coefficient for each pair of systems
    
    """
    
    # keep top ranked samples only
    if pool_depth > 0:
        # rank samples (more relevant to less relevant)
        R = rank(X, higher_first=higher_first)
        X = X[top_ranked(R, pool_depth=pool_depth), :]
    
    # pool size & number of systems
    N, D = X.shape
    
    # one pair at a time
    A = np.zeros((D, D), dtype=np.float64)
    for i in range(D):
        A[i, i] = 1.
        for j in range(i+1, D):
            A[i, j], _ = scipy.stats.pearsonr(X[:,i], X[:,j])
            # coefficient is symmetric
            A[j, i] = A[i, j]
            
    return A


if __name__ == "__main__":
    import doctest
    doctest.testmod()
