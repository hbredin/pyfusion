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
#     PyAnnote is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
# 
#     You should have received a copy of the GNU General Public License
#     along with PyAnnote.  If not, see <http://www.gnu.org/licenses/>.

"""
Agreement between information retrieval (IR) systems
"""

import numpy as np
import scipy.stats
import sys

import util

def kendall_tau(X, pool_depth=0, higher_first=True):
    """
    Kendall's tau
    
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
        Kendall's tau for each pair of systems
    
    """
    
    # rank samples (more relevant to less relevant)
    R = util.rank(X, higher_first=higher_first)
    
    # keep top ranked samples only
    if pool_depth > 0:
        R = R[util.top_ranked(R, pool_depth=pool_depth), :]
    
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
    R = util.rank(X, higher_first=higher_first)
    
    # keep top ranked samples only
    if pool_depth > 0:
        R = R[util.top_ranked(R, pool_depth=pool_depth), :]
    
    # pool size & number of systems
    N, D = R.shape
    
    # initialize returned matrix
    A = np.zeros((D, D))
    
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

if __name__ == "__main__":
    import doctest
    doctest.testmod()
