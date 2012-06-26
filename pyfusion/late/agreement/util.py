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
Utility functions
"""

import scipy.stats
import numpy as np

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

if __name__ == "__main__":
    import doctest
    doctest.testmod()
