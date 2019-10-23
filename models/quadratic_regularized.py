# coding: utf-8
# linear_regularized.py - python script to compute the linear + regularized knowledge enriched tensor factorization
# Current code is built on code base of RESCAL tensor factorization script
# Copyright (C) 2019 Ankur Padia <pankur1@umbc.edu>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import logging
import time

#import matplotlib.pyplot as plt
import numpy as np
from numpy import dot, zeros, array, eye, kron, prod
from numpy.random import rand
from scipy.linalg import pinv, svd, solve, inv, norm
from scipy.sparse import csr_matrix, issparse
from scipy.sparse.linalg import eigsh
from random import randint
from numpy import linalg as LA

from models.distance_local import *

__version__ = "0.1"
__all__ = ['quadratic_regularized']

_DEF_MAXITER = 200
_DEF_INIT = 'nvecs'
_DEF_CONV = 1e-4
_DEF_LMBDA = 0
_DEF_ATTR = []
_DEF_NO_FIT = 1e9
_DEF_FIT_METHOD = None

_log = logging.getLogger('quadratic_regularized')

def _set_random_value_to_one(tensor, count):
    for slice in range(len(tensor)):
        rel_slice = tensor[slice]
        for i in range(count):
            random_row = randint(1, np.shape(rel_slice)[0]-1)
            random_col = randint(1, np.shape(rel_slice)[0]-1)
            tensor[slice][random_row, random_col] = 1

    return tensor

def quadratic_regularized(X, rank, iteration, **kwargs):
    # ------------ init options ----------------------------------------------
    ainit = kwargs.pop('init', _DEF_INIT)
    _DEF_MAXITER = iteration
    maxIter = kwargs.pop('maxIter', _DEF_MAXITER)
    conv = kwargs.pop('conv', _DEF_CONV)
    lmbdaA = kwargs.pop('lambda_A', _DEF_LMBDA)
    lmbdaR = kwargs.pop('lambda_R', _DEF_LMBDA)
    lmbdaRelSimilarity= kwargs.pop('lmbdaRelSimilarity', _DEF_LMBDA)
    compute_fit = kwargs.pop('compute_fit', _DEF_FIT_METHOD)
    P = kwargs.pop('attr', _DEF_ATTR)
    dtype = kwargs.pop('dtype', np.float)


    # ------------- check input ----------------------------------------------
    if not len(kwargs) == 0:
        raise ValueError('Unknown keywords (%s)' % (kwargs.keys()))

    # check frontal slices have same size and are matrices
    sz = X[0].shape
    for i in range(len(X)):
        if X[i].ndim != 2:
            raise ValueError('Frontal slices of X must be matrices')
        if X[i].shape != sz:
            raise ValueError('Frontal slices of X must be all of same shape')
            # if not issparse(X[i]):
            # raise ValueError('X[%d] is not a sparse matrix' % i)

    if compute_fit is None:
        if prod(X[0].shape) * len(X) > _DEF_NO_FIT:
            _log.warn(
                'For large tensors automatic computation of fit is disabled by default\nTo compute the fit, call rescal_als with "compute_fit=True" ')
            compute_fit = False
        else:
            compute_fit = True

    n = sz[0]
    k = len(X)

    _log.debug(
        '[Config] rank: %d | maxIter: %d | conv: %7.1e | lmbda: %7.1e' %
        (rank, maxIter, conv, lmbdaA)
    )
    _log.debug('[Config] dtype: %s / %s' % (dtype, X[0].dtype))

    # ------- convert X and P to CSR ------------------------------------------
    for i in range(k):
        if issparse(X[i]):
            X[i] = X[i].tocsr()
            X[i].sort_indices()
    for i in range(len(P)):
        if issparse(P[i]):
            P[i] = P[i].tocoo().tocsr()
            P[i].sort_indices()

    # ---------- initialize A ------------------------------------------------
    _log.debug('Initializing A')
    if ainit == 'random':
        A = array(rand(n, rank), dtype=dtype)
    elif ainit == 'nvecs':
        S = csr_matrix((n, n), dtype=dtype)
        for i in range(k):
            S = S + X[i]
            S = S + X[i].T
        _, A = eigsh(csr_matrix(S, dtype=dtype, shape=(n, n)), rank)
        A = array(A, dtype=dtype)
    else:
        raise ValueError('Unknown init option ("%s")' % ainit)

    d = distance_local()
    co_var = d.find_slice_co_var(X)
   

    # ------- initialize R and Z ---------------------------------------------
    R = _updateR(X, A, [], co_var, lmbdaR, lmbdaRelSimilarity)
    #Z = _updateZ(A, P, lmbdaV)

    #  ------ compute factorization ------------------------------------------
    fit = fitchange = fitold = f = 0
    exectimes = []
    norm_A = []
    norm_R = []
    fit_lst = []
    for itr in range(maxIter):
        tic = time.time()
        fitold = fit
        A = _updateA(X, A, R, P, None, lmbdaA)
        R = _updateR(X, A, R, co_var, lmbdaR, lmbdaRelSimilarity)
        #Z = _updateZ(A, P, lmbdaV)

        norm_A.append(LA.norm(A))
        norm_R.append(LA.norm([LA.norm(r) for r in R]))
        # compute fit value
        if compute_fit:
            fit = _compute_fit(X, A, R) #, P, Z, lmbdaA, lmbdaR, lmbdaV, iter)
            fit_lst.append(fit)
        else:
            fit = itr

        fitchange = abs(fitold - fit)

        toc = time.time()
        exectimes.append(toc - tic)

        _log.info('[%3d] fit: %0.5f | delta: %7.1e | secs: %.5f' % (
            itr, fit, fitchange, exectimes[-1]
        ))

        print('[%3d] fit: %0.5f | delta: %7.1e | secs: %.5f' % (itr, fit, fitchange, exectimes[-1]))



        #if itr > 0 and fitchange < conv:
        #    break

    '''
    plt.plot(range(maxIter), norm_A)
    plt.plot(range(maxIter), norm_R)
    #plt.plot(range(maxIter), fit_lst)
    plt.xlabel('Iteraiton')
    plt.ylabel('Norm')

    plt.grid()
    plt.legend(['norm-A', 'norm_R', 'reconstruction error'], loc='upper right')
    plt.title('lambda-A:{0};lambda-R:{1};lambda-s:{2};threshold:{3}'.format(lmbdaA, lmbdaR, lmbdaRelSimilarity, threshold_value))
    #plt.show()
    '''
    return [], A, R, f, itr + 1, array(exectimes)


# ------------------ Update A ------------------------------------------------
def _updateA(X, A, R, P, Z, lmbdaA):
    """Update step for A"""
    _log.debug('Updating A')
    n, rank = A.shape
    F = zeros((n, rank), dtype=A.dtype)
    E = zeros((rank, rank), dtype=A.dtype)

    AtA = dot(A.T, A)

    for i in range(len(X)):
        F += X[i].dot(dot(A, R[i].T)) + X[i].T.dot(dot(A, R[i]))
        E += dot(R[i], dot(AtA, R[i].T)) + dot(R[i].T, dot(AtA, R[i]))

    # regularization
    I = lmbdaA * eye(rank, dtype=A.dtype)

    # attributes
    #for i in range(len(Z)):
    #    F += P[i].dot(Z[i].T)
    #    E += dot(Z[i], Z[i].T)

    # finally compute update for A
    A = solve(I + E.T, F.T).T
    #A = dot(F, inv(I + E))
    #_log.debug('Updated A lambda_A:%f, dtype:%s' % (lmbdaA, A.dtype))
    return A



# ------------------ Update R ------------------------------------------------

def _updateR(X, A, R, co_var, lmbdaR, lmbdaRelSimilarity):

    _log.debug('Updating R (SVD) lambda R: %s' % str(lmbdaR))

    rank = A.shape[1]
    U, S, Vt = svd(A, full_matrices=False)
    Shat = kron(S, S)
    updated_R = []

    if len(R) == 0 or R is None:
        updated_R = np.array([np.random.rand(rank, rank) for i in range(len(X))])
        return updated_R

        #for i in range(len(X)):
        #    Rn = first_term * dot(U.T, X[i].dot(U))
        #    Rn = dot(Vt.T, dot(Rn, Vt))
        #    updated_R.append(Rn)
        #return updated_R

    #inverse_term = Shat ** 2 + lmbdaR  # + lmbdaRelSimilarity * np.sum(co_var[i, :] - 1.0)
    #Shat = (Shat / inverse_term).reshape(rank, rank)

    for i in range(len(X)):
        inverse_term = Shat ** 2 + lmbdaR + lmbdaRelSimilarity * np.sum(co_var[i, :])
        first_term = (Shat / inverse_term).reshape(rank, rank)
        Rn = first_term * dot(U.T, X[i].dot(U))
        Rn = dot(Vt.T, dot(Rn, Vt))
        Rn = np.add(Rn,((lmbdaRelSimilarity * np.sum([co_var[i][j] * R[j].reshape(rank * rank, ) for j in range(len(X))], axis=0))/inverse_term).reshape(rank, rank))
        updated_R.append(Rn)

        #cij_vec_Rj =
        #second_term = np.sum(cij_vec_Rj, axis=0)
        #second_term = lmbdaRelSimilarity * second_term
        #second_term = second_term / inverse_term
        #second_term = second_term.reshape(rank, rank)

        #Rn = first_term_Rn # + second_term

        #Rn = Shat * dot(U.T, X[i].dot(U))
        #Rn = dot(Vt.T, dot(Rn, Vt))
        #R.append(Rn

        #if len(R) == 0:
        #    Rn = Shat * dot(U.T, X[i].dot(U))
        #else:
        #    rel_sum_similarity = []
        #    for k in range(len(R)):
        #        if k == i:
        #            continue
        #        weight = co_var[i][k]
        #        if len(rel_sum_similarity) == 0:
        #            rel_sum_similarity = lmbdaRelSimilarity * weight * R[k]
        #        else:
        #            rel_sum_similarity = rel_sum_similarity + lmbdaRelSimilarity * weight * R[k]
        #
        #    + rel_sum_similarity

        #Rn =

        #R.append(Rn)

    return updated_R

# ------------------ Update Z ------------------------------------------------
def _updateZ(A, P, lmbdaZ):
    Z = []
    if len(P) == 0:
        return Z
    #_log.debug('Updating Z (Norm EQ, %d)' % len(P))
    pinvAt = inv(dot(A.T, A) + lmbdaZ * eye(A.shape[1], dtype=A.dtype))
    pinvAt = dot(pinvAt, A.T).T
    for i in range(len(P)):
        if issparse(P[i]):
            Zn = P[i].tocoo().T.tocsr().dot(pinvAt).T
        else:
            Zn = dot(pinvAt.T, P[i])
        Z.append(Zn)
    return Z


def _compute_fit(X, A, R): # P, Z, lmbdaA, lmbdaR, lmbdaZ):
    """Compute fit for full slices"""
    f = 0
    # precompute norms of X
    normX = [sum(M.data ** 2) for M in X]
    sumNorm = sum(normX)

    for i in range(len(X)):
        ARAt = dot(A, dot(R[i], A.T))
        f += norm(X[i] - ARAt) ** 2

    return 1 - f / sumNorm