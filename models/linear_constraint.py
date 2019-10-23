# coding: utf-8
# linear_constraint.py - python script to compute the linear + constraint knowledge enriched tensor factorization
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
import numpy as np
from numpy import dot, zeros, array, eye, kron, prod
from numpy.linalg import norm, solve, inv, svd
from scipy.sparse import csr_matrix, issparse
from numpy import linalg as LA


from models.distance_local import *
from models import variables

__version__ = "0.1"
__all__ = ['linear_constraint']

_DEF_MAXITER = 200
_DEF_INIT = 'nvecs'
_DEF_CONV = 1e-6
_DEF_LMBDA = 0
_DEF_ATTR = []
_DEF_NO_FIT = 1e9
_DEF_FIT_METHOD = None
_DEF_lambda_IJ = None
_log = logging.getLogger('linear constraint')


def linear_constraint(X, rank, iteration, **kwargs):

    # ------------ init options ----------------------------------------------
    ainit = kwargs.pop('init', _DEF_INIT)
    _DEF_MAXITER = iteration
    maxIter = kwargs.pop('maxIter', _DEF_MAXITER)

    lmbdaA = kwargs.pop('lambda_A', _DEF_LMBDA)
    lmbdaR = kwargs.pop('lambda_R', _DEF_LMBDA)
    lmbdaE = kwargs.pop('lambda_E', _DEF_LMBDA)

    alpha_a = kwargs.pop('alpha_a', None)
    alpha_r = kwargs.pop('alpha_r', None)
    alpha_lag_mult = kwargs.pop('alpha_lag_mult', None)

    lambda_IJ = kwargs.pop('lambda_IJ', _DEF_lambda_IJ)

    d = distance_local()
    C = d.find_slice_co_var(X)

    compute_fit = kwargs.pop('compute_fit', _DEF_FIT_METHOD)
    P = kwargs.pop('attr', _DEF_ATTR)

    # check frontal slices have same size and are matrices
    sz = X[0].shape
    for i in range(len(X)):
        if X[i].ndim != 2:
            raise ValueError('Frontal slices of X must be matrices')
        if X[i].shape != sz:
            raise ValueError('Frontal slices of X must be all of same shape')


    if compute_fit is None:
        if prod(X[0].shape) * len(X) > _DEF_NO_FIT:
            _log.warn('For large tensors automatic computation of fit is disabled by default\nTo compute the fit, call rescal_als with "compute_fit=True" ')
            compute_fit = False
        else:
            compute_fit = True

    n = sz[0]
    k = len(X)

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
        A1 = np.random.randn(n, rank)
        A2 = np.random.randn(n, rank)
    elif ainit == 'nvecs':
        A1 = []
        A2 = []
    else:
        raise ValueError('Unknown init option ("%s")' % ainit)

    # ------- initialize R and Z ---------------------------------------------
    R = [np.random.rand(rank, rank) for i in range(len(X)) ]

    m_A1 = None
    v_A1 = None
    m_A2 = None
    v_A2 = None
    m_R = None
    v_R = None
    m_IJ = None
    v_IJ = None
    fit_lst = []
    A1_norm = []
    A2_norm = []
    R_norm = []
    lambda_IJ_norm_lst = []
    beta_1 = 0.9
    beta_2 = 0.999
    epsilon = 0.00000001

    lambda_IJ, delta_lambda_IJ, m_IJ, v_IJ = _updateLambdaIJ_adam_no_gemma(X, R, C, lambda_IJ, m_IJ, v_IJ, 0,
                                                                           beta_1,
                                                                           beta_2, alpha=alpha_lag_mult,
                                                                           epsilon=epsilon)
    R, deltaR, m_R, v_R = _updateR(X, A1, A2, lmbdaR, lambda_IJ, C, R, m_R, v_R, 0, beta_1, beta_2,
                                   alpha=alpha_r, epsilon=epsilon)

    a1_norm_current = LA.norm(A1)
    a2_norm_current = LA.norm(A2)
    r_norm_current = LA.norm([LA.norm(r) for r in R])
    A1_norm.append(a1_norm_current)
    A2_norm.append(a2_norm_current)
    R_norm.append(r_norm_current)

    if compute_fit:
        fit = _compute_fit(X, A1, A2, R)
        fit_lst.append(fit)

    #  ------ compute factorization ------------------------------------------
    fitchange = f = 0
    exectimes = []

    for itr in range(maxIter):
        tic = time.time()
        A1, deltaA1, m_A1, v_A1 = _updateA1(X, A1, A2, R, P, lmbdaA, lmbdaE, m_A1, v_A1, itr,
                                            beta_1, beta_2, alpha=alpha_a,
                                            epsilon=epsilon)
        A2, deltaA2, m_A2, v_A2 = _updateA2(X, A1, A2, R, P, lmbdaA, lmbdaE, m_A2, v_A2, itr,
                                            beta_1, beta_2,
                                            alpha=alpha_a,
                                            epsilon=epsilon)
        R, deltaR, m_R, v_R = _updateR(X, A1, A2, lmbdaR, lambda_IJ, C, R, m_R, v_R, itr, beta_1, beta_2,
                                       alpha=alpha_r, epsilon=epsilon)
        lambda_IJ, delta_lambda_IJ, m_IJ, v_IJ = _updateLambdaIJ_adam_no_gemma(X, R, C, lambda_IJ, m_IJ, v_IJ,
                                                                               itr, beta_1, beta_2,
                                                                               alpha=alpha_lag_mult,
                                                                               epsilon=epsilon)

        a1_norm_current = LA.norm(A1)
        a2_norm_current = LA.norm(A2)
        r_norm_current = LA.norm([LA.norm(r) for r in R])
        A1_norm.append(a1_norm_current)
        A2_norm.append(a2_norm_current)
        R_norm.append(r_norm_current)
        lambda_IJ_norm_lst.append(LA.norm(lambda_IJ))

        # compute fit value
        if compute_fit:
            fit = _compute_fit(X, A1, A2, R)
            fit_lst.append(fit)
        else:
            fit = itr

        toc = time.time()
        exectimes.append(toc - tic)

        _log.info('[%3d] fit: %0.5f | delta: %7.1e | secs: %.5f' % (
            itr, fit, fitchange, exectimes[-1]
        ))
        print ('[%3d] fit: %0.5f | delta: %7.1e | secs: %.5f' % (itr, fit, fitchange, exectimes[-1]))





    return lambda_IJ, A1, A2, R, f, itr + 1, array(exectimes)

# Update LambdaIJ
def _updateLambdaIJ(X, R, C, lambda_IJ):
    old_lambda_IJ = np.array(lambda_IJ, copy=True)
    rows, column = len(X), len(X)
    for i in range(rows):
        for j in range(column):
            r = np.power(LA.norm((R[i] - R[j]), 'fro'), 2)
            delta_lambda_IJ = 1 - C[i][j] - r
            lambda_IJ[i][j] =  delta_lambda_IJ # (1 - C[i][j] - np.power(LA.norm((R[i] - R[j]), 'fro'), 2))

    delta = np.sum(np.subtract(lambda_IJ, old_lambda_IJ))
    return  lambda_IJ, delta

def _get_adam_update(gradient, iteration, m_previous, v_previous, alpha, beta_1, beta_2, epsilon):
    first_term = np.multiply(beta_1, m_previous)
    second_term = np.multiply(1 - beta_1, gradient)
    mt = np.add(first_term, second_term)

    first_term = np.multiply(beta_2, v_previous)
    second_term = np.multiply(1 - beta_2, np.power(gradient, 2))
    vt = np.add(first_term, second_term)

    denominator = 1 - (beta_1 ** (iteration + 1))
    mt_hat = np.divide(mt, denominator)

    denominator = 1 - (beta_2 ** (iteration + 1))
    vt_hat = np.divide(vt, denominator)

    term3 = np.divide(mt_hat, np.add(np.sqrt(vt_hat), epsilon))
    term4 = np.multiply(alpha, term3)
    updated_parameter = np.subtract(gradient, term4)
    return updated_parameter, mt, vt


def _updateA1(X, A1, A2, R, P, lmbdaA, lmbdaE, m_previous, v_previous, iteration, beta_1, beta_2, alpha,
              epsilon):
    old_A1 = np.array(A1, copy=True)
    if m_previous is None:
        m_previous = np.zeros((A1.shape[0], A1.shape[1]))
    if v_previous is None:
        v_previous = np.zeros((A1.shape[0], A1.shape[1]))


    """Update step for A"""
    _log.debug('Updating A')
    n, rank = A1.shape
    F = zeros((n, rank), dtype=A1.dtype)
    E = zeros((rank, rank), dtype=A1.dtype)

    A2tA2 = dot(A2.T, A2)

    for i in range(len(X)):
        F += X[i].dot(dot(A2, R[i].T))
        E += dot(R[i], dot(A2tA2, R[i].T))

    F = np.add(np.multiply(lmbdaE, A2), F)

    # regularization
    I = (lmbdaA + lmbdaE) * eye(rank, dtype=A1.dtype)

    gradient = solve(I + E.T, F.T).T
    A1, mt, vt = _get_adam_update(gradient, iteration, m_previous, v_previous, alpha, beta_1, beta_2, epsilon)
    delta_sum = np.sum(np.subtract(A1, old_A1))
    return A1, delta_sum, mt, vt

def _updateA2(X, A1, A2, R, P, lmbdaA, lmbdaE, m_previous, v_previous, iteration, beta_1, beta_2, alpha,
              epsilon):
    old_A2 = np.array(A2, copy=True)
    if m_previous is None:
        m_previous = np.zeros((A2.shape[0], A2.shape[1]))
    if v_previous is None:
        v_previous = np.zeros((A2.shape[0], A2.shape[1]))


    """Update step for A"""
    _log.debug('Updating A')
    n, rank = A2.shape
    F = zeros((n, rank), dtype=A2.dtype)
    E = zeros((rank, rank), dtype=A2.dtype)

    A1tA1 = dot(A1.T, A1)

    for i in range(len(X)):
        F += X[i].T.dot(dot(A1, R[i].T))
        E += dot(R[i].T, dot(A1tA1, R[i].T))

    F = np.add(np.multiply(lmbdaE, A1), F)
    # regularization
    I = (lmbdaA + lmbdaE) * eye(rank, dtype=A2.dtype)

    gradient = solve(I + E.T, F.T).T
    A2, mt, vt = _get_adam_update(gradient, iteration, m_previous, v_previous, alpha, beta_1, beta_2, epsilon)
    delta_sum = np.sum(np.subtract(A2, old_A2))
    return A2, delta_sum, mt, vt

def _updateR(X, A1, A2, lmbdaR, lambda_IJ, C, old_R, m_previous, v_previous, iteration, beta_1, beta_2, alpha,
             epsilon):
    old_R = np.array(old_R, copy=True)
    _log.debug('Updating R (SVD) lambda R: %s' % str(lmbdaR))
    rank = A1.shape[1]

    if m_previous is None:
        m_previous = [np.zeros((rank, rank)) for i in range(len(X))]
    if v_previous is None:
        v_previous = [np.zeros((rank, rank)) for i in range(len(X))]

    U1, S1, V1t = svd(A1, full_matrices=False)
    U2, S2, V2t = svd(A2, full_matrices=False)
    Shat = kron(S2, S1)
    R = []
    m = []
    v = []
    for i in range(len(X)):
        sum_over_j = None
        for j in range(len(X)):
            if sum_over_j is None:
                sum_over_j = lambda_IJ[i][j] * old_R[j].reshape(rank * rank, )
            else:
                sum_over_j = sum_over_j + (lambda_IJ[i][j] * old_R[j].reshape(rank * rank, ))

        sum_over_j = -1.0 * sum_over_j

        lambda_IJ_row_i = lambda_IJ[i, :]
        row_sum = np.sum(lambda_IJ_row_i)
        inverse_term =  Shat ** 2 + lmbdaR - row_sum
        first_term = (Shat / inverse_term).reshape(rank, rank)
        first_term = first_term * dot(U1.T, X[i].dot(U2))
        first_term = dot(V1t.T, dot(first_term, V2t))
        second_term = (sum_over_j / inverse_term).reshape(rank, rank)
        gradient = np.add(first_term,  second_term)

        gtRn, mt_n, vt_n = _get_adam_update(gradient, iteration, m_previous[i], v_previous[i], alpha, beta_1, beta_2, epsilon)

        R.append(gtRn)
        m.append(mt_n)
        v.append(vt_n)

    delta_sum = np.sum([np.sum(np.subtract(i, j)) for i,j in zip(R, old_R)])
    return R, delta_sum, m, v


def _updateLambdaIJ_adam_no_gemma(X, R, C, lambda_IJ, m_previous, v_previous, iteration, beta_1, beta_2, alpha,
                         epsilon):
    if m_previous is None:
        m_previous = np.zeros(lambda_IJ.shape)
    if v_previous is None:
        v_previous = np.zeros(lambda_IJ.shape)

    old_lambda_IJ = np.array(lambda_IJ, copy=True)
    rows, column = len(X), len(X)

    r_ij = np.zeros((rows, column))
    for i in range(rows):
        for j in range(column):
            r_ij[i][j] = np.power(LA.norm((R[i] - R[j]), 'fro'), 2)

    gradient = np.multiply(-1.0, np.subtract(np.add(r_ij, C), 1))
    lambda_IJ, mt, vt = _get_adam_update(gradient, iteration, m_previous, v_previous, alpha, beta_1, beta_2, epsilon)

    delta = LA.norm(np.subtract(lambda_IJ, old_lambda_IJ))
    return lambda_IJ, delta, mt, vt




def _compute_fit(X, A1, A2, R):
    #Compute fit for full slices
    f = 0
    # precompute norms of X
    normX = [sum(M.data ** 2) for M in X]
    sumNorm = sum(normX)

    for i in range(len(X)):
        ARAt = dot(A1, dot(R[i], A2.T))
        f += norm(X[i] - ARAt) ** 2

    return f /sumNorm
