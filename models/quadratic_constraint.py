# coding: utf-8
# quadratic_constraint.py - python script to compute the quadratic + constraint knowledge enriched tensor factorization
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
from scipy.sparse.linalg import eigsh
from numpy.random import rand
from numpy import linalg as LA

from models.distance_local import distance_local

__version__ = "0.1"
__all__ = ['quadratic_constraint']

_DEF_MAXITER = 200
_DEF_INIT = 'nvecs'
_DEF_CONV = 1e-6
_DEF_LMBDA = 0
_DEF_GEMMA = 0.1
_DEF_ATTR = []
_DEF_NO_FIT = 1e9
_DEF_FIT_METHOD = None
_DEF_lambda_IJ = None
_log = logging.getLogger('quadratic_constraint function')


def quadratic_constraint(X, rank, iteration, **kwargs):

    # ------------ init options ----------------------------------------------
    ainit = kwargs.pop('init', _DEF_INIT)
    _DEF_MAXITER = iteration
    maxIter = kwargs.pop('maxIter', _DEF_MAXITER)
    conv = kwargs.pop('conv', _DEF_CONV)

    lmbdaA = kwargs.pop('lambda_A', _DEF_LMBDA)
    lmbdaR = kwargs.pop('lambda_R', _DEF_LMBDA)
    gemma = kwargs.pop('gemma', _DEF_LMBDA) #1000000

    alpha_a = kwargs.pop('alpha_a', None)
    alpha_r = kwargs.pop('alpha_r', None)
    alpha_lag_mult = kwargs.pop('alpha_lag_mult', None)

    lambda_IJ = kwargs.pop('lambda_IJ', _DEF_lambda_IJ)

    d = distance_local()
    C = d.find_slice_co_var(X)

    compute_fit = kwargs.pop('compute_fit', _DEF_FIT_METHOD)
    P = kwargs.pop('attr', _DEF_ATTR)
    dtype = kwargs.pop('dtype', np.float)

    # ------------- check input ----------------------------------------------
    #if not len(kwargs) == 0:
    #    raise ValueError('Unknown keywords (%s)' % (kwargs.keys()))

    # check frontal slices have same size and are matrices
    sz = X[0].shape
    for i in range(len(X)):
        if X[i].ndim != 2:
            raise ValueError('Frontal slices of X must be matrices')
        if X[i].shape != sz:
            raise ValueError('Frontal slices of X must be all of same shape')
        #if not issparse(X[i]):
            #raise ValueError('X[%d] is not a sparse matrix' % i)

    if compute_fit is None:
        if prod(X[0].shape) * len(X) > _DEF_NO_FIT:
            _log.warn('For large tensors automatic computation of fit is disabled by default\nTo compute the fit, call rescal_als with "compute_fit=True" ')
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
        A = np.random.randn(n, rank)
    elif ainit == 'nvecs':
        S = csr_matrix((n, n), dtype=dtype)
        for i in range(k):
            S = S + X[i]
            S = S + X[i].T
        _, A = eigsh(csr_matrix(S, dtype=dtype, shape=(n, n)), rank)
        A = array(A, dtype=dtype)
    else:
        raise ValueError('Unknown init option ("%s")' % ainit)

    # ------- initialize R and Z ---------------------------------------------
    R = [np.random.rand(rank, rank) for i in range(len(X)) ]

    m_A = None
    v_A = None
    m_R = None
    v_R = None
    m_IJ = None
    v_IJ = None
    deltaA_lst = []
    deltaR_lst = []
    delta_lambda_IJ_lst = []
    fit_lst = []
    A_norm = []
    R_norm = []
    lambda_IJ_norm_lst = []
    beta_1 = 0.9
    beta_2 = 0.999
    epsilon = 0.00000001
    iteration = 0

    '''
    if gd_algo == 'adam_no_gemma':
        lambda_IJ, delta_lambda_IJ, m_IJ, v_IJ = _updateLambdaIJ_adam_no_gemma(X, R, C, lambda_IJ, gemma, m_IJ, v_IJ, 0, beta_1,
                                                                      beta_2, alpha=alpha_lag_mult,
                                                                      epsilon=epsilon)
        R, deltaR, m_R, v_R = _updateR_adam_no_gemma(X, A, lmbdaR, lambda_IJ, C, R, m_R, v_R, 0, beta_1, beta_2, alpha=alpha_r,
                                            epsilon=epsilon)
    else:
        lambda_IJ, delta_lambda_IJ, m_IJ, v_IJ = _updateLambdaIJ_adam(X, R, C, lambda_IJ, gemma, m_IJ, v_IJ, 0, beta_1,
                                                                      beta_2, alpha_lag_mult,
                                                                      epsilon)
        R, deltaR, m_R, v_R = _updateR_adam(X, A, lmbdaR, lambda_IJ, C, R,  m_R, v_R, 0, beta_1, beta_2, alpha=alpha_r, epsilon=epsilon)
        #lambda_IJ, delta_lambda_IJ = _updateLambdaIJ_adam(X, R, C, lambda_IJ, gemma)
    '''

    lambda_IJ, delta_lambda_IJ, m_IJ, v_IJ = _updateLambdaIJ_adam_no_gemma(X, R, C, lambda_IJ, gemma, m_IJ, v_IJ, 0,
                                                                           beta_1,
                                                                           beta_2, alpha=alpha_lag_mult,
                                                                           epsilon=epsilon)
    R, deltaR, m_R, v_R = _updateR_adam_no_gemma(X, A, lmbdaR, lambda_IJ, C, R, m_R, v_R, 0, beta_1, beta_2,
                                                 alpha=alpha_r,
                                                 epsilon=epsilon)

    a_norm_current = LA.norm(A)
    r_norm_current = LA.norm([LA.norm(r) for r in R])
    A_norm.append(a_norm_current)
    R_norm.append(r_norm_current)

    if compute_fit:
        fit = _compute_fit(X, A, R, C, lambda_IJ, lmbdaA, lmbdaR)
        fit_lst.append(fit)

    #  ------ compute factorization ------------------------------------------
    fit = fitchange = fitold = f = 0
    exectimes = []

    for itr in range(maxIter):
        #print ('Iterating {0}'.format(itr))
        tic = time.time()

        '''
        if gd_algo == 'adam':
            A, deltaA, m_A, v_A = _updateA_adam(X, A, R, P, lmbdaA, m_A, v_A, itr, beta_1, beta_2, alpha=alpha_a, epsilon=epsilon)
            R, deltaR, m_R, v_R = _updateR_adam(X, A, lmbdaR, lambda_IJ, C, R,  m_R, v_R, itr, beta_1, beta_2, alpha=alpha_r, epsilon=epsilon)
            lambda_IJ, delta_lambda_IJ, m_IJ, v_IJ = _updateLambdaIJ_adam(X, R, C, lambda_IJ,  gemma, m_IJ, v_IJ, itr, beta_1, beta_2, alpha=alpha_lag_mult, epsilon=epsilon)

            #deltaA_lst.append(deltaA)
            #deltaR_lst.append(deltaR)
            #delta_lambda_IJ_lst.append(delta_lambda_IJ)

            a_norm_current = LA.norm(A)
            r_norm_current = LA.norm([LA.norm(r) for r in R])
            A_norm.append(a_norm_current)
            R_norm.append(r_norm_current)
            lambda_IJ_norm_lst.append(LA.norm(lambda_IJ))
        elif gd_algo == 'adam_no_gemma':
            A, deltaA, m_A, v_A = _updateA_adam_no_gemma(X, A, R, P, lmbdaA, m_A, v_A, itr, beta_1, beta_2, alpha=alpha_a,
                                                epsilon=epsilon)
            R, deltaR, m_R, v_R = _updateR_adam_no_gemma(X, A, lmbdaR, lambda_IJ, C, R, m_R, v_R, itr, beta_1, beta_2,
                                                alpha=alpha_r, epsilon=epsilon)
            lambda_IJ, delta_lambda_IJ, m_IJ, v_IJ = _updateLambdaIJ_adam_no_gemma(X, R, C, lambda_IJ, gemma, m_IJ, v_IJ,
                                                                          itr, beta_1, beta_2, alpha=alpha_lag_mult,
                                                                          epsilon=epsilon)

            # deltaA_lst.append(deltaA)
            # deltaR_lst.append(deltaR)
            # delta_lambda_IJ_lst.append(delta_lambda_IJ)

            a_norm_current = LA.norm(A)
            r_norm_current = LA.norm([LA.norm(r) for r in R])
            A_norm.append(a_norm_current)
            R_norm.append(r_norm_current)
            lambda_IJ_norm_lst.append(LA.norm(lambda_IJ))
        else:
            A, deltaA = _updateA(X, A, R, P, lmbdaA)
            R, deltaR = _updateR(X, A, lmbdaR, lambda_IJ, C, R)
            lambda_IJ, delta_lambda_IJ = _updateLambdaIJ(X, R, C, lambda_IJ, gemma)

            A_norm.append(LA.norm(A))
            R_norm.append(LA.norm([LA.norm(r) for r in R]))
        '''

        A, deltaA, m_A, v_A = _updateA_adam_no_gemma(X, A, R, P, lmbdaA, m_A, v_A, itr, beta_1, beta_2, alpha=alpha_a,
                                                     epsilon=epsilon)
        R, deltaR, m_R, v_R = _updateR_adam_no_gemma(X, A, lmbdaR, lambda_IJ, C, R, m_R, v_R, itr, beta_1, beta_2,
                                                     alpha=alpha_r, epsilon=epsilon)
        lambda_IJ, delta_lambda_IJ, m_IJ, v_IJ = _updateLambdaIJ_adam_no_gemma(X, R, C, lambda_IJ, gemma, m_IJ, v_IJ,
                                                                               itr, beta_1, beta_2,
                                                                               alpha=alpha_lag_mult,
                                                                               epsilon=epsilon)

        # deltaA_lst.append(deltaA)
        # deltaR_lst.append(deltaR)
        # delta_lambda_IJ_lst.append(delta_lambda_IJ)

        a_norm_current = LA.norm(A)
        r_norm_current = LA.norm([LA.norm(r) for r in R])
        A_norm.append(a_norm_current)
        R_norm.append(r_norm_current)
        lambda_IJ_norm_lst.append(LA.norm(lambda_IJ))



        # compute fit value
        if compute_fit:
            fit = _compute_fit(X, A, R, C, lambda_IJ, lmbdaA, lmbdaR)
            fit_lst.append(fit)
        else:
            fit = itr

        toc = time.time()
        exectimes.append(toc - tic)

        _log.info('[%3d] fit: %0.5f | delta: %7.1e | secs: %.5f' % (
            itr, fit, fitchange, exectimes[-1]
        ))
        print ('[%3d] fit: %0.5f | delta: %7.1e | secs: %.5f' % (itr, fit, fitchange, exectimes[-1]))
        #print ('[%3d] delta_A: %0.5f | delta_R: %7.1e | delta_lambda: %.5f' % (itr, deltaA, deltaR, 0)) #delta_lambda_IJ))

        #if itr > 0 and fitchange < conv and compute_fit:
        #    break

    #plt.plot(range(maxIter), deltaA_lst)
    #plt.plot(range(maxIter), deltaR_lst)
    #plt.plot(range(maxIter), delta_lambda_IJ_lst)
    #plt.title('alpha={0}, beta1={1}, beta2={2}'.format(alpha, beta_1, beta_2))
    #plt.plot(range(maxIter), fit_lst)
    x_axis = range(itr)
    to_plot_log = True
    '''
    if not to_plot_log:
        if compute_fit:
            plt.plot(x_axis, fit_lst[:itr])
        plt.plot(x_axis, A_norm[:itr])
        plt.plot(x_axis, R_norm[:itr])
        plt.plot(x_axis, lambda_IJ_norm_lst[:itr])
        plt.ylabel('norm', fontsize=18)
    else:
        if compute_fit:
            plt.plot(x_axis, np.log(fit_lst[:itr]))
        plt.plot(x_axis, np.log(A_norm[:itr]))
        plt.plot(x_axis, np.log(R_norm[:itr]))
        plt.plot(x_axis, np.log(lambda_IJ_norm_lst[:itr]))
        plt.ylabel('log(norm)', fontsize=18)

    plt.title('A-norm , R-norm, Lag multiplier norm: beta1={0}, beta2={1}, {2}, {3}, {4}, {5}, {6}'.format(beta_1, beta_2,lmbdaA, lmbdaR, alpha_a, alpha_r, alpha_lag_mult))

    plt.grid()

    if compute_fit:
        plt.legend(['reconstruction error', 'A norm', 'R norm', 'Lag. Multiplier norm'], loc='upper right')
    else:
        plt.legend(['A norm', 'R norm', 'Lag. Multiplier norm'], loc='upper right')
    #plt.legend(['A norm', 'R norm', 'Lag. Multiplier norm'], loc='upper right')
    plt.xlabel('Iteration', fontsize=18)
    '''

    #plt.show()
    return lambda_IJ, A, R, f, itr + 1, array(exectimes)

# Update LambdaIJ
def _updateLambdaIJ(X, R, C, lambda_IJ, gemma):
    old_lambda_IJ = np.array(lambda_IJ, copy=True)
    rows, column = len(X), len(X)
    for i in range(rows):
        for j in range(column):
            r = np.power(LA.norm((R[i] - R[j]), 'fro'), 2)
            delta_lambda_IJ = 1 - C[i][j] - r
            lambda_IJ[i][j] =  delta_lambda_IJ # (1 - C[i][j] - np.power(LA.norm((R[i] - R[j]), 'fro'), 2))

    delta = np.sum(np.subtract(lambda_IJ, old_lambda_IJ))
    return  lambda_IJ, delta


# ------------------ Update A ------------------------------------------------
def _updateA(X, A, R, P, lmbdaA):
    old_A = np.array(A , copy=True)
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

    # finally compute update for A
    A = solve(I + E.T, F.T).T

    delta = LA.norm(np.subtract(A, old_A))
    #print(delta)
    return A, delta

# ------------------ Update R ------------------------------------------------

def _updateR(X, A, lmbdaR, lambda_IJ, C, old_R):
    old_R = np.array(old_R, copy=True)
    _log.debug('Updating R (SVD) lambda R: %s' % str(lmbdaR))
    rank = A.shape[1]
    U, S, Vt = svd(A, full_matrices=False)
    Shat = kron(S, S)
    R = []
    for i in range(len(X)):

        #sum_over_j = None
        #for j in range(len(X)):
        #   if i != j:
        #        if sum_over_j is None:
        #            sum_over_j = lambda_IJ[i][j] * old_R[j]
        #        else:
        #            sum_over_j = sum_over_j + lambda_IJ[i][j] * old_R[j]

        #row_sum = np.sum(lambda_IJ[i, :])
        inverse_term =  Shat ** 2 + lmbdaR #+ row_sum
        Shat = (Shat / inverse_term).reshape(rank, rank)
        Rn = Shat * dot(U.T, X[i].dot(U))
        Rn = dot(Vt.T, dot(Rn, Vt)) #+ inverse_term.reshape(rank, rank) * sum_over_j
        #print(Rn)
        R.append(Rn)

    delta = LA.norm([LA.norm(np.subtract(i, j)) for i,j in zip(R, old_R)])
    return R, delta

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


def _get_adam_update(gradient, iteration, m_previous, v_previous, alpha, beta_1, beta_2, epsilon):
    #print (gradient.shape)
    #print(gradient.shape, alpha)
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

    # A = old_A - 0.01 * gtA
    term3 = np.divide(mt_hat, np.add(np.sqrt(vt_hat), epsilon))
    term4 = np.multiply(alpha, term3)
    updated_parameter = np.subtract(gradient, term4)
    return updated_parameter, mt, vt


# ------------------- Update A, R and LAMBADA without Gemma --------------------
def _updateA_adam_no_gemma(X, A, R, P, lmbdaA, m_previous, v_previous, iteration, beta_1, beta_2, alpha,
                  epsilon):
    old_A = np.array(A, copy=True)
    if m_previous is None:
        m_previous = np.zeros((A.shape[0], A.shape[1]))
    if v_previous is None:
        v_previous = np.zeros((A.shape[0], A.shape[1]))


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

    gradient = solve(I + E.T, F.T).T
    #A = solve(I + E.T, F.T).T
    A, mt, vt = _get_adam_update(gradient, iteration, m_previous, v_previous, alpha, beta_1, beta_2, epsilon)
    delta_sum = np.sum(np.subtract(A, old_A))
    #mt, vt = None, None
    return A, delta_sum, mt, vt

def _updateR_adam_no_gemma(X, A, lmbdaR, lambda_IJ, C, old_R, m_previous, v_previous,iteration, beta_1, beta_2, alpha,
                  epsilon):
    old_R = np.array(old_R, copy=True)
    _log.debug('Updating R (SVD) lambda R: %s' % str(lmbdaR))
    rank = A.shape[1]

    if m_previous is None:
        m_previous = [np.zeros((rank, rank)) for i in range(len(X))]
    if v_previous is None:
        v_previous = [np.zeros((rank, rank)) for i in range(len(X))]

    U, S, Vt = svd(A, full_matrices=False)
    Shat = kron(S, S)
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
        #for j in range(len(X)):
        #    if i != j:

        lambda_IJ_row_i = lambda_IJ[i, :]
        row_sum = np.sum(lambda_IJ_row_i)
        inverse_term =  Shat ** 2 + lmbdaR - row_sum
        first_term = (Shat / inverse_term).reshape(rank, rank)
        first_term = first_term * dot(U.T, X[i].dot(U))
        first_term = dot(Vt.T, dot(first_term, Vt))
        second_term = (sum_over_j / inverse_term).reshape(rank, rank)
        gradient = np.add(first_term,  second_term)

        gtRn, mt_n, vt_n = _get_adam_update(gradient, iteration, m_previous[i], v_previous[i], alpha, beta_1, beta_2, epsilon)

        #mt_n = np.add(np.multiply(beta_1, m_previous[i]), np.multiply(1 - beta_1, gtRn))
        #vt_n = np.add(np.multiply(beta_2, v_previous[i]), np.multiply(1-beta_2, np.power(gtRn, 2)))
        #beta_2 * v_previous[i] + (1 - beta_2) * np.power(gtRn, 2)
        #denominator = 1 - beta_1 ** (iteration+1)
        #m_hat = np.divide(mt_n, denominator)

        #denominator = 1 - beta_2 ** (iteration+1)
        #v_hat = np.divide(vt_n, denominator)


        #term1 = np.sqrt(v_hat)
        #term2 = np.add(term1, epsilon)
        #term3 = np.divide(m_hat, term2)
        #term4 = np.multiply(alpha, term3)
        #other_term = np.multiply(alpha, np.divide(m_hat, np.add(np.sqrt(v_hat), epsilon)))
        #alpha * (m_hat / (np.sqrt(v_hat) + epsilon))
        #Rn = np.subtract(old_R[i], term4)
        #Rn = old_R[i] - alpha * gtRn
        R.append(gtRn)
        m.append(mt_n)
        v.append(vt_n)

    delta_sum = np.sum([np.sum(np.subtract(i, j)) for i,j in zip(R, old_R)])
    return R, delta_sum, m, v


def _updateLambdaIJ_adam_no_gemma(X, R, C, lambda_IJ, gemma, m_previous, v_previous, iteration, beta_1, beta_2, alpha,
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

    #r_ij =
    #for i in range(rows):
    #    for j in range(column):
    #        r =
    #        #delta_lambda_IJ = (1.0/gemma) * (1 - C[i][j] - r)
    #        delta_lambda_IJ = r + C[i][j] - 1
    #        m[i][j] = beta_1 * m_previous[i][j] + (1 - beta_1) * delta_lambda_IJ
    #        v[i][j] = beta_2 * v_previous[i][j] + (1 - beta_2) * np.power(delta_lambda_IJ, 2)
    #        m_hat = (m_previous[i][j] * 1.0) / (1 - beta_1 ** (iteration+1))
    #        v_hat = (v_previous[i][j] * 1.0) / (1 - beta_2 ** (iteration+1))

    #        lambda_IJ[i][j] = lambda_IJ[i][j] - alpha * (m_hat / (np.sqrt(v_hat) + epsilon))

    delta = LA.norm(np.subtract(lambda_IJ, old_lambda_IJ))
    return lambda_IJ, delta, mt, vt






# ------------------- Update A - adam --------------------------------
def _updateA_adam(X, A, R, P, lmbdaA, m_previous, v_previous, iteration, beta_1, beta_2, alpha,
                  epsilon):
    old_A = np.array(A, copy=True)
    if m_previous is None:
        m_previous = np.zeros((A.shape[0], A.shape[1]))
    if v_previous is None:
        v_previous = np.zeros((A.shape[0], A.shape[1]))


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

    gradient = solve(I + E.T, F.T).T
    #A = solve(I + E.T, F.T).T
    A, mt, vt = _get_adam_update(gradient, iteration, m_previous, v_previous, alpha, beta_1, beta_2, epsilon)
    delta_sum = np.sum(np.subtract(A, old_A))
    #mt, vt = None, None
    return A, delta_sum, mt, vt

def _updateR_adam(X, A, lmbdaR, lambda_IJ, C, old_R, m_previous, v_previous,iteration, beta_1, beta_2, alpha,
                  epsilon):
    old_R = np.array(old_R, copy=True)
    _log.debug('Updating R (SVD) lambda R: %s' % str(lmbdaR))
    rank = A.shape[1]

    if m_previous is None:
        m_previous = [np.zeros((rank, rank)) for i in range(len(X))]
    if v_previous is None:
        v_previous = [np.zeros((rank, rank)) for i in range(len(X))]

    U, S, Vt = svd(A, full_matrices=False)
    Shat = kron(S, S)
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

        #for j in range(len(X)):
        #    if i != j:

        lambda_IJ_row_i = lambda_IJ[i, :]
        row_sum = np.sum(lambda_IJ_row_i)
        inverse_term =  Shat ** 2 + lmbdaR + row_sum
        first_term = (Shat / inverse_term).reshape(rank, rank)
        first_term = first_term * dot(U.T, X[i].dot(U))
        first_term = dot(Vt.T, dot(first_term, Vt))
        second_term = (sum_over_j / inverse_term).reshape(rank, rank)
        gradient = np.add(first_term,  second_term)

        gtRn, mt_n, vt_n = _get_adam_update(gradient, iteration, m_previous[i], v_previous[i], alpha, beta_1, beta_2, epsilon)

        #mt_n = np.add(np.multiply(beta_1, m_previous[i]), np.multiply(1 - beta_1, gtRn))
        #vt_n = np.add(np.multiply(beta_2, v_previous[i]), np.multiply(1-beta_2, np.power(gtRn, 2)))
        #beta_2 * v_previous[i] + (1 - beta_2) * np.power(gtRn, 2)
        #denominator = 1 - beta_1 ** (iteration+1)
        #m_hat = np.divide(mt_n, denominator)

        #denominator = 1 - beta_2 ** (iteration+1)
        #v_hat = np.divide(vt_n, denominator)


        #term1 = np.sqrt(v_hat)
        #term2 = np.add(term1, epsilon)
        #term3 = np.divide(m_hat, term2)
        #term4 = np.multiply(alpha, term3)
        #other_term = np.multiply(alpha, np.divide(m_hat, np.add(np.sqrt(v_hat), epsilon)))
        #alpha * (m_hat / (np.sqrt(v_hat) + epsilon))
        #Rn = np.subtract(old_R[i], term4)
        #Rn = old_R[i] - alpha * gtRn
        R.append(gtRn)
        m.append(mt_n)
        v.append(vt_n)

    delta_sum = np.sum([np.sum(np.subtract(i, j)) for i,j in zip(R, old_R)])
    return R, delta_sum, m, v


def _updateLambdaIJ_adam(X, R, C, lambda_IJ, gemma, m_previous, v_previous, iteration, beta_1, beta_2, alpha,
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

    gradient = np.multiply(1/gemma, np.subtract(1, np.add(r_ij, C)))
    lambda_IJ, mt, vt = _get_adam_update(gradient, iteration, m_previous, v_previous, alpha, beta_1, beta_2, epsilon)

    #r_ij =
    #for i in range(rows):
    #    for j in range(column):
    #        r =
    #        #delta_lambda_IJ = (1.0/gemma) * (1 - C[i][j] - r)
    #        delta_lambda_IJ = r + C[i][j] - 1
    #        m[i][j] = beta_1 * m_previous[i][j] + (1 - beta_1) * delta_lambda_IJ
    #        v[i][j] = beta_2 * v_previous[i][j] + (1 - beta_2) * np.power(delta_lambda_IJ, 2)
    #        m_hat = (m_previous[i][j] * 1.0) / (1 - beta_1 ** (iteration+1))
    #        v_hat = (v_previous[i][j] * 1.0) / (1 - beta_2 ** (iteration+1))

    #        lambda_IJ[i][j] = lambda_IJ[i][j] - alpha * (m_hat / (np.sqrt(v_hat) + epsilon))

    delta = LA.norm(np.subtract(lambda_IJ, old_lambda_IJ))
    return lambda_IJ, delta, mt, vt


'''
def _compute_fit(X, A, R):
    """Compute fit for full slices"""
    f = 0
    # precompute norms of X
    normX = [sum(M.data ** 2) for M in X]
    sumNorm = sum(normX)

    for i in range(len(X)):
        ARAt = dot(A, dot(R[i], A.T))
        f += norm(X[i] - ARAt) ** 2

    return 1 - f / sumNorm



def _compute_data_fit(X, A, R):
    f = 0
    divider = np.math.floor(len(A) / 2)
    steps = 10
    for i in range(len(X)):
        for step in range(steps):
            width = np.math.floor(len(A) / steps)
            start = step * width
            end = (step+1) * width
            if end > len(A):
                end = len(A)
            Alower = A[start:end]
            AlowerR = dot(Alower, R[i])
            del Alower
            AlowerRAt = dot(AlowerR, A.T)
            f  = f + norm(X[i][start:end] - AlowerRAt)
            del AlowerRAt

            Aupper = A[start:end]
            AupperR = dot(Aupper, R[i])
            del Aupper
            AupperRAt = dot(AupperR, A.T)
            f = f + norm(X[i][start:end] - AupperRAt)
            del AupperRAt

    f = f/ len(X)
    return f


def _compute_variable_fit(a_old, a, r_old, r, method='norm'):
    if method is 'norm':
        from numpy import linalg as LA
        a_diff = LA.norm(a_old - a)
        slice_diff = 0
        for i in range(len(r)):
            slice_old = r_old[i]
            slice_new = r[i]
            slice_diff = slice_diff + LA.norm(slice_old - slice_new)
        slice_diff = slice_diff/len(r)
        return (a_diff + slice_diff)/2
    elif method is 'max':
        max = []
        from numpy import linalg as LA
        max.append(np.amax(a_old - a))
        for i in range(len(r)):
            slice_old = r_old[i]
            slice_new = r[i]
            max.append(np.amax(slice_old - slice_new))
        return np.amax(max)


'''
def _compute_fit(X, A, R, C, lambdaIJ, lmbdaA, lmbdaR):
    #Compute fit for full slices
    f = 0
    # precompute norms of X
    normX = [sum(M.data ** 2) for M in X]
    sumNorm = sum(normX)

    for i in range(len(X)):
        ARAt = dot(A, dot(R[i], A.T))
        f += norm(X[i] - ARAt) ** 2 #+ np.multiply(0.5 * lmbdaR, np.power(LA.norm(R[i], 'fro'), 2))
        #for j in range(len(X)):
        #    r = np.power(LA.norm(np.subtract(R[i], R[j]), 'fro'), 2) + C[i][j] - 1
        #    f += lambdaIJ[i][j] * r

    #f = f + np.multiply(0.5 * lmbdaA, np.power(LA.norm(A, 'fro'), 2))

    return f /sumNorm #1 - f / sumNorm
