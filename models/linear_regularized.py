# coding: utf-8
# quadratic_regularized.py - python script to compute the quadratic + regularized knowledge enriched tensor factorization
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

from numpy import dot, zeros, array, eye, kron, prod
from numpy.random import rand
from scipy.linalg import svd, solve, inv, norm
from scipy.sparse import issparse
from random import randint

from models.distance_local import *
import numpy as np

__version__ = "0.1"
__all__ = ['linear_regularized']

_DEF_MAXITER = 500
_DEF_INIT = 'nvecs'
_DEF_CONV = 1e-6
_DEF_LMBDA = 0
_DEF_ATTR = []
_DEF_NO_FIT = 1e9
_DEF_FIT_METHOD = None

_algo  = 'quadratic_regularized'
_fold_number = 0
_dataset_name = 'dataset'
_rho_inv = 0

_log = logging.getLogger('linear regularized')

def _set_random_value_to_one(tensor, count):
    for slice in range(len(tensor)):
        rel_slice = tensor[slice]
        for i in range(count):
            random_row = randint(1, np.shape(rel_slice)[0]-1)
            random_col = randint(1, np.shape(rel_slice)[0]-1)
            tensor[slice][random_row, random_col] = 1

    return tensor

def linear_regularized(X, rank, iteration, **kwargs):
    # ------------ init options ----------------------------------------------
    ainit = kwargs.pop('init', _DEF_INIT)
    _DEF_MAXITER = iteration
    maxIter = kwargs.pop('maxIter', _DEF_MAXITER)
    conv = kwargs.pop('conv', _DEF_CONV)
    lmbdaA = kwargs.pop('lambda_A', _DEF_LMBDA)
    lmbdaR = kwargs.pop('lambda_R', _DEF_LMBDA)
    lmbdaV = kwargs.pop('lambda_V', _DEF_LMBDA)
    lmbdaRelSimilarity= kwargs.pop('lmbdaRelSimilarity', _DEF_LMBDA)
    lambda_e = kwargs.pop('lambda_e', _DEF_LMBDA)

    rho_inv = kwargs.pop('rho_inv', _DEF_LMBDA)
    compute_fit = kwargs.pop('compute_fit', _DEF_FIT_METHOD)
    P = kwargs.pop('attr', _DEF_ATTR)
    dtype = kwargs.pop('dtype', np.float)



    algo = kwargs.pop('algo', _algo)
    fold_number = kwargs.pop('fold', _fold_number)
    dataset_name = kwargs.pop('dataset', _dataset_name)
    image_path = kwargs.pop('image_path', '/home/a/git/person_project_new/tensor/models/convergence_images/')
    image_name = kwargs.pop('image_name', algo + "_fold_" + str(fold_number) + "_" + str(maxIter) + "_variable_" + dataset_name)
    is_compute_fit_relative = kwargs.pop('is_compute_fit_relative', False)
    method=kwargs.pop('method', 'max')

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
        A1 = array(rand(n, rank), dtype=dtype)
        A2 = array(rand(n, rank), dtype=dtype)
    else:
        raise ValueError('Unknown init option ("%s")' % ainit)

    d = distance_local()
    co_var = d.find_slice_co_var(X)
    #co_var = np.identity(rank)

    # Ignore - to overcome effect of more number of zeros, set random value to 1.0
    #X = _set_random_value_to_one(X, 10000)

    # ------- initialize R and Z ---------------------------------------------
    R = _updateR(X, A1, A2, co_var, lmbdaR, lmbdaRelSimilarity ,rho_inv)
    Z = _updateZ(A1, A2, P, lmbdaV)


    a1_old = A1
    a2_old = A2
    r_old = R
    #  ------ compute factorization ------------------------------------------
    fitchange = fitold = f = 0
    fitchange_variable = fitold_variable = 0

    fit = 0#_compute_fit(X, A1, A2, R)
    fit_variable = 0



    fits = []
    fits_variables = []

    fits.append(fit)
    #fits_variables.append(fit_variable)

    exectimes = []

    for itr in range(maxIter):
        tic = time.time()
        fitold = fit
        fitold_variable = fit_variable

        A1 = _updateA1(X, A2, R, P, Z, lmbdaA, lambda_e, rho_inv)
        A2 = _updateA2(X, A1, R, P, Z, lmbdaA, lambda_e, rho_inv)
        R = _updateR(X, A1, A2, co_var, lmbdaR, lmbdaRelSimilarity, rho_inv)
        Z = _updateZ(A1, A2, P, lmbdaV)

        # compute fit value
        if compute_fit:
            fit_variable = _compute_variable_fit(a1_old, A1, a2_old, A2, r_old, R, method=method, is_compute_fit_relative=is_compute_fit_relative)
            fit = -1 # __compute_data_fit(X, A1, A2, R)
        else:
            fit = itr

        fitchange = abs(fitold - fit)
        fitchange_variable = abs(fitold_variable - fit_variable)
        if fitold_variable < fit_variable:
            rho_inv = rho_inv / 2.0

        fits.append(fit)
        fits_variables.append(fitchange_variable)

        toc = time.time()
        exectimes.append(toc - tic)

        _log.info('[%3d] fit: %0.5f | delta: %7.1e | secs: %.5f' % (
            itr, fit, fitchange, exectimes[-1]
        ))

        print('[%3d] fit: %0.5f | delta: %7.1e | secs: %.5f' % (itr, fit_variable, fitchange_variable, exectimes[-1]))

        a1_old = A1
        a2_old = A2
        r_old = R



        if itr > 0 and fitchange_variable < conv and compute_fit:
            break


    #A = (A1+A2)/2.0

#    errors = fits + [fits[-1]] * (100 -len(fits))
#    import matplotlib.pyplot as plt
#    if image_path is '':
#        plt.clf()
#        plt.plot(range(100), errors[:100], marker='o')
#        plt.ylabel('Error')
#        plt.xlabel('Iteration')
#        #plt.show()
#        fig = plt.gcf()
#        image_name = algo + "_fold_" + str(fold_number) + "_" + str(maxIter) + "_" + dataset_name
#        #fig.savefig('/home/a/git/person_project_new/tensor/models/convergence_images/' + image_name + ".png")

#        plt.clf()
#        errors_variable = fits_variables + [fits_variables[-1]] * (100 - len(fits_variables))
#        plt.plot(range(100), errors_variable, marker='o',  markersize=3, color="red")
#        plt.ylabel('Error-Variable')
#        plt.xlabel('Iteration')
#        #plt.show()
#        fig = plt.gcf()
#        image_name =
#        fig.savefig( + image_name + ".png")
#        plt.clf()
#    else:

    save_plot(image_path, image_name, range(100), fits_variables, 'o')

    return [], A1, A2, R, fits_variables, itr + 1, array(exectimes)

def save_plot(path, file_name, x_axis, y_axis, marker, show=False):
    if path is None:
        return
    '''
    if len(y_axis) < len(x_axis):
        y_axis = y_axis + [y_axis[-1]] * (len(x_axis)-len(y_axis))
    plt.clf()
    plt.plot(x_axis, y_axis, marker=marker)
    plt.ylabel('Error Value')
    plt.xlabel('Iteration')
    if show:
        plt.show()
    fig = plt.gcf()
    fig.savefig(path + "/" + file_name)
    plt.clf()
    '''



# ------------------ Update A1 ------------------------------------------------
def _updateA1(X, A2, R, P, Z, lmbdaA, lambdaE, rho_inv):
    """Update step for A"""
    _log.debug('Updating A')
    n, rank = A2.shape
    F = zeros((n, rank), dtype=A2.dtype)
    E = zeros((rank, rank), dtype=A2.dtype)

    A2tA2 = dot(A2.T, A2)

    for i in range(len(X)):
        F += X[i].dot(dot(A2, R[i].T))#X[i].T.dot(dot(A, R[i]))
        E += dot(R[i], dot(A2tA2, R[i].T))# + dot(R[i].T, dot(AtA, R[i]))

    # regularization
    I = (lmbdaA + lambdaE + rho_inv) * eye(rank, dtype=A2.dtype)
    lambdaE_A2 = lambdaE * A2

    # attributes
    for i in range(len(Z)):
        F += P[i].dot(Z[i].T)
        E += dot(Z[i], Z[i].T)
    # finally compute update for A1
    A1 = solve(E.T + I, F.T + lambdaE_A2.T).T
    return A1


# ------------------ Update A2 ------------------------------------------------
def _updateA2(X, A1, R, P, Z, lambdaA, lambdaE, rho_inv):
    """Update step for A"""
    _log.debug('Updating A')
    n, rank = A1.shape
    F = zeros((n, rank), dtype=A1.dtype)
    E = zeros((rank, rank), dtype=A1.dtype)

    A1tA1 = dot(A1.T, A1)

    for i in range(len(X)):
        F += X[i].T.dot(dot(A1, R[i]))  # X[i].T.dot(dot(A, R[i]))
        E += dot(R[i].T, dot(A1tA1, R[i]))  # + dot(R[i].T, dot(AtA, R[i]))

    # regularization
    I = (lambdaA + lambdaE + rho_inv) * eye(rank, dtype=A1.dtype)
    lambdaE_A1 = lambdaE * A1

    # attributes
    for i in range(len(Z)):
        F += P[i].dot(Z[i].T)
        E += dot(Z[i], Z[i].T)

    # finally compute update for A2
    A2 = solve(E.T + I, lambdaE_A1.T + F.T).T
    return A2

'''
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
    for i in range(len(Z)):
        F += P[i].dot(Z[i].T)
        E += dot(Z[i], Z[i].T)

    # finally compute update for A
    A = solve(I + E.T, F.T).T
    #A = dot(F, inv(I + E))
    #_log.debug('Updated A lambda_A:%f, dtype:%s' % (lmbdaA, A.dtype))
    return A


'''
# ------------------ Update R ------------------------------------------------

def _updateR(X, A1, A2, co_var, lmbdaR, lmbdaRelSimilarity, rho_inv):
    _log.debug('Updating R (SVD) lambda R: %s' % str(lmbdaR))
    rank1 = A1.shape[1]
    rank2 = A2.shape[1]
    U1, S1, V1t = svd(A1, full_matrices=False)
    U2, S2, V2t = svd(A2, full_matrices=False)
    Shat = kron(S2, S1)
    R = []

    for i in range(len(X)):
        Shat = (Shat / (Shat ** 2 + lmbdaR + rho_inv + lmbdaRelSimilarity * (np.sum(co_var[i])))).reshape(rank2, rank1)
        if len(R) == 0:
            Rn = Shat * dot(U1.T, X[i].dot(U2))
        #else:
            #rel_sum_similarity = []
            #for j in range(len(R)):
            #    if j == i:
            #        continue
            #    weight = co_var[i][j]
            #    if len(rel_sum_similarity) == 0:
            #        rel_sum_similarity = lmbdaRelSimilarity * weight * R[j]
            #    else:
            #        rel_sum_similarity = rel_sum_similarity + lmbdaRelSimilarity * weight * R[j]
            #
        Rn = Shat * dot(U1.T, X[i].dot(U2))# + rel_sum_similarity
        Rn = dot(V1t.T, dot(Rn, V2t))
        R.append(Rn)
    return R

# ------------------ Update Z ------------------------------------------------
def _updateZ(A1, A2, P, lmbdaZ):
    pass
    #'''
    Z = []
    if len(P) == 0:
        return Z
    #_log.debug('Updating Z (Norm EQ, %d)' % len(P))
    pinvAt = inv(dot(A2.T, A1) + lmbdaZ * eye(A1.shape[1], dtype=A1.dtype))
    pinvAt = dot(pinvAt, A2.T).T
    for i in range(len(P)):
        if issparse(P[i]):
            Zn = P[i].tocoo().T.tocsr().dot(pinvAt).T
        else:
            Zn = dot(pinvAt.T, P[i])
        Z.append(Zn)
    return Z
    #'''

def get_relative_difference(old_value, new_value):
    old_value[old_value == 0] = 1
    new_value[new_value == 0] = 1

    numerator = abs(old_value - new_value)
    denominator = abs(old_value) + abs(new_value)
    denominator_average = np.divide(denominator, 2.0)
    final_value = np.divide(numerator, denominator_average)
    return np.argmax(final_value)


def _compute_variable_fit(a1_old, a1, a2_old, a2, r_old, r, method='max', is_compute_fit_relative=False):
    if method is 'norm':
        from numpy import linalg as LA
        a1_diff = LA.norm(a1_old - a1)
        a2_diff = LA.norm(a2_old - a2)

        slice_diff = 0
        for i in range(len(r)):
            slice_old = r_old[i]
            slice_new = r[i]
            slice_diff = slice_diff + abs(LA.norm(slice_old - slice_new))

        return (a1_diff + a2_diff + slice_diff) / (1 + 1 + len(r))

    if method is 'norm' and is_compute_fit_relative:
        from numpy import linalg as LA
        #a1_diff = LA.norm(a1_old - a1)
        a1_old[a1_old == 0.0] = 1.0
        a1_diff = LA.norm((a1_old - a1)/a1_old)

        a2_old[a2_old == 0.0] = 1.0
        a2_diff = LA.norm((a2_old - a2)/a2_old)

        slice_diff = 0
        for i in range(len(r)):
            slice_old = r_old[i]
            slice_old[slice_old == 0] = 1.0
            slice_new = r[i]
            slice_diff = slice_diff + abs(LA.norm((slice_old - slice_new)/slice_old))

        return (a1_diff + a2_diff + slice_diff) / (1 + 1 + len(r))


    elif method is 'max' and is_compute_fit_relative:
        max = []

        a1_difference = get_relative_difference(a1_old, a1)
        max.append(a1_difference)
        a2_difference = get_relative_difference(a2_old, a2)
        max.append(a2_difference)

        for i in range(len(r)):
            slice_old = r_old[i]
            slice_new = r[i]
            slice_difference = get_relative_difference(slice_old, slice_new)
            max.append(slice_difference)

        return np.amax(max)

    elif method is 'max':
        max = []
        max.append(np.amax(abs(a1_old - a1)))
        max.append(np.amax(abs(a2_old - a2)))

        for i in range(len(r)):
            slice_old = r_old[i]
            slice_new = r[i]
            max.append(np.amax(abs((slice_old - slice_new))))

        return np.amax(max)

def __compute_data_fit(X, A1, A2, R):
    f = 0
    for i in range(len(X)):
        ar = dot(A1, R[i])
        f = f + norm(abs(X[i] - dot(ar, A2.T)))

    return f / (len(X))