#!/usr/bin/env python
#import sys
#import logging
#logging.basicConfig(level=logging.INFO)
#_log = logging.getLogger('Example Kinships')
import argparse
import os
import sys

from tabulate import tabulate

from models.distance_local import distance_local

sys.path.append('/home/pankur1/ferraro_user/lint')

from data_processing.dataset_parameters import ds_parameters

import argparse

from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

'''
sys.path.append(os.getcwd())
exp_dir = os.getcwd()
print (exp_dir)
exp_dir = exp_dir[:os.getcwd().rfind('/')+1]
print (exp_dir)
model_package = exp_dir + "models/"
sys.path.append(exp_dir + "models/nn_rescal")
sys.path.append(exp_dir + "models/logger")
print (sys.path)
'''

from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from math import *
from numpy import dot
from numpy.linalg import norm
from scipy.sparse import lil_matrix
from sklearn.metrics import precision_recall_curve, auc, accuracy_score


import copy
import numpy as np

np.random.seed(123)

import logger.Logger as log

from data_processing.tensor_factory import Tensor_Factory
from data_processing.dataset_parameters import *

from models.linear_constraint import *
from models.linear_regularized import *
from models.quadratic_constraint import *
from models.quadratic_regularized import *

from models import variables

metric = ['last-iter', 'auc', 'exectime', 'f1_micro', 'f1_macro']

GROUND_TRUTH = None

compute_fit=False

def get_prediction(T, target_idx_c, f=1):

    algo = variables.arguments['algo']
    dataset_name = variables.arguments['dataset_name']

    if algo == "quadratic_regularized":
        parameter_dict = ds_parameters[algo][dataset_name]
        lambda_A = parameter_dict['lambda_A']
        lambda_R = parameter_dict['lambda_R']
        lmbdaRelSimilarity = parameter_dict['lambda_sim']
        lambda_IJ, A, R, _, last_itr, exectimes = quadratic_regularized(
            T, len(T), maxIter, init='random', conv=conv,
            lambda_A=lambda_A, lambda_R=lambda_R, compute_fit=compute_fit, lmbdaRelSimilarity=lmbdaRelSimilarity)

        P = []
        P_multiclass = []
        for i in range(len(target_idx_c)):
            for t in target_idx_c[i]:
                row, col, slice = t[0], t[1], t[2]
                pipe = []
                a = A[row]
                aT = A[col].T

                for j in range(len(T)):
                    r = R[j]
                    dot_prod = dot(a, dot(r, aT))
                    pipe.append(dot_prod)
                nrm = norm(np.asarray(pipe))
                v = pipe[slice] / nrm
                v = np.round_(v)
                P.append(abs(v)) if nrm != 0 else P.append(0)
                P_multiclass.append(slice) if nrm != 0 and abs(
                    np.round_(pipe[slice] / nrm)) > 0.5 else P_multiclass.append(-1)

        return P, np.average(exectimes), last_itr, P_multiclass

    elif algo == "linear_regularized":
        parameter_dict = ds_parameters[algo][dataset_name]

        lambda_A = parameter_dict['lambda_A']
        lambda_R = parameter_dict['lambda_R']
        lambda_E = parameter_dict['lambda_E']
        rho_inv = parameter_dict['rho_inv']
        lambda_sim = parameter_dict['lambda_sim']

        C, A1, A2, R, _, last_itr, exectimes = linear_regularized(
            T, len(T), maxIter, init='random', conv=conv,
            lambda_A=lambda_A, lambda_R=lambda_R, lambda_e=lambda_E, rho_inv=rho_inv, compute_fit=compute_fit,
            algo='linear_regularized',
            dataset=dataset_name, fold=f, lmbdaRelSimilarity=lambda_sim)

        P = []
        P_multiclass = []

        for i in range(len(target_idx_c)):
            for t in target_idx_c[i]:
                row, col, slice = t[0], t[1], t[2]
                pipe = []
                a = A1[row]
                aT = A2[col].T

                for j in range(len(T)):
                    r = R[j]
                    dot_prod = dot(a, dot(r, aT))
                    pipe.append(dot_prod)
                nrm = norm(np.asarray(pipe))
                v = pipe[slice] / nrm
                v = np.round_(v)
                P.append(abs(v)) if nrm != 0 else P.append(0)
                P_multiclass.append(slice) if nrm != 0 and abs(np.round_(pipe[slice] / nrm)) > 0.5 else P_multiclass.append(-1)

        return P, np.average(exectimes), last_itr, P_multiclass

    elif algo == "quadratic_constraint":
        parameter_dict = ds_parameters[algo][dataset_name]

        print('parameter_dict =', parameter_dict)

        lambda_A = parameter_dict['lambda_A']
        lambda_R = parameter_dict['lambda_R']
        alpha_r = parameter_dict['alpha_r']
        alpha_a = parameter_dict['alpha_a']
        alpha_lag_mult = parameter_dict['alpha_lag_mult']
        lambda_IJ = np.random.rand(len(T), len(T))
        lambda_IJ, A, R, _, last_itr, exectimes = quadratic_constraint(
            T, len(T), maxIter, init='random', conv=conv,
            lambda_A=lambda_A, lambda_R=lambda_R, compute_fit=compute_fit,
            algo='quadratic_constraint', alpha_r=alpha_r, alpha_a=alpha_a, alpha_lag_mult=alpha_lag_mult,
            dataset=dataset_name, fold=f, lambda_IJ=lambda_IJ)

        P = []
        P_multiclass = []
        for i in range(len(target_idx_c)):
            for t in target_idx_c[i]:
                row, col, slice = t[0], t[1], t[2]
                pipe = []
                a = A[row]
                aT = A[col].T

                for j in range(len(T)):
                    r = R[j]
                    dot_prod = dot(a, dot(r, aT))
                    pipe.append(dot_prod)
                nrm = norm(np.asarray(pipe))
                v = pipe[slice] / nrm
                v = np.round_(v)
                P.append(abs(v)) if nrm != 0 else P.append(0)
                P_multiclass.append(slice) if nrm!=0 and abs(np.round_(pipe[slice] / nrm)) > 0.5 else P_multiclass.append(-1)

        return P, np.average(exectimes), last_itr, P_multiclass

    elif algo == "linear_constraint":
        parameter_dict = ds_parameters[algo][dataset_name]

        lambda_A = parameter_dict['lambda_A']
        lambda_E = parameter_dict['lambda_E']
        lambda_R = parameter_dict['lambda_R']

        alpha_r = parameter_dict['alpha_r']
        alpha_a = parameter_dict['alpha_a']
        alpha_lag_mult = parameter_dict['alpha_lag_mult']
        lambda_IJ = np.random.rand(len(T), len(T))

        lambda_IJ, A1, A2, R, _, last_itr, exectimes = linear_constraint(
            T, len(T), maxIter, init='random', conv=conv,
            lambda_A1=lambda_A, lambda_E=lambda_E, lambda_R=lambda_R, compute_fit=compute_fit,
            alpha_r=alpha_r, alpha_a=alpha_a, alpha_lag_mult=alpha_lag_mult, lambda_IJ=lambda_IJ)

        P = []
        P_multiclass = []

        for i in range(len(target_idx_c)):
            for t in target_idx_c[i]:
                row, col, slice = t[0], t[1], t[2]
                pipe = []
                a = A1[row]
                aT = A2[col].T

                for j in range(len(T)):
                    r = R[j]
                    dot_prod = dot(a, dot(r, aT))
                    pipe.append(dot_prod)
                nrm = norm(np.asarray(pipe))
                v = pipe[slice] / nrm
                v = np.round_(v)
                P.append(abs(v)) if nrm != 0 else P.append(0)
                P_multiclass.append(slice) if nrm!=0 and abs(np.round_(pipe[slice] / nrm)) > 0.5 else P_multiclass.append(-1)

        return P, np.average(exectimes), last_itr, P_multiclass

    else:
        pass

def get_confusion_matrix(ground_truth, P):
    # Confusion matrix
    count_zeros = ground_truth.count(0)
    count_ones = ground_truth.count(1)

    pred_zeros = P.count(0)
    pred_ones = P.count(1)

    # groundTruth_Prediction
    z_z = 0
    zero_one = 0
    one_one = 0
    one_zero = 0

    for g in range(len(ground_truth)):
        if ground_truth[g] == 0 and P[g] == 0:
            z_z += 1
        if ground_truth[g] == 0 and P[g] == 1:
            zero_one += 1

        if ground_truth[g] == 1 and P[g] == 1:
            one_one += 1
        if ground_truth[g] == 1 and P[g] == 0:
            one_zero += 1
    try:
      # to catch division by zero
        confusion_matrix = [[z_z / (z_z + zero_one), zero_one / (z_z + zero_one)],
                        [one_zero / (one_zero + one_one), one_one / (one_zero + one_one)]]
    except:
        confusion_matrix = [[0,0], [0,0]]

    return confusion_matrix

def get_average_conf_matrix(confusion_matrix):
    return (confusion_matrix[0][0] + confusion_matrix[1][1]) / 2.0

def get_conf_g_mean(confusion_matrix):
    one_one = confusion_matrix[1][1]
    one_zero = confusion_matrix[1][0]
    zero_zero = confusion_matrix[0][0]
    zero_one = confusion_matrix[0][1]

    a_plus = (one_one+0.0001) / (one_one + one_zero)
    a_minus = (zero_zero+0.0001) / (zero_zero + zero_one)
    g_mean = a_plus * a_minus
    return sqrt(g_mean)

def get_area_under_curve(ground_truth, P):
    prec, recall, _ = precision_recall_curve(ground_truth, P)

    '''pr = auc(recall, prec)
    try:
        pr = auc(recall, prec)
    except:
        pr = -1.0'''

    return auc(recall, prec)

def innerfold(T, mask_idx, target_idx, n, k, sz, f, r=0, iter=500):
    Tc = [Ti.copy() for Ti in T]

    print(type(mask_idx), type(target_idx), type(T))
    for i in range(len(mask_idx)):
       for t in mask_idx[i]:
           Tc[t[2]][t[0], t[1]] = 0

    print(type(Tc), type(target_idx), type(f))

    P, avg_exectime, last_itr, P_multiclass = get_prediction(Tc, target_idx, f)

    ground_truth = []
    ground_truth_multiclass = []

    for i in range(len(target_idx)):
        for t in target_idx[i]:
            truth = GROUND_TRUTH[t[2]][t[0], t[1]]
            ground_truth.append(truth)
            ground_truth_multiclass.append(t[2]) if truth ==1 else ground_truth_multiclass.append(-1)

    pr = get_area_under_curve(ground_truth, P)

    f1_macro = f1_score(ground_truth_multiclass, P_multiclass, average='macro')
    f1_micro = f1_score(ground_truth_multiclass, P_multiclass, average='micro')
    #f1_weighted = f1_score(ground_truth_multiclass, P_multiclass, average='weighted')
    results = (pr, avg_exectime, last_itr, f1_micro, f1_macro)
    
    #print(classification_report(ground_truth_multiclass, P_multiclass))
    #print (results)

    return results



def convert2dict(result):

    dict = {
        'auc' : result[0]*100,
        'exectime' : result[1],
        'last-iter' : result[2],
        'f1_micro' : result[3] * 100,
        'f1_macro' : result[4] * 100,
    }

    return dict

def dict2tuple(record):
    return ["{0:.2f}".format(record[m]) for m in metric]

def process_folds_result(dict_results_test):
    final_result = {}
    folds = len(list(dict_results_test.keys()))
    for measure in metric:
        total = 0.0
        for fold in dict_results_test:
            total += dict_results_test[fold][measure]
        final_result[measure] = total / folds

    return final_result



def get_fold_values(p_value_dict, approach_name, metric_name):
    folds_value = []
    for f in p_value_dict[approach_name].keys():
        folds_value.append(p_value_dict[approach_name][f][metric_name])

    return folds_value

def get_data_tensor_for_fold(T, tens_factory, f, FOLDS, k, positive, negative):

    dataset = variables.arguments['dataset_name']

    idx_test = tens_factory.get_pos_neg(T, T[0].shape[0], T[0].shape[1], k, positive, negative,
                                        dataset=dataset, fold_number=f)

    return idx_test

def train_test_tensor_model(args=None):
    global GROUND_TRUTH

    # set variables
    variables.is_similarity_symmetric = True
    algo = variables.arguments['algo']
    FOLDS = variables.arguments['FOLDS']
    dataset_name = variables.arguments['dataset_name']

    positive = sample_size[dataset_name]["positive"]
    negative = sample_size[dataset_name]["negative"]
    parameters_dict = ds_parameters[algo][dataset_name]

    print(parameters_dict)

    # Get data Tensor
    tens_factory = Tensor_Factory()
    is_mat = tens_factory.is_mat(dataset_name)
    n, k, SZ, T = tens_factory.get_data_tensor(dataset_name, is_mat=is_mat)
    print('Datasize: {} x {} x {} | No. of classes: {}'.format(T[0].shape[0], T[0].shape[1], len(T), k))

    dict_results_test = {}
    GROUND_TRUTH = copy.copy(T)

    # Run it across 5 folds

    for f in range(FOLDS):
        print("Creating Testing data")

        idx_test = get_data_tensor_for_fold(T, tens_factory, f, FOLDS, k, positive, negative)

        print("Training + Testing")

        results = innerfold(T, idx_test, idx_test, n, k, SZ, f, 0, 0)
        
        dict_results_test[f] = convert2dict(results)

    return dict_results_test

def print_table(results):
    preformance = process_folds_result(results)

    result_table = []

    for l in range(5):
        values = ['fold-{}'.format(l)] + ["{0:.2f}".format(results[l][m]) for m in metric]
        result_table.append(values)

    avg_value = ['average'] + ["{0:.2f}".format(preformance[m]) for m in metric]
    result_table.append(avg_value)

    print(tabulate(result_table, headers=metric))

def experiment():
    parser = argparse.ArgumentParser(description='command line agruments for Tensor based model to run.')

    parser.add_argument('-o', '--algo', required=True, default='qconst')
    parser.add_argument('-s', '--dataset_name', required=True, default='kinship')
    '''
    parser.add_argument('-a', '--lambda_A', required=True, default=0)
    parser.add_argument('-b', '--lambda_A1', required=False, default=0)
    parser.add_argument('-c', '--lambda_A2', required=False, default=0)

    parser.add_argument('-t', '--lambda_sim', required=False, default=0)
    parser.add_argument('-r', '--lambda_R', required=True, default=0)
    parser.add_argument('-e', '--lambda_E', required=False, default=0)

    parser.add_argument('-d', '--alpha_a', required=False, default=0)
    parser.add_argument('-p', '--alpha_r', required=False, default=0)
    parser.add_argument('-i', '--alpha_lag_mult', required=False, default=0)

    parser.add_argument('-j', '--rho_inv', required=False, default=0)
    parser.add_argument('-m', '--gemma', required=False, default=0)
    parser.add_argument('-l', '--threshold', required=False, default=0)
    '''
    parser.add_argument('-n', '--distance', required=True, default='transitivity')
    parser.add_argument('-f', '--FOLDS', type=int, default=1)

    args = parser.parse_args()

    variables.arguments = vars(args)
    print(args)

    results = train_test_tensor_model(args)

    print_table(results)

if __name__ == '__main__':
        experiment()