import numpy as np

from . import variables

import numpy

class distance_local:
    def __init__(self):
        pass

    def get_subjects(self, slice):
        i_slice = np.nonzero(slice)
        i_slice = list(map(lambda x, y: (x, y), i_slice[0], i_slice[1]))
        i_slice_subject = list(map(lambda x: (x[0]), i_slice))
        return i_slice_subject

    def get_objects(self, slice):
        i_slice = np.nonzero(slice)
        i_slice = list(map(lambda x, y: (x, y), i_slice[0], i_slice[1]))
        i_slice_object = list(map(lambda x: (x[1]), i_slice))
        return i_slice_object

    def transitivity(self, X):
        distance_matrix = numpy.zeros((len(X), len(X)))

        for i in range(len(X)):
            i_object = set(self.get_objects(X[i]))

            for j in range(len(X)):
                j_sub = set(self.get_subjects(X[j]))
                numerator = len(i_object.intersection(j_sub)) * 1.0
                denominator = len(i_object.union(j_sub))
                ratio = numerator / denominator if denominator > 0 else 0
                distance_matrix[i][j] = ratio

        return distance_matrix

    def reverse_transitivity(self, X):
        distance_matrix = numpy.zeros((len(X), len(X)))

        for i in range(len(X)):
            i_subjects = set(self.get_subjects(X[i]))

            for j in range(len(X)):
                j_objects = set(self.get_objects(X[j]))

                numerator = len(i_subjects.intersection(j_objects)) * 1.0
                denominator = len(i_subjects.union(j_objects))
                ratio = numerator / denominator if denominator > 0 else 0
                distance_matrix[i][j] = ratio

        return distance_matrix


    def agency(self, X):
        distance_matrix = numpy.zeros((len(X), len(X)))

        for i in range(len(X)):
            i_subject = set(self.get_subjects(X[i]))

            for j in range(len(X)):
                j_sub = set(self.get_subjects(X[j]))
                numerator = len(i_subject.intersection(j_sub)) * 1.0
                denominator = len(i_subject.union(j_sub))
                ratio = numerator / denominator if denominator > 0 else 0
                distance_matrix[i][j] = ratio

        return distance_matrix

    def patient(self, X):
        distance_matrix = numpy.zeros((len(X), len(X)))

        for i in range(len(X)):
            i_objects = set(self.get_objects(X[i]))
            for j in range(len(X)):
                j_objects = set(self.get_objects(X[j]))
                numerator = len(i_objects.intersection(j_objects)) * 1.0
                denominator = len(i_objects.union(j_objects))
                ratio = numerator / denominator if denominator > 0 else 0
                distance_matrix[i][j] = ratio

        return distance_matrix

    def SOIntersectionSO(self, X):
        epsilon = 0.00001
        co_var = np.eye(len(X))
        for i in range(len(X)):
            i_slice = np.nonzero(X[i])
            i_slice = list(map(lambda x, y: (x, y), i_slice[0], i_slice[1]))
            i_slice_subject = list(map(lambda x: (x[0]), i_slice))
            i_slice_object = list(map(lambda x: (x[1]), i_slice))

            i_slice = set(i_slice_object).union(set(i_slice_subject))
            i_slice_elts = len(i_slice)
            for j in range(len(X)):
                if i == j:
                    continue

                j_slice = np.nonzero(X[j])

                # for matlab files only (next 2 lines)
                j_slice_subject = set(j_slice[0])
                j_slice_object = set(j_slice[1])


                j_slice = set(j_slice_object).union(set(j_slice_subject))
                # j_slice = j_slice_subject
                j_slice_elts = len(j_slice)

                if j_slice_elts == i_slice_elts:
                    j_slice_elts = j_slice_elts + epsilon

                if j_slice_elts == 0:
                    co_var[i][j] = epsilon
                    continue

                cmn_elts = len(list(set(i_slice).intersection(j_slice))) * 1.0
                max_value = max([i_slice_elts, j_slice_elts])
                ratio = cmn_elts / max_value
                co_var[i][j] = ratio

        return co_var

    def get_distance_metrix(self, X):
        distance = variables.arguments['distance']
        print('computing matrix using {}'.format(distance))

        if distance == 'agency':
            return self.agency(X)
        elif distance == 'transitivity':
            return self.transitivity(X)
        elif distance == 'reverse_transitivity':
            return self.reverse_transitivity(X)
        elif distance == 'patient':
            return self.patient(X)
        else:
            variables.arguments['distance'] = "SOIntersectionSO"
            return self.SOIntersectionSO(X)

    def asymmetric_similarity(self, X, alpha=0.1, beta=0.9):
        co_var = np.eye(len(X))
        for i in range(len(X)):
            i_slice = np.nonzero(X[i])
            i_slice = list(map(lambda x, y: (x, y), i_slice[0], i_slice[1]))
            i_slice_subject = list(map(lambda x: (x[0]), i_slice))
            i_slice_object = list(map(lambda x: (x[1]), i_slice))
            i_slice = set(i_slice_object).union(set(i_slice_subject))

            i_slice_elts = len(i_slice)
            for j in range(len(X)):
                if i == j:
                    continue

                j_slice = np.nonzero(X[j])

                j_slice_subject = set(j_slice[0])
                j_slice_object = set(j_slice[1])
                j_slice = set(j_slice_object).union(set(j_slice_subject))

                xIntersectionY = len(list(set(i_slice).intersection(j_slice))) * 1.0
                xMinusY = len(list(set(i_slice).difference(j_slice))) * 1.0
                yMinusX = len(list(set(j_slice).difference(i_slice))) * 1.0

                similarity = xIntersectionY / (xIntersectionY + alpha * xMinusY + beta * yMinusX)
                co_var[i][j] = similarity

        return co_var

    def find_slice_co_var(self, X):
        co_var = self.get_distance_metrix(X)
        return co_var

    '''
    def get_t_sne(self, matrix, rank):
        norm_mat = normalize(matrix, norm='l1', axis=1)
        result = [x * 1 for x in norm_mat]
        model = TSNE(n_components=rank, random_state=0, n_iter=1000)
        np.set_printoptions(suppress=False)
        compressed_mat = model.fit_transform(result)
        return compressed_mat
    '''