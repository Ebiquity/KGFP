from scipy.sparse import lil_matrix
from scipy.io.matlab import loadmat
from numpy import dot, array, zeros
import numpy as np
import os
import random
import pickle
import sys
from models import variables

class Tensor_Factory:
    def __init__(self):
        self.dimensions = None
        self.dimension_reader = None
        self.triple_position = None
        self.triple_position_reader = None
        self.dir = os.getcwd()
        self.X = None

    def is_mat(self, dataset_name):
        if 'kinship' in dataset_name or 'umls' in dataset_name:
            return True
        else:
            return False

    def __get_dimensions__(self, dimension_reader):
        self.dimension_reader = open(os.path.join(self.dir, dimension_reader), 'r')
        dimension_line = self.dimension_reader.readline()
        print(dimension_line)

        try:
            self.dimensions = [int(x) for x in dimension_line.strip().split("\t")]
        except:
            self.dimensions = [int(x) for x in dimension_line.strip().split(",")]

        n = self.dimensions[0]
        k = self.dimensions[1]
        return [n,k]


    def csv_tensor(self, data_reader, dimension_reader):

        [n,k] = self.__get_dimensions__(dimension_reader)
        print (n,k)
        for i in range(k):
            #T = lil_matrix((n, n))
            if self.X is None:
                self.X = []
            self.X.append(lil_matrix(lil_matrix((n, n)), i, dtype=int))
            print ('slicing for i = ', i, ' out of ', k, ' is done')

        empty_relation = []
        filled_relation = []
        with open(os.path.join(self.dir, data_reader), 'r') as self.triple_position_reader:

            for position in self.triple_position_reader:
                # rcs = row;column;slice
                rcs = position.strip().split("\t")
                row = int(rcs[0])
                column = int(rcs[2])
                slice = int(rcs[1])
                if slice not in filled_relation:
                    filled_relation.append(slice)
                print(slice, row, column)
                self.X[slice][row, column] = 1

        empty_relation = set(list([x for x in range(k)])).difference(filled_relation)
        empty_relation = sorted(empty_relation, reverse=True)
        for i in empty_relation:
            del self.X[i]

        return self.X, n, k, n*n*k

    def get_tensor(self, is_matlab):
        if is_matlab:
            mat = loadmat('./dataset/uml.mat')
            K = array(mat['Rs'], np.float32)
            K_ = K.copy()
            e, k = K.shape[0], K.shape[2]
            n = e
            SZ = e * e * k

            T = [lil_matrix(K_[:, :, i]) for i in range(k)]
            return n, k, SZ, T

        else:
            tensor_factory = Tensor_Factory()
            [T, n, k] = tensor_factory.csv_tensor(variables.triple_num, variables.dim)
            SZ = n * n * k
            return n, k, SZ, T



    def get_pos_neg(self, T, row, col, k, pos, neg, s=-1, dataset="dataset", fold_number=1):
        import sys
        py_version = sys.version_info.major
        try:
            if py_version == 2:
               pos_neg_instances = pickle.load(open(variables.project_dir +'datasets/{0}_{1}_of_{2}.p2'.format(dataset, fold_number, 5), 'rb'))
            else:
               pos_neg_instances = pickle.load(open(variables.project_dir +'datasets/{0}_{1}_of_{2}'.format(dataset, fold_number, 5), 'rb'))
            return pos_neg_instances["instances"]
        except:
            print('Tensor_Factory cannot load file')
            #sys.exit()
            print('Creating 5 folds for the dataset')
            pos_neg_instances = {}
            pickle.dump(pos_neg_instances,
                open(variables.project_dir + 'datasets/{0}_{1}_of_{2}'.format(dataset, fold_number, 5), 'wb'))

        pos_rnd = {}
        neg_rnd = {}

        for slice in range(k):
            if s >= 0:
                slice = s

            if pos_rnd.get(slice, None) is None:
                pos_rnd[slice] = []
            else:
                continue

            if neg_rnd.get(slice, None) is None:
                neg_rnd[slice] = []
            else:
                continue

            non_zero = np.nonzero(T[slice])  # , np.nonzero(X[i])[1])
            pos_ex = list(map(lambda x, y: (x, y), non_zero[0], non_zero[1]))
            for p in range(pos):
                rnd_idx = int(random.uniform(0, len(pos_ex) - 1))
                if pos_ex[rnd_idx] in pos_rnd:
                    p = p - 1
                    continue
                else:
                    tuple = (pos_ex[rnd_idx][0], pos_ex[rnd_idx][1], slice)
                    pos_rnd[slice].append(tuple)

            for n in range(neg):

                rnd_row = int(random.uniform(0, row - 1))
                rnd_col = int(random.uniform(0, col - 1))
                if (rnd_row, rnd_col, slice) in pos_rnd:
                    n = n - 1
                    continue
                else:
                    neg_rnd[slice].append((rnd_row, rnd_col, slice))

        pos_list = []
        neg_list = []
        for k in pos_rnd.keys():
            pos_list.append(pos_rnd[k])
            neg_list.append(neg_rnd[k])

        final_list = []
        if pos > 0:
            final_list = pos_list
        if neg > 0:
            final_list = final_list + neg_list

        pos_neg_instances["instances"] = final_list
        pickle.dump(pos_neg_instances,
                                        open(variables.project_dir + 'data/{0}_{1}_of_{2}'.format(dataset,
                                                                                                                fold_number,
                                                                                                                5),
                                             'wb'))
        return final_list

    def get_pos_neg_any(self, T, row, col, k, pos, neg, s=-1):
        pos_rnd = {}
        neg_rnd = {}

        for p in range(pos):
            rnd_slice = random.randint(0, len(T)-1)

            non_zero = np.nonzero(T[rnd_slice])  # , np.nonzero(X[i])[1])
            pos_ex = list(map(lambda x, y: (x, y), non_zero[0], non_zero[1]))
            rnd_idx = int(random.uniform(0, len(pos_ex) - 1))

            tuple = (pos_ex[rnd_idx][0], pos_ex[rnd_idx][1], rnd_slice)
            if tuple in pos_rnd:
                p = p -1
                continue
            else:
                if pos_rnd.get(rnd_slice) is None:
                    pos_rnd[rnd_slice] = []
                pos_rnd[rnd_slice].append(tuple)


        for n in range(neg):
            rnd_slice = random.randint(0, len(T) - 1)
            rnd_row = int(random.uniform(0, row - 1))
            rnd_col = int(random.uniform(0, col - 1))
            if (rnd_row, rnd_col, rnd_slice) in pos_rnd:
                n = n - 1
                continue
            else:
                if neg_rnd.get(rnd_slice) is None:
                    neg_rnd[rnd_slice] = []
                neg_rnd[rnd_slice].append((rnd_row, rnd_col, rnd_slice))


        pos_list = []
        neg_list = []

        for k in pos_rnd.keys():
            pos_list.append(pos_rnd[k])

        for k in neg_rnd.keys():
            neg_list.append(neg_rnd[k])

        return pos_list + neg_list

    def get_data_tensor(self, dataset_name, is_mat=True):
        # kinship, umls, wordnet, freebase, dbpedia

        file_path = ''
        mat_path = ''

        suffix = "datasets/{0}/{1}.p" if sys.version_info.major > 2 else "datasets/{0}/{1}.p2"

        if dataset_name in 'kinship':
            mat_path = variables.project_dir + 'datasets/alyawarradata.mat'
        elif dataset_name in 'umls':
            mat_path = variables.project_dir + 'datasets/uml.mat'
        elif dataset_name in 'wordnet':
            file_path = variables.project_dir + suffix.format("wordnet/wn18", "wn18.test.train")
        elif dataset_name in 'freebase':
            file_path = variables.project_dir + suffix.format("fb13", "fb13.test.train")
        elif dataset_name in 'dbpedia':
            file_path = variables.project_dir + suffix.format("dbpedia/person", "person")
        elif dataset_name in 'framenet':
            file_path = variables.project_dir + suffix.format("framenet", "framenet")
        elif dataset_name in 'wn18rr':
            file_path = variables.project_dir + suffix.format("wn18rr", "wn18rr")
        elif dataset_name in 'wn11ntn':
            file_path = variables.project_dir + suffix.format("wn11ntn", "wn11ntn")
        elif dataset_name in 'fb13ntn':
            file_path = variables.project_dir + suffix.format("fb13ntn", "fb13ntn")
        elif dataset_name in 'fb15k_237':
            file_path = variables.project_dir + suffix.format('fb15k_237', 'fb15k_237')
        elif dataset_name in 'frame_trigger_lex_unit_pos_tag':
            file_path = variables.project_dir + suffix.format('frame_trigger_lex_unit_pos_tag', 'frame_trigger_lex_unit_pos_tag')


        print(file_path)
        if not is_mat:
            tensor = pickle.load(open(file_path, "rb"))
            n, k, SZ, T = tensor["n"], tensor["k"], tensor["SZ"], tensor["T"]
        else:
            mat = loadmat(mat_path)
            K = array(mat['Rs'], np.float32)
            n, k = K.shape[0], K.shape[2]
            SZ = n * n * k
            T = [lil_matrix(K[:, :, i]) for i in range(k)]

        return n, k, SZ, T