# -*- coding: utf-8 -*-

'''
Created on 2016-9-7

@author: chin
'''

import numpy as np

class Floyd(object):
    def __init__(self):
        self._dis_array = None
        self._pat_array = None
        self._adj_array = None

    def floyd_adjoint(self, adj_array):
        pt_cnt = adj_array.shape[0]
        self._adj_array = np.copy(adj_array)
        self._dis_array = np.copy(adj_array)
        self._pat_array = np.array([range(pt_cnt) for _ in range(pt_cnt)])

        for i in range(pt_cnt):
            for j in range(pt_cnt):
                for k in range(pt_cnt):
                    if self._dis_array[j, k] > self._dis_array[j, i] + self._dis_array[i, k]:
                        self._dis_array[j, k] = self._dis_array[j, i] + self._dis_array[i, k]
                        self._pat_array[j, k] = self._pat_array[j, i]

    @property
    def pat_array(self):
        return self._pat_array

    @property
    def dis_array(self):
        return self._dis_array

    def get_minimal_path(self, pt_s, pt_e):
        min_dist = self._dis_array[pt_s, pt_e]
        min_path = [pt_s]
        if min_dist == np.inf:
            return min_dist, []
        while True:
            min_path.append(self._pat_array[min_path[-1], pt_e])
            if min_path[-1] == pt_e:
                return min_dist, min_path


if __name__ == "__main__":
    adj_array = np.array([[0,         50,     10,     np.inf,     45,     np.inf],
                          [np.inf,    0,      15,     np.inf,     5,      np.inf],
                          [20,        np.inf, 0,      15,         np.inf, np.inf],
                          [np.inf,    20,     np.inf, 0,          35,     np.inf],
                          [np.inf,    np.inf, np.inf, 30,         0,      np.inf],
                          [np.inf,    np.inf, np.inf, 3,          np.inf, 0]])
    floyd_method = Floyd()
    floyd_method.floyd_adjoint(adj_array)
    print floyd_method.dis_array
    print floyd_method.pat_array
    print floyd_method.get_minimal_path(0, 1)
