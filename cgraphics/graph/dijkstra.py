# -*- coding: utf-8 -*-

'''
Created on 2016-9-5

@author: chin
'''

import numpy as np


class Dijkstra(object):
    def __init__(self):
        self._dis_array = None
        self._pat_array = None
        self.pt_start = None

    def dijkstra_adjoint(self, adj_array, pt_id):
        self.pt_start = pt_id
        pt_cnt = adj_array.shape[0]
        self._dis_array = np.array([np.inf for _ in xrange(pt_cnt)])
        self._pat_array = np.array([pt_id for _ in xrange(pt_cnt)])
        rest_array = np.copy(adj_array[pt_id, :])
        for it in range(pt_cnt):
            if np.isinf(rest_array).all():  # 已经找不到相连的路径
                break
            min_id = rest_array.argmin()
    #         if rest_array[min_id] == np.inf:
    #             break
            self._dis_array[min_id] = rest_array[min_id]
            rest_array[min_id] = np.inf     # 标示已经被选过了
            for j in range(pt_cnt):
                if self._dis_array[j] == np.inf:
                    if rest_array[j] > self._dis_array[min_id] + adj_array[min_id, j]:
                        rest_array[j] = self._dis_array[min_id] + adj_array[min_id, j]
                        self._pat_array[j] = min_id   # 代表在目前最有的i->j路径中, j的前驱为min_id

    @property
    def dis_array(self):
        return self._dis_array

    @property
    def pat_array(self):
        return self._pat_array

    def get_minimal_path(self, pt_e):
        if self._dis_array[pt_e] == np.inf:
            return np.inf, None
        min_path = [pt_e]
        while True:
            min_path.insert(0, self._pat_array[min_path[0]])
            if min_path[0] == self.pt_start:
                return self._dis_array[pt_e], min_path


if __name__ == "__main__":
    adj_array = np.array([[0,         50,     10,     np.inf,     45,     np.inf],
                          [np.inf,    0,      15,     np.inf,     5,      np.inf],
                          [20,        np.inf, 0,      15,         np.inf, np.inf],
                          [np.inf,    20,     np.inf, 0,          35,     np.inf],
                          [np.inf,    np.inf, np.inf, 30,         0,      np.inf],
                          [np.inf,    np.inf, np.inf, 3,          np.inf, 0]])
    dijkstra_method = Dijkstra()
    dijkstra_method.dijkstra_adjoint(adj_array, 0)
    print dijkstra_method.dis_array
    print dijkstra_method.pat_array
    print dijkstra_method.get_minimal_path(1)
