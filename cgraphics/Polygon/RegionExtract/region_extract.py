# -*- coding: utf-8 -*-

'''
Created on 2016-8-24

@author: chin

reference: 
《An optimal algorithm for extracting the regions of a plane graph》X.Y.Jiang, H.Bunke 1992
'''

from __future__ import division

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from itertools import groupby
from operator import itemgetter

plt.style.use('ggplot') 

ACCEPT_HATCH = r"/\|-+xoO.*"


class PlaneGraph(object):
    '''
    An optimal algorithm for extracting the regions of a plane graph

    >>> pg = PlaneGraph(vertices=[[0,0],[0,1],[0.5,1.5],[1,1],[1,0],
    ...                           [2,0],[2,-1],[2.5,-1.5],[3,-1],[3,0]],
    ...                 edges=[[0,1],[1,2],[2,3],[3,4],[4,0],[1,3],
    ...                        [5,6],[6,7],[7,8],[8,9],[9,5],[6,8]])
    >>> pg.extract_regions()
    [[0, 1, 3, 4], [1, 2, 3], [5, 9, 8, 6], [6, 8, 7]]
    >>> # pg.visual()
    '''

    def __init__(self, vertices=[], edges=[]):
        self.vertices = np.array(vertices)
        self.edges = edges
        self.wedges = []
        self.regions = []


    def visual(self):
        fig, ax = plt.subplots()
        for edge in self.edges:
#             print self.vertices[[edge[0],edge[1]], 0], self.vertices[[edge[0],edge[1]], 1]
            ax.plot(self.vertices[[edge[0],edge[1]], 0], self.vertices[[edge[0],edge[1]], 1], "ro-", linewidth=3)
        if self.regions:
            for idx, region in enumerate(self.regions):
                ax.add_patch(mpatches.Polygon(self.vertices[region], hatch=ACCEPT_HATCH[idx], label="patch-%d"%idx))
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.spines['left'].set_position(('data', 0))
        ax.spines['bottom'].set_position(('data', 0))
        ax.set_xlim(self.vertices[:, 0].min()-1, self.vertices[:,0].max()+1)
        ax.set_ylim(self.vertices[:, 1].min()-1, self.vertices[:,1].max()+1)
        plt.legend(loc='best')
        plt.show()


    def extract_regions(self):
        self._find_wedges()
        self._wedges_to_regions()
        return self.regions


    def _find_wedges(self):
        complete_edges = []
        for edge in self.edges:
            complete_edges.append([edge[0], edge[1], 
                                   self.calc_edge_angle(np.array([self.vertices[edge[0]], self.vertices[edge[1]]]))])
            complete_edges.append([edge[1], edge[0], 
                                   self.calc_edge_angle(np.array([self.vertices[edge[1]], self.vertices[edge[0]]]))])

        complete_edges = sorted(complete_edges, key=lambda x: (x[0], x[2]))
        for key, item in groupby(complete_edges, itemgetter(0)):
            grouped = []
            for sub_item in item:
                if grouped:
                    self.wedges.append([grouped[-1][1], key, sub_item[1], sub_item[2]-grouped[-1][2]])
                grouped.append(sub_item)
            self.wedges.append([grouped[-1][1], key, grouped[0][1], 2*np.pi+grouped[0][2]-grouped[-1][2]])
        return self.wedges


    def _wedges_to_regions(self):
        sorted_wedges = sorted(self.wedges, key=lambda x: (x[0], x[1]))
        polys = []
        while sorted_wedges:
            if not polys:
                polys.append(sorted_wedges[0])
                sorted_wedges.remove(polys[0])
            else:         
                wedge = self._bi_search_wedge(polys[-1][1], polys[-1][2], sorted_wedges)
                polys.append(wedge)
                sorted_wedges.remove(wedge)
                if (wedge[1], wedge[2]) == (polys[0][0], polys[0][1]):
#                     print "polygon:", polys
                    # distinguish interior angle or exterior angle
                    if sum([wed[-1] for wed in polys]) < np.pi*len(polys):
                        self.regions.append([wed[0] for wed in polys])
                    polys = []


    def _bi_search_wedge(self, key1, key2, sorted_wedges):
        # TODO Binary Search
        wedges_cnt = len(sorted_wedges)
        if wedges_cnt == 0:
            return None
        if key1 == sorted_wedges[wedges_cnt//2][0]:
            if key2 == sorted_wedges[wedges_cnt//2][1]:
                return sorted_wedges[wedges_cnt//2]
            elif key2 < sorted_wedges[wedges_cnt//2][1]:
                return self._bi_search_wedge(key1, key2, sorted_wedges[:wedges_cnt//2])
            else:
                return self._bi_search_wedge(key1, key2, sorted_wedges[wedges_cnt//2+1:])
        elif key1 < sorted_wedges[wedges_cnt//2][0]:
            return self._bi_search_wedge(key1, key2, sorted_wedges[:wedges_cnt//2])
        else:
            return self._bi_search_wedge(key1, key2, sorted_wedges[wedges_cnt//2+1:])
        
#         for item1, item2, item3, theta in sorted_wedges:
#             if item1 == key1 and item2 == key2:
#                 return [item1, item2, item3, theta]


    @classmethod
    def calc_edge_angle(cls, edge):
        vect = edge[1] - edge[0]
        if vect[1] >= 0:
            return np.arccos(vect[0]/np.sqrt(np.dot(vect, vect)))
        elif vect[1] < 0:
            return 2 * np.pi - np.arccos(vect[0]/np.sqrt(np.dot(vect, vect)))


if __name__ == "__main__":
    import doctest
    doctest.testmod()
