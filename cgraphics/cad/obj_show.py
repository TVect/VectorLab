# -*- coding: utf-8 -*-

'''
Created on 2016-9-2

@author: chin
'''

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

class ObjView(object):

    def __init__(self):
        super(ObjView, self).__init__()
        self.vertices = []
        self.faces = []


    def load_obj(self, filename):
        with open(filename) as fr:
            for line in fr:
                item = line.split()
                if item[0] == 'v':
                    self.vertices.append(map(float, item[1:]))
                elif item[0] == 'f':
                    self.faces.append(map(int, item[1:]))
        self.vertices = np.array(self.vertices)
        self.faces = np.array(self.faces)


    def visual(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.plot_trisurf(self.vertices[:, 0], self.vertices[:, 1], self.vertices[:, 2], 
                        triangles=self.faces-1)

#         ax.axis("off")
        ax.set_aspect("equal")

        # Maybe matplotlib does not yet set correctly equal axis in 3D
        # So we use a trick instead of set_aspect
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()
    
        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)
    
        # The plot bounding box is a sphere in the sense of the infinity
        # norm, hence I call half the max range the plot radius.
        plot_radius = 0.5*max([x_range, y_range, z_range])
    
        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
        plt.show()
