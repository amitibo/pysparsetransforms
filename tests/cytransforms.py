"""
"""

from __future__ import division
import numpy as np
import sparse_transforms as spt
from sparse_transforms import cytransforms as cyt
import unittest
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class TestCytransforms(unittest.TestCase):
    
    def setUp(self):
        
        self.Y = np.linspace(0, 2, 5)
        self.X = np.linspace(0, 3, 6)
        self.Z = np.linspace(0, 4, 7)
        
        self.grids = spt.Grids(self.Y, self.X, self.Z)
        
    def test01(self):
        """Check local_argsearch_left"""
        
        self.assertTrue(cyt.test_local_argsearch_left(self.Y, 0.1) == 1)
    
    def test02(self):
        """Check interpolatePoints"""
        
        p0 = np.array((0.1, 0.1, 0.1))
        p1 = np.array((1.9, 1.9, 1.9))
        
        y_i0 = cyt.test_local_argsearch_left(self.Y, p0[0])
        y_i1 = cyt.test_local_argsearch_left(self.Y, p1[0])
        x_i0 = cyt.test_local_argsearch_left(self.X, p0[1])
        x_i1 = cyt.test_local_argsearch_left(self.X, p1[1])
        z_i0 = cyt.test_local_argsearch_left(self.Z, p0[2])
        z_i1 = cyt.test_local_argsearch_left(self.Z, p1[2])

        d = np.empty(3)
        d[0] = p1[0] - p0[0]
        d[1] = p1[1] - p0[1]
        d[2] = p1[2] - p0[2]
        points_num = abs(y_i1 - y_i0) + abs(x_i1 - x_i0) + abs(z_i1 - z_i0)
        
        I = np.zeros((3, points_num), dtype=np.int32)
        P = np.empty((3, points_num))
        
        if cyt.test_interpolatePoints(self.Y, y_i0, y_i1, 0, 0, p0, d, I, P):
            sort_index = 0
        if cyt.test_interpolatePoints(self.X, x_i0, x_i1, abs(y_i1 - y_i0), 1, p0, d, I, P):
            sort_index = 1
        if cyt.test_interpolatePoints(self.Z, z_i0, z_i1, abs(y_i1 - y_i0)+abs(x_i1 - x_i0), 2, p0, d, I, P):
            sort_index = 2

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        
        for i, (c, m) in enumerate(zip(('r', 'g', 'b'), ('o', '^', 'x'))):
            crossing = I[i, :] == 1
            ax.scatter(P[1,crossing], P[0,crossing], P[2,crossing], c=c, marker=m)        

        ax.set_ylim(self.Y[0], self.Y[-1])
        ax.set_xlim(self.X[0], self.X[-1])
        ax.set_zlim(self.Z[0], self.Z[-1])
        ax.set_yticks(self.Y)
        ax.set_xticks(self.X)
        ax.set_zticks(self.Z)
        plt.show()

    
if __name__ == '__main__':
    unittest.main()


    
    