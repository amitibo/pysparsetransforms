"""
"""

from __future__ import division
import numpy as np
import sparse_transforms as spt
from sparse_transforms import cytransforms as cyt
import unittest
import matplotlib.pyplot as plt

eps = 1e-10

class TestCytransforms(unittest.TestCase):
    
    def setUp(self):
        
        Y = np.linspace(0, 5, 40)
        X = np.linspace(0, 5, 40)
        Z = np.linspace(0, 5, 40)

        #
        # The grids are created without the last index as self.Y-Z represent closed
        # grids.
        #
        self.grids = spt.Grids(Y, X, Z)
        
        self.p0 = np.array((4.9, 4.9, 0.5))
        self.p1 = np.array((0.1, 0.1, 0.1))
        self.Y, self.X, self.Z = self.grids.closed
        
    @unittest.skip("skip")    
    def test01(self):
        """Check local_argsearch_left"""
        
        self.assertTrue(cyt.test_local_argsearch_left(self.Y, 0.1) == 1)
    
    @unittest.skip("skip")    
    def test02(self):
        """Check calcCrossings"""
        
        r, indices = cyt.test_calcCrossings(self.Y, self.X, self.Z, self.p0, self.p1)
        
        for inds in zip(r, *np.unravel_index(indices, self.grids.shape)):
            print inds
            
        self.assertAlmostEqual(r.sum(), np.linalg.norm(self.p1-self.p0), msg='r=%g != d=%g' % (r.sum(), np.linalg.norm(self.p1-self.p0)))

    @unittest.skip("skip")    
    def test03(self):
        """Check interpolatePoints"""
        
        y_i0 = cyt.test_local_argsearch_left(self.Y, self.p0[0])
        y_i1 = cyt.test_local_argsearch_left(self.Y, self.p1[0])
        x_i0 = cyt.test_local_argsearch_left(self.X, self.p0[1])
        x_i1 = cyt.test_local_argsearch_left(self.X, self.p1[1])
        z_i0 = cyt.test_local_argsearch_left(self.Z, self.p0[2])
        z_i1 = cyt.test_local_argsearch_left(self.Z, self.p1[2])

        d = self.p1 - self.p0
        points_num = abs(y_i1 - y_i0) + abs(x_i1 - x_i0) + abs(z_i1 - z_i0)
        
        I = np.zeros((3, points_num), dtype=np.int32)
        P = np.empty((3, points_num))
        
        if cyt.test_interpolatePoints(self.Y, y_i0, y_i1, 0, 0, self.p0, d, I, P):
            sort_index = 0
        if cyt.test_interpolatePoints(self.X, x_i0, x_i1, abs(y_i1 - y_i0), 1, self.p0, d, I, P):
            sort_index = 1
        if cyt.test_interpolatePoints(self.Z, z_i0, z_i1, abs(y_i1 - y_i0)+abs(x_i1 - x_i0), 2, self.p0, d, I, P):
            sort_index = 2

        fig = plt.figure()        
        ax = fig.add_subplot(111)
        for i, c, m in zip((0, 1), ('r', 'g'), ('o', 'x')):
            crossing = I[i, :] != 0
            plt.plot(P[1,crossing], P[0,crossing], c=c, marker=m)        

        ax.set_ylim(self.Y[0], self.Y[-1])
        ax.set_xlim(self.X[0], self.X[-1])
        ax.set_yticks(self.Y)
        ax.set_xticks(self.X)
        plt.grid()
        plt.title('X, Y')
        
        fig = plt.figure()        
        ax = fig.add_subplot(111)
        for i, c, m in zip((1, 2), ('r', 'g'), ('o', 'x')):
            crossing = I[i, :] != 0
            plt.plot(P[1,crossing], P[2,crossing], c=c, marker=m)        

        ax.set_ylim(self.Z[0], self.Z[-1])
        ax.set_xlim(self.X[0], self.X[-1])
        ax.set_yticks(self.Z)
        ax.set_xticks(self.X)
        plt.grid()
        plt.title('X, Z')

        plt.show()

    #@unittest.skip("skip")
    def test04(self):
        """Check calcCrossings"""
        
        Y, X, Z = self.grids.closed
        
        r, indices = cyt.test_calcCrossings(Y, X, Z, self.p0, self.p1)
        zr = r > eps
        r = r[zr]
        indices = indices[zr]
        
        for inds in zip(r, indices, *np.unravel_index(indices, self.grids.shape)):
            print inds
            
        self.assertAlmostEqual(r.sum(), np.linalg.norm(self.p1-self.p0), msg='r=%g != d=%g' % (r.sum(), np.linalg.norm(self.p1-self.p0)))

if __name__ == '__main__':
    unittest.main()


    
    