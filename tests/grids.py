import unittest

import sparse_transforms as spt
import numpy as np


class TestGrids(unittest.TestCase):
    
    def setUp(self):
        
        self.Y = np.linspace(0, 1, 10)
        self.X = np.linspace(0, 2, 20)
        self.Z = np.linspace(0, 3, 30)
        
        self.grids = spt.Grids(self.Y, self.X, self.Z)
        self.expanded_grids = np.mgrid[0:2:20j, 0:2:20j, 0:2:20j]
        self.general_grids = spt.GeneralGrids(*self.expanded_grids)
        
    def test01(self):
        """Check shape"""
        
        self.assertEqual(self.grids.shape, (10, 20, 30))
        self.assertEqual(self.grids.size, 10*20*30)
        
        for i, (grid, expected_shape) in enumerate(zip(self.grids, ((10, 1, 1), (1, 20, 1), (1, 1, 30)))):
            self.assertEqual(grid.shape, expected_shape)

    def test02(self):
        """?"""
        
        self.grids.copy()
        
    def test03(self):
        """Test translation"""
        
        translation = (0.5, 1.5, 2.5)
        translated_grids = self.grids.translate(translation)
        
        for grid, orig_grid, delta in zip(translated_grids, (self.Y, self.X, self.Z), translation):
            self.assertTrue(np.allclose(grid.ravel(), orig_grid+delta))

    def test04(self):
        """Test expansion"""
        
        expanded_grids = self.grids.expanded

        for grid, ref_grid in zip(expanded_grids, self.expanded_grids):
            self.assertTrue(np.allclose(grid, ref_grid))
    
    def test05(self):
        """Test the general grid and its rotation"""
        
        Y, X, Z = self.general_grids.expanded
        
        Y_rot, X_rot, Z_rot = self.general_grids.rotate(np.pi/4, 0, 0)
        
        import mayavi.mlab as mlab
        import amitibo
        
        import amitibo
        import mayavi.mlab as mlab
        amitibo.viz3D(Y, X, Z, Y)
        amitibo.viz3D(Y, X, Z, X)
        amitibo.viz3D(Y, X, Z, Z)

        amitibo.viz3D(Y, X, Z, Y_rot)
        amitibo.viz3D(Y, X, Z, X_rot)
        amitibo.viz3D(Y, X, Z, Z_rot)
        mlab.show()

if __name__ == '__main__':
    unittest.main()

