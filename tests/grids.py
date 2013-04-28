import unittest

import sparse_transforms as spt
import numpy as np


class TestGrids(unittest.TestCase):
    
    def setUp(self):
        
        self.Y = np.linspace(0, 1, 10)
        self.X = np.linspace(1, 2, 20)
        self.Z = np.linspace(2, 3, 30)
        
        self.grids = spt.Grids(self.Y, self.X, self.Z)
        
    def test01(self):
        
        self.assertEqual(self.grids.shape, (10, 20, 30))
        self.assertEqual(self.grids.size, 10*20*30)
        
        for i, (grid, expected_shape) in enumerate(zip(self.grids, ((10, 1, 1), (1, 20, 1), (1, 1, 30)))):
            self.assertEqual(grid.shape, expected_shape)

    def test02(self):
        
        self.grids.copy()
        
    def test03(self):
        
        translation = (0.5, 1.5, 2.5)
        translated_grids = self.grids.translate(translation)
        
        for grid, orig_grid, delta in zip(translated_grids, (self.Y, self.X, self.Z), translation):
            self.assertTrue(np.allclose(grid.ravel(), orig_grid+delta))

        
if __name__ == '__main__':
    unittest.main()

