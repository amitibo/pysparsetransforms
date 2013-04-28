import unittest

import matplotlib.pyplot as plt
import sparse_transforms as spt
import numpy as np
import time


class TestTransforms(unittest.TestCase):
    
    def test01(self):
        """Test basic functions of the transform"""
        
        from scipy.sparse import lil_matrix
        
        A = lil_matrix((1000, 1000))
        A[0, :100] = np.random.rand(100)
        A[1, 100:200] = A[0, :100]
        A.setdiag(np.random.rand(1000))
        A.tocsr()
        
        B = lil_matrix((1000, 1000))
        B[0, :100] = np.random.rand(100)
        B[1, 100:200] = B[0, :100]
        B.setdiag(np.random.rand(1000))
        B.tocsr()

        T1 = spt.BaseTransform(A)
        T2 = spt.BaseTransform(B)
        
        C = T1 + T2
        
        self.assertTrue(isinstance(C, spt.BaseTransform))
        self.assertTrue(np.allclose((A+B).todense(), C.H.todense()))
        
    @unittest.skip("not implemented")    
    def test2D(self):
    
        from scipy.misc import lena
        
        ##############################################################
        # 2D data
        ##############################################################
        lena = lena()
        lena = lena[:256, ...]
        lena_ = lena.reshape((-1, 1))    
        X, Y = np.meshgrid(np.arange(lena.shape[1]), np.arange(lena.shape[0]))
    
        #
        # Polar transform
        #
        t0 = time.time()
        Hpol = spt.polarTransformMatrix(X, Y, (256, 2))[0]
        lena_pol = Hpol * lena_
        print time.time() - t0
        
        plt.figure()
        plt.imshow(lena_pol.reshape((512, 512)), interpolation='nearest')
    
        #
        # Rotation transform
        #
        Hrot1, X_rot, Y_rot = spt.rotationTransformMatrix(X, Y, angle=-np.pi/3)
        Hrot2 = spt.rotationTransformMatrix(X_rot, Y_rot, np.pi/3, X, Y)[0]
        lena_rot1 = Hrot1 * lena_
        lena_rot2 = Hrot2 * lena_rot1
    
        plt.figure()
        plt.subplot(121)
        plt.imshow(lena_rot1.reshape(X_rot.shape))
        plt.subplot(122)
        plt.imshow(lena_rot2.reshape(lena.shape))
    
        #
        # Cumsum transform
        #
        Hcs1 = spt.cumsumTransformMatrix((Y, X), axis=0, direction=1)
        Hcs2 = spt.cumsumTransformMatrix((Y, X), axis=1, direction=1)
        Hcs3 = spt.cumsumTransformMatrix((Y, X), axis=0, direction=-1)
        Hcs4 = spt.cumsumTransformMatrix((Y, X), axis=1, direction=-1)
        lena_cs1 = Hcs1 * lena_
        lena_cs2 = Hcs2 * lena_
        lena_cs3 = Hcs3 * lena_
        lena_cs4 = Hcs4 * lena_
    
        plt.figure()
        plt.subplot(221)
        plt.imshow(lena_cs1.reshape(lena.shape))
        plt.subplot(222)
        plt.imshow(lena_cs2.reshape(lena.shape))
        plt.subplot(223)
        plt.imshow(lena_cs3.reshape(lena.shape))
        plt.subplot(224)
        plt.imshow(lena_cs4.reshape(lena.shape))
    
        plt.show()
        
    
    @unittest.skip("not implemented")    
    def test3D(self):
    
        #
        # Test several of the above functions
        #
        ##############################################################
        # 3D data
        ##############################################################
        Y, X, Z = np.mgrid[-10:10:50j, -10:10:50j, -10:10:50j]
        V = np.sqrt(Y**2 + X**2 + Z**2)
        V_ = V.reshape((-1, 1))
        
        # #
        # # Spherical transform
        # #
        # t0 = time.time()
        # Hsph = sphericalTransformMatrix(Y, X, Z, (0, 0, 0))[0]
        # Vsph = Hsph * V_
        # print time.time() - t0
         
        #
        # Rotation transform
        #
        t0 = time.time()
        Hrot, rotation, Y_rot, X_rot, Z_rot = spt.rotation3DTransformMatrix(Y, X, Z, (np.pi/4, np.pi/4, 0))
        Vrot = Hrot * V_
        Hrot2 = spt.rotation3DTransformMatrix(Y_rot, X_rot, Z_rot, np.linalg.inv(rotation), Y, X, Z)[0]
        Vrot2 = Hrot2 * Vrot
        print time.time() - t0
         
        # #
        # # Cumsum transform
        # #
        # Hcs1 = cumsumTransformMatrix((Y, X, Z), axis=0, direction=-1)
        # Vcs1 = Hcs1 * V_
    
        # #
        # # Integral transform
        # #
        # Hit1 = integralTransformMatrix((Y, X, Z), axis=0, direction=-1)
        # Vit1 = Hit1 * V_
    
        #
        # 3D visualization
        #
        import mayavi.mlab as mlab
        mlab.figure()
        s = mlab.pipeline.scalar_field(Vrot.reshape(Y_rot.shape))
        ipw_x = mlab.pipeline.image_plane_widget(s, plane_orientation='x_axes')
        ipw_y = mlab.pipeline.image_plane_widget(s, plane_orientation='y_axes')
        mlab.title('V Rotated')
        mlab.colorbar()
        mlab.outline()
        
        mlab.figure()
        s = mlab.pipeline.scalar_field(Vrot2.reshape(Y.shape))
        ipw_x = mlab.pipeline.image_plane_widget(s, plane_orientation='x_axes')
        ipw_y = mlab.pipeline.image_plane_widget(s, plane_orientation='y_axes')
        mlab.title('V Rotated Back')
        mlab.colorbar()
        mlab.outline()
        
        # mlab.figure()
        # mlab.contour3d(Vcs1.reshape(V.shape), contours=[1, 2, 3], transparent=True)
        # mlab.outline()
        
        mlab.show()
        
        #
        # 2D visualization
        #
        # import matplotlib.pyplot as plt
        
        # plt.figure()
        # plt.imshow(Vit1.reshape(V.shape[:2]))
        # plt.show()
    
    @unittest.skip("not implemented")    
    def testProjection(self):
    
        from scipy.misc import lena
        
        l = lena()
    
        PHI, THETA = np.mgrid[0:2*np.pi:512j, 0:np.pi/2*0.9:512j]
        
        H = spt.cameraTransformMatrix(PHI, THETA, focal_ratio=0.15)
        lp = H * l.reshape((-1, 1))
    
        plt.figure()
        plt.imshow(l)
        
        plt.figure()
        plt.imshow(lp.reshape((256, 256)))
    
        plt.show()


if __name__ == '__main__':
    unittest.main()

