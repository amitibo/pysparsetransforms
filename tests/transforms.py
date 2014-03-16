import unittest

import matplotlib.pyplot as plt
import sparse_transforms as spt
import numpy as np
import time


def imshow(img, ax=None, *args, **kwargs):
    if ax is None:
        ax = plt.gca()
    im = ax.imshow(img, *args, **kwargs)
    
    def format_coord(x, y):
        x = int(x+0.5)
        y = int(y+0.5)
        
        return 'r=%d, c=%d, val=%g' % (x, y, img[y, x])
    
    ax.format_coord = format_coord
    
    ax.figure.canvas.draw()
    return im


class TestTransforms(unittest.TestCase):
    
    def setUp(self):
        
        self.Y = np.linspace(0, 2, 3)
        self.X = np.linspace(0, 2, 3)
        self.Z = np.linspace(0, 2, 3)
        
        self.grids = spt.Grids(self.Y, self.X, self.Z)
    
    def test01(self):
        """Test basic functions of the transform"""
        
        from scipy.sparse import lil_matrix
        
        in_grids = spt.Grids(np.arange(10), np.arange(10), np.arange(10))
        out_grids = spt.Grids(np.arange(10), np.arange(10), np.arange(10))
        
        A = lil_matrix((1000, 1000))
        A[0, :100] = np.random.rand(100)
        A[1, 100:200] = A[0, :100]
        A.setdiag(np.random.rand(1000))
        A = A.tocsr()
        
        B = lil_matrix((1000, 1000))
        B[0, :100] = np.random.rand(100)
        B[1, 100:200] = B[0, :100]
        B.setdiag(np.random.rand(1000))
        B = B.tocsr()

        T1 = spt.BaseTransform(A, in_grids=in_grids, out_grids=out_grids)
        T2 = spt.BaseTransform(B)
        
        C = T1 + T2
        
        self.assertTrue(isinstance(C, spt.BaseTransform))
        self.assertTrue(np.allclose((A+B).todense(), C.H.todense()))
        
        C = T1 * T2
        
        self.assertTrue(isinstance(C, spt.BaseTransform))
        self.assertTrue(np.allclose(np.dot(A, B).todense(), C.H.todense()))
    
        x = np.random.rand(1000, 1)
        b = T1 * x
        
        self.assertTrue(np.allclose(b, A * x))
    
        T3 = -T1
        
        self.assertTrue(np.allclose(T3.H.todense(), -(T1.H.todense())))

        T1.save('./test_save_transform')
        T4 = spt.loadTransform('./test_save_transform')
        
        self.assertTrue(np.allclose(T1.H.todense(), T4.H.todense()))
        
    def test02(self):
        """Test the direction and point transforms"""
        
        Y, X, Z = self.grids.expanded
        point = (1.0, 1.0, -3.0)
        
        t0 = time.time()
        
        #H1 = spt.directionTransform(self.grids, 0, np.pi/2)
        H2 = spt.pointTransform(self.grids, point)
        
        print time.time() - t0
    
        x = (Y<.5)
        y1 = H1 * x
        y2 = H2 * x
        
        import amitibo
        import mayavi.mlab as mlab
        
        mlab.figure()
        amitibo.viz3D(Y, X, Z, y1)
        mlab.title('Direction Transform')
        mlab.figure()
        amitibo.viz3D(Y, X, Z, y2)
        mlab.title('Point Transform')        
        mlab.show()

        
    def test03(self):
        """Test the sensor transform"""
        
        Y, X, Z = self.grids.expanded
        
        t0 = time.time()
        
        H = spt.sensorTransform(
            in_grids=self.grids,
            sensor_center=(1.0, 1., 0.0),
            sensor_res=(201, 201),
            depth_res=30,
            samples_num=1000,
            dither_noise=10,
            replicate=1
        )
        
        print time.time() - t0
    
        #plt.gray()
        #plt.imshow((H.H.todense()>0).astype(np.float), interpolation='nearest')
        #plt.show()
        
        x = (Z<0.2).astype(np.float)
        y = H * x
        
        import amitibo
        import mayavi.mlab as mlab
        
        amitibo.viz3D(Y, X, Z, x)
        Y_sensor, X_sensor, Z_sensor = H.out_grids.expanded
        amitibo.viz3D(Y_sensor, X_sensor, Z_sensor, y)
        mlab.show()

    def test_fisheye(self):
        """Test the fisheye transform"""
        
        Y, X, Z = self.grids.expanded
        
        t0 = time.time()
        
        H = spt.fisheyeTransform(
            in_grids=self.grids,
            sensor_center=(1.0, 1., 0.0),
            sensor_res=(501, 501),
            samples_num=1000,
            dither_noise=0.1,
            replicate=1
        )
        
        print time.time() - t0
        
        x = (Z<0.2).astype(np.float)
        y = H * x
        
        plt.gray()
        plt.imshow(y, interpolation='nearest')
        plt.show()

    def test04(self):
        """Test the integral transform"""
        
        Y, X, Z = self.grids.expanded
        
        H = spt.integralTransform(
            in_grids=self.grids
        )
        
        x = (X<1.).astype(np.float)
        y = H * x
        
        import amitibo
        import mayavi.mlab as mlab
        amitibo.viz3D(Y, X, Z, x)
        mlab.show()

        plt.figure()
        plt.imshow(y)
        plt.show()
        
    def test05(self):
        """Test the integral transform"""
        
        Y, X, Z = self.grids.expanded
        
        H = spt.cumsumTransform(
            in_grids=self.grids,
            direction=-1
        )
        
        x = (X<1.).astype(np.float)
        y = H * x
        
        import amitibo
        import mayavi.mlab as mlab
        amitibo.viz3D(Y, X, Z, x)
        amitibo.viz3D(Y, X, Z, y)

        mlab.show()
    
    def test06(self):
        """Test the SensorTransform"""
        
        cart_grids = spt.Grids(
            np.arange(0, 50000, 1000.0),
            np.arange(0, 50000, 1000.0),
            np.arange(0, 10000, 100.0)
        )
        
        #H1 = spt.sensorTransform(in_grids=cart_grids, sensor_center=(25001.0, 25001.0, 1.0), sensor_res=128, depth_res=100, samples_num=4000, replicate=40)
        #H1.save('./sensor_transform')
        H1 = spt.loadTransform('./sensor_transform')
        #H2 = spt.integralTransform(in_grids=H1.out_grids)
        #H2.save('./integral_transform')
        H2 = spt.loadTransform('./integral_transform')
                
        #
        # Create the GUI
        #
        from enthought.traits.api import HasTraits, Range, on_trait_change, Float, Instance
        from enthought.traits.ui.api import View, Item, HGroup, VGroup, EnumEditor
        from enthought.chaco.api import Plot, ArrayPlotData, gray, PlotLabel
        from chaco.tools.cursor_tool import CursorTool, BaseCursorTool
        from enthought.enable.component_editor import ComponentEditor
        import atmotomo

        class resultAnalayzer(HasTraits):
            """Gui Application"""
            
            x = Range(0, 49, 24, desc='pixel coord x', enter_set=True,
                      auto_set=False)
            y = Range(0, 49, 24, desc='pixel coord y', enter_set=True,
                      auto_set=False)
            z = Range(0, 99, desc='pixel coord z', enter_set=True,
                      auto_set=False)
            scaling = Range(-5.0, 5.0, 0.0, desc='Radiance scaling logarithmic')

            tr_cross_plot1 = Instance(Plot)
            tr_cross_plot2 = Instance(Plot)
            tr_cursor = Instance(BaseCursorTool)
            
            traits_view  = View(
                VGroup(
                    HGroup(
                        Item('img_container', editor=ComponentEditor(), show_label=False),
                        VGroup(
                            Item('tr_cross_plot1', editor=ComponentEditor(), show_label=False),
                            Item('tr_cross_plot2', editor=ComponentEditor(), show_label=False),
                            ),
                        ),
                    'y',
                    'x',
                    'z',
                    'scaling'
                    ),
                resizable = True,
            )
        
            def __init__(self, H1, H2):
                super(resultAnalayzer, self).__init__()
        
                self.H1 = H1
                self.H2 = H2
                
                self.mu = atmotomo.calcScatterMu(self.H1.inv_grids, -np.pi/4)
                self.R_derivatives = (spt.Grids(*H1.out_grids.derivatives).expanded)[0]

                #
                # Prepare all the plots.
                # ArrayPlotData - A class that holds a list of numpy arrays.
                # Plot - Represents a correlated set of data, renderers, and
                # axes in a single screen region.
                #
                img = np.zeros((128, 128), dtype=np.float)
                
                self.plotdata = ArrayPlotData(result_img=img)
                
                self.img_container = Plot(self.plotdata)
                img_plot = self.img_container.img_plot('result_img', colormap=gray)[0]

                self.tr_cursor = CursorTool(
                    component=img_plot,
                    drag_button='left',
                    color='white',
                    line_width=1.0
                )                
                img_plot.overlays.append(self.tr_cursor)
                self.tr_cursor.current_position = 64, 64
        
                self._updateImg(img)

                self.tr_cross_plot1 = Plot(self.plotdata, resizable="h")
                self.tr_cross_plot1.height = 10
                plots = self.tr_cross_plot1.plot(("basex", "img_x"))
                self.tr_cross_plot1.overlays.append(PlotLabel("X section",
                                              component=self.tr_cross_plot1,
                                              font = "swiss 16",
                                              overlay_position="top"))        
                self.tr_cross_plot2 = Plot(self.plotdata, resizable="h")
                self.tr_cross_plot2.height = 10
                plots = self.tr_cross_plot2.plot(("basey", "img_y"))
                self.tr_cross_plot2.overlays.append(PlotLabel("Y section",
                                              component=self.tr_cross_plot2,
                                              font = "swiss 16",
                                              overlay_position="top"))        
                
            def _updateImg(self, img):
                img = img * 10**self.scaling
                img[img<0] = 0
                img[img>255] = 255
                
                self.plotdata.set_data('result_img', img.astype(np.uint8))

                self.plotdata.set_data('basex', np.arange(128))
                self.plotdata.set_data('basey', np.arange(128))
                self.plotdata.set_data('img_x', img[self.tr_cursor.current_index[1], :])
                self.plotdata.set_data('img_y', img[:, self.tr_cursor.current_index[0]])

            @on_trait_change('x, y, z, scaling, tr_cursor.current_index')
            def update_volume(self):
                V = np.zeros(cart_grids.shape)
                V[self.y, self.x, self.z] = 0.1
                
                #temp = (self.H1 * V) * atmotomo.calcHG(self.mu, .7) * np.exp(-self.H1 * V)
                temp = (self.H1 * V) * atmotomo.calcHG(self.mu, .7) * self.R_derivatives
                #temp = (self.H1 * V)
                
                img = self.H2 * temp
                
                self._updateImg(img)
                
        
        app = resultAnalayzer(H1, H2)
        app.configure_traits()
    
    def test07(self):
        """Test the SensorTransform, same as test06 but with mayavi scene"""
        
        cart_grids = spt.Grids(
            np.arange(0, 50000, 1000.0),
            np.arange(0, 50000, 1000.0),
            np.arange(0, 10000, 100.0)
        )
        
        #H1 = spt.SensorTransform(in_grids=cart_grids, sensor_center=(25001.0, 25001.0, 1.0), sensor_res=128, depth_res=100, samples_num=8000, replicate=40)
        #H1.save('./sensor_transform')
        H1 = spt.loadTransform('./sensor_transform')
        #H2 = spt.IntegralTransform(in_grids=H1.out_grids)
        #H2.save('./integral_transform')
        H2 = spt.loadTransform('./integral_transform')
                
        #
        # Create the GUI
        #
        from enthought.traits.api import HasTraits, Range, on_trait_change, Float, Instance
        from enthought.traits.ui.api import View, Item, VGroup, EnumEditor, HGroup
        from enthought.chaco.api import Plot, ArrayPlotData, gray
        from enthought.enable.component_editor import ComponentEditor
        from mayavi.core.ui.api import MlabSceneModel, SceneEditor
        from mayavi import mlab
        import atmotomo

        class resultAnalayzer(HasTraits):
            """Gui Application"""
            
            x = Range(0, 49, 24, desc='pixel coord x', enter_set=True,
                      auto_set=False)
            y = Range(0, 49, 24, desc='pixel coord y', enter_set=True,
                      auto_set=False)
            z = Range(0, 99, desc='pixel coord z', enter_set=True,
                      auto_set=False)
            scaling = Range(-5.0, 5.0, 0.0, desc='Radiance scaling logarithmic')

            scene = Instance(MlabSceneModel, ())
            
            traits_view  = View(
                VGroup(
                    HGroup(
                        Item('img_container', editor=ComponentEditor(), show_label=False),
                        Item('scene',
                             editor=SceneEditor(), height=250,
                             width=300),
                    ),                        
                    'y',
                    'x',
                    'z',
                    'scaling'
                    ),
                resizable = True,
            )
        
            def __init__(self, H1, H2):
                super(resultAnalayzer, self).__init__()
        
                self.H1 = H1
                self.H2 = H2
                
                self.mu = atmotomo.calcScatterMu(self.H1.inv_grids, np.pi/4)

                #
                # Prepare all the plots.
                # ArrayPlotData - A class that holds a list of numpy arrays.
                # Plot - Represents a correlated set of data, renderers, and
                # axes in a single screen region.
                #
                img = np.zeros((128, 128), dtype=np.float)
                
                self.plotdata = ArrayPlotData()
                self._updateImg(img)
                
                self.img_container = Plot(self.plotdata)
                self.img_container.img_plot('result_img', colormap=gray)
                        
            @on_trait_change('scene.activated')
            def updateScene(self):
                mlab.clf(figure=self.scene.mayavi_scene)
                
                X, Y = np.mgrid[-1:1:128j, -1:1:128j]
        
                self.scene.mlab.surf(X, Y, self.img, colormap='gist_earth', warp_scale='auto')
                self.scene.mlab.axes()
                
            def _updateImg(self, img):
                img = img * 10**self.scaling
                
                self.img = img
                
                self.plotdata.set_data('result_img', img)
                self.updateScene()
                
            @on_trait_change('x, y, z, scaling')
            def update_volume(self):
                V = np.zeros(cart_grids.shape)
                V[self.y, self.x, self.z] = 0.1
                
                #temp = (self.H1 * V) * atmotomo.calcHG(self.mu, .7) * np.exp(-self.H1 * V)
                temp = self.H1 * V/10
                print '------------'
                print temp.min(), temp.max()
                #temp = np.exp(-temp)
                print temp.min(), temp.max()
                img = self.H2 * temp
                print img.min(), img.max()
                
                self._updateImg(img)
                
        
        app = resultAnalayzer(H1, H2)
        app.configure_traits()
    
    def test08(self):
        """Test the sensor transform"""
        
        import amitibo
        import mayavi.mlab as mlab
        
        H = spt.loadTransform('./sensor_transform')
        
        #
        # Create the GUI
        #
        from enthought.traits.api import HasTraits, Range, on_trait_change, Float, Instance
        from enthought.traits.ui.api import View, Item, VGroup, EnumEditor, HGroup
        from enthought.chaco.api import Plot, ArrayPlotData, gray
        from enthought.enable.component_editor import ComponentEditor
        from mayavi.core.ui.api import MlabSceneModel, SceneEditor
        from mayavi import mlab
        import atmotomo

        class resultAnalayzer(HasTraits):
            """Gui Application"""
            
            x = Range(0, 49, 24, desc='pixel coord x', enter_set=True,
                      auto_set=False)
            y = Range(0, 49, 24, desc='pixel coord y', enter_set=True,
                      auto_set=False)
            z = Range(0, 99, desc='pixel coord z', enter_set=True,
                      auto_set=False)

            scene1 = Instance(MlabSceneModel, ())
            scene2 = Instance(MlabSceneModel, ())
            
            traits_view  = View(
                VGroup(
                    HGroup(
                        Item('scene1',
                             editor=SceneEditor(), height=250,
                             width=300),
                         Item('scene2',
                             editor=SceneEditor(), height=250,
                             width=300),
                    ),                        
                    'y',
                    'x',
                    'z'
                    ),
                resizable = True,
            )
        
            def __init__(self, H):
                super(resultAnalayzer, self).__init__()
        
                self.H = H
                
            @on_trait_change('scene1.activated')
            def createScene1(self):
                mlab.clf(figure=self.scene1.mayavi_scene)
                
                Y, X, Z = self.H.in_grids.expanded
                
                self.src1 = mlab.pipeline.scalar_field(Y, X, Z, np.zeros_like(Y), figure=self.scene1.mayavi_scene)
                ipw_x = mlab.pipeline.image_plane_widget(self.src1, plane_orientation='x_axes')
                ipw_y = mlab.pipeline.image_plane_widget(self.src1, plane_orientation='y_axes')
                ipw_z = mlab.pipeline.image_plane_widget(self.src1, plane_orientation='z_axes')
                mlab.colorbar()
                mlab.axes()
                
            @on_trait_change('scene2.activated')
            def createScene2(self):
                mlab.clf(figure=self.scene2.mayavi_scene)
                
                Y, X, Z = self.H.out_grids.expanded
                
                self.src2 = mlab.pipeline.scalar_field(Y, X, Z,  np.zeros_like(Y), figure=self.scene2.mayavi_scene)
                ipw_x = mlab.pipeline.image_plane_widget(self.src2, plane_orientation='x_axes')
                ipw_y = mlab.pipeline.image_plane_widget(self.src2, plane_orientation='y_axes')
                ipw_z = mlab.pipeline.image_plane_widget(self.src2, plane_orientation='z_axes')
                mlab.colorbar()
                mlab.axes()
                
            @on_trait_change('x, y, z, scaling')
            def update_volume(self):
                V = np.zeros(self.H.in_grids.shape)
                V[self.y, self.x, self.z] = 1
                
                self.src1.mlab_source.scalars = V
                self.src2.mlab_source.scalars = self.H * V
                
                
        
        app = resultAnalayzer(H)
        app.configure_traits()
    
    def test09(self):
        
        H1 = spt.loadTransform('./sensor_transform')
        H2 = spt.loadTransform('./integral_transform')

        V = np.zeros(H1.in_grids.shape)
        V[24, 24, 2] = 0.1
        
        temp1 = H1 * V/10
        img1 = H2 * temp1
        
        temp2 = np.exp(-temp1)
        img2 = H2 * temp2
        
        plt.figure()
        plt.subplot(121)
        imshow(temp1[:, 60, :], interpolation='nearest')
        plt.subplot(122)
        imshow(temp2[:, 60, :], interpolation='nearest')
        
        plt.figure()
        plt.subplot(121)
        plt.plot(img1[:, 60])
        plt.subplot(122)
        plt.plot(img2[:, 60])
        
        plt.figure()
        plt.plot(H2.in_grids.derivatives[0])
        
        plt.show()
        
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

    def test10(self):
        """Test the point transform"""
        
        Y, X, Z = self.grids.expanded
        point = (1, 1, -1)
        
        t0 = time.time()
        
        H = spt.directionTransform(point, self.grids)
        
        print time.time() - t0
    
        x = (Y<.5)
        y = H * x
        
        import amitibo
        import mayavi.mlab as mlab
        
        amitibo.viz3D(Y, X, Z, y)
        mlab.show()
        

if __name__ == '__main__':
    unittest.main()

