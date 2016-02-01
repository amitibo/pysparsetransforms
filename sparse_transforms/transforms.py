"""
"""

from __future__ import division
import numpy as np
import numpy.linalg as linalg
import scipy.sparse as sps
from .base import *
from .transformation_matrices import euler_matrix
from cytransforms import point2grids, direction2grids
import itertools

__all__ = (
    'directionTransform',
    'pointTransform',
    'rotationTransform',
    'integralTransform',
    'sensorTransform',
    'sensorTransformK',
    'cumsumTransform',
    'fisheyeTransform'
)


def directionTransform(
    in_grids,
    direction_phi,
    direction_theta
    ):

    H = direction2grids(
        direction_phi,
        direction_theta,
        in_grids.expanded[0],
        in_grids.expanded[1],
        in_grids.expanded[2]
    )

    return BaseTransform(
        H=H,
        in_grids=in_grids,
        out_grids=in_grids
    )


def pointTransform(
    in_grids,
    point
    ):

    Y, X, Z = in_grids.closed
    
    assert point[0] > Y.min() and point[0] < Y.max(), "point is not directly below the grid"
    assert point[1] > X.min() and point[1] < X.max(), "point is not directly below the grid"
    assert point[2] < Z.max(), "point is not directly below the grid"

    H = point2grids(
        point,
        in_grids.expanded[0],
        in_grids.expanded[1],
        in_grids.expanded[2]
    )

    return BaseTransform(
        H=H,
        in_grids=in_grids,
        out_grids=in_grids
    )


def rotationTransform(in_grids, rotation, out_grids=None):
    """Calculate a transform representing a rotation in 3D.
    
    Parameters
    ----------
    in_grids : Grids object
        List of grids. 

    rotation : list of floats or rotation matrix
        Either a list of floats representating the rotation in euler angles
        (axis used is 'sxyz'). Alternatively, rotation can be a 4x4 rotation matrix
    
    out_grids : Grids object, optional (default=None)
        List of grids. The grids are expected to be of the form created by mgrid
        and in the same order of creation. The transform is calculated into these
        grids. This enables croping of the target domain after the rotation transform.
        If none, the destination grids will be calculated to contain the full transformed
        source.
"""

    if isinstance(rotation, np.ndarray) and rotation.shape == (4, 4):
        H_rot = rotation
    else:
        H_rot = euler_matrix(*rotation)
        
    if out_grids == None:
        Y_dst, X_dst, Z_dst = _calcRotateGrid(in_grids, H_rot)
    else:
        Y_dst, X_dst, Z_dst = out_grids

    #
    # Calculate a rotated grid by applying the rotation.
    #
    XYZ_dst = np.vstack((X_dst.ravel(), Y_dst.ravel(), Z_dst.ravel(), np.ones(X_dst.size)))
    XYZ_src_ = np.dot(np.linalg.inv(H_rot), XYZ_dst)

    Y_indices = XYZ_src_[1, :].reshape(X_dst.shape)
    X_indices = XYZ_src_[0, :].reshape(X_dst.shape)
    Z_indices = XYZ_src_[2, :].reshape(X_dst.shape)

    H = calcTransformMatrix(in_grids, (Y_indices, X_indices, Z_indices))

    return BaseTransform(
        H=H,
        in_grids=in_grids,
        out_grids=Grids(Y_dst, X_dst, Z_dst)
    )

#
# Some globals
#
SPARSE_SIZE_LIMIT = 1e6
GRID_DIM_LIMIT = 100

def _calcRotateGrid(in_grid, H_rot):
    #
    # Calculate the target grid.
    # The calculation is based on calculating the minimal grid that contains
    # the transformed input grid.
    #
    Y_slim, X_slim, Z_slim = [g.ravel() for g in in_grid]
    x0_src = np.floor(np.min(X_slim)).astype(np.int)
    y0_src = np.floor(np.min(Y_slim)).astype(np.int)
    z0_src = np.floor(np.min(Z_slim)).astype(np.int)
    x1_src = np.ceil(np.max(X_slim)).astype(np.int)
    y1_src = np.ceil(np.max(Y_slim)).astype(np.int)
    z1_src = np.ceil(np.max(Z_slim)).astype(np.int)

    src_coords = np.array(
        [
            [x0_src, x0_src, x1_src, x1_src, x0_src, x0_src, x1_src, x1_src],
            [y0_src, y1_src, y0_src, y1_src, y0_src, y1_src, y0_src, y1_src],
            [z0_src, z0_src, z0_src, z0_src, z1_src, z1_src, z1_src, z1_src],
            [1, 1, 1, 1, 1, 1, 1, 1]
        ]
    )
    dst_coords = np.dot(H_rot, src_coords)

    x0_dst, y0_dst, z0_dst, dump = np.floor(np.min(dst_coords, axis=1)).astype(np.int)
    x1_dst, y1_dst, z1_dst, dump = np.ceil(np.max(dst_coords, axis=1)).astype(np.int)

    #
    # Calculate the grid density.
    # Note:
    # This calculation is important as having a dense grid results in a huge transform
    # matrix even if it is sparse.
    #
    dy, dx, dz = [d[0, 0, 0] for d in in_grid.derivatives]

    delta_src_coords = np.array(
        [
            [0, dx, 0, 0, -dx, 0, 0],
            [0, 0, dy, 0, 0, -dy, 0],
            [0, 0, 0, dz, 0, 0, -dz],
            [1, 1, 1, 1, 1, 1, 1]
        ]
    )
    delta_dst_coords = np.dot(H_rot, delta_src_coords)
    delta_dst_coords.sort(axis=1)
    delta_dst_coords = delta_dst_coords[:, 1:] - delta_dst_coords[:, :-1]
    delta_dst_coords[delta_dst_coords<=0] = 10000000
    
    dx, dy, dz, dump = np.min(delta_dst_coords, axis=1)
    x_samples = min(int((x1_dst-x0_dst)/dx), GRID_DIM_LIMIT)
    y_samples = min(int((y1_dst-y0_dst)/dy), GRID_DIM_LIMIT)
    z_samples = min(int((z1_dst-z0_dst)/dz), GRID_DIM_LIMIT)
    
    dim_ratio = x_samples * y_samples * z_samples / SPARSE_SIZE_LIMIT
    if  dim_ratio > 1:
        dim_reduction = dim_ratio ** (-1/3)
        
        x_samples = int(x_samples * dim_reduction)
        y_samples = int(y_samples * dim_reduction)
        z_samples = int(z_samples * dim_reduction)
        
    Y_dst, X_dst, Z_dst = np.mgrid[
        y0_dst:y1_dst:complex(0, y_samples),
        x0_dst:x1_dst:complex(0, x_samples),
        z0_dst:z1_dst:complex(0, z_samples),
    ]
    return Y_dst, X_dst, Z_dst


def integralTransform(
    in_grids,
    jacobian=None,
    axis=0,
    direction=1
    ):
    """
    Calculate a transform representing integration.

    Parameters
    ----------
    in_grids : Grids object
        List of grids. 

    jacobian : array like (default=None)
        If given, will be used as the Jacobian of the integration.

    axis : int, optional (default=0)
        The axis by which the integration is performed.

    direction : {1, -1}, optional (default=1)
        Direction of integration
        direction - 1: integrate up the indices, -1: integrate down the indices.
    """

    grid_shape = in_grids.shape
    strides = np.array(in_grids.expanded[0].strides)
    strides /= strides[-1]

    derivatives = in_grids.derivatives

    inner_stride = strides[axis]

    if direction != 1:
        direction  = -1

    inner_height = np.abs(inner_stride)
    inner_width = np.prod(grid_shape[axis:])

    inner_H = sps.spdiags(
        np.ones((grid_shape[axis], max(inner_height, inner_width)))*derivatives[axis].reshape((-1, 1))*direction,
        inner_stride*np.arange(grid_shape[axis]),
        inner_height,
        inner_width
    )

    if axis == 0:
        H = inner_H
    else:
        m = np.prod(grid_shape[:axis])
        H = sps.kron(sps.eye(m, m), inner_H)

    H = H.tocsr()

    if jacobian != None:
        H = H * spdiag(jacobian)

    temp = range(in_grids.ndim)
    temp.remove(axis)
    out_grids = [in_grids[i].ravel() for i in temp]

    return BaseTransform(
        H=H,
        in_grids=in_grids,
        out_grids=Grids(*out_grids)
    )


def cumsumTransform(
    in_grids,
    axis=0,
    direction=1,
    masked_rows=None
    ):
    """
    Calculate a transform representing cumsum operation.

    Parameters
    ----------
    in_grids : Grids object
        List of grids. 

    axis : int, optional (default=0)
        Axis along which the cumsum operation is preformed.

    direction : {1, -1}, optional (default=1)
        Direction of integration, 1 for integrating up the indices
        -1 for integrating down the indices.

    masked_rows: array, optional(default=None)
        If not None, leave only the rows that are non zero in the
        masked_rows array.
    """

    grid_shape = in_grids.shape
    strides = np.array(in_grids.expanded[0].strides)
    strides /= strides[-1]

    derivatives = in_grids.derivatives

    inner_stride = strides[axis]
    if direction == 1:
        inner_stride = -inner_stride

    inner_size = np.prod(grid_shape[axis:])

    inner_H = sps.spdiags(
        np.ones((grid_shape[axis], inner_size))*derivatives[axis].reshape((-1, 1)),
        inner_stride*np.arange(grid_shape[axis]),
        inner_size,
        inner_size)

    if axis == 0:
        H = inner_H
    else:
        m = np.prod(grid_shape[:axis])
        H = sps.kron(sps.eye(m, m), inner_H)

    if masked_rows != None:
        H = H.tolil()
        indices = masked_rows.ravel() == 0
        for i in indices.nonzero()[0]:
            H.rows[i] = []
            H.data[i] = []

    return BaseTransform(
        H=H.tocsr(),
        in_grids=in_grids,
        out_grids=in_grids
    )


def sensorTransform(
    in_grids,
    sensor_center,
    sensor_res,
    depth_res,
    samples_num=1000,
    dither_noise=10,
    replicate=10
    ):
    """
    Transofrmation of a linear fisheye (to a x, y, R space). In this version the camera is assumed to point up.
    Implements ray tracing algorithm.
    
    Parameters:
    -----------
    in_grids: Grids object
        Grids of the 3D space.
    sensor_center: array like
        Center of the camera/sensor
    sensor_res : two tuple.
        resolution of sensor as a tuple of two ints.
    depth_res : int
        Resolution of the R axis in the output grid.
    samples_num : int
        Number of samples along the R axis
    dither_noise : int
        Noise in the samples along the R axis (used for avoiding aliasing).
    replicate : int
        Number of replications at each pixel.
    """
    
    #
    # Center the grids
    #
    centered_grids = in_grids.translate(-np.array(sensor_center))
    Y, X, Z = centered_grids.closed

    #
    # Convert image pixels to ray direction
    # The image is assumed the [-1, 1]x[-1, 1] square.
    #
    Y_sensor, step = np.linspace(-1.0, 1.0, sensor_res[0], endpoint=False, retstep=True)
    X_sensor = np.linspace(-1.0, 1.0, sensor_res[1], endpoint=False)

    #
    # Calculate sample steps along ray
    #
    R_max = np.max(np.sqrt(centered_grids.expanded[0]**2 + centered_grids.expanded[1]**2 + centered_grids.expanded[2]**2))
    R_samples, R_step = np.linspace(0.0, R_max, samples_num, retstep=True)
    R_samples = R_samples[1:]
    R_dither = np.random.rand(*sensor_res) * R_step * dither_noise

    #
    # Calculate depth bins
    #
    #depth_bins = np.logspace(np.log10(R_samples[0]), np.log10(R_samples[-1]+R_step), depth_res+1)
    temp = np.linspace(0, 1, depth_res+1)
    temp = np.cumsum(temp)
    depth_bins = temp / temp[-1] * R_samples[-1]
    samples_bin = np.digitize(R_samples, depth_bins)
    samples_array = []
    for i in range(1, depth_res+1):
        samples_array.append(R_samples[samples_bin==i].reshape((-1, 1)))

    #
    # Create the output grids
    #
    out_grids = Grids(depth_bins[:-1], Y_sensor, X_sensor)

    #
    # Calculate inverse grid
    #
    X_sensor, Y_sensor = np.meshgrid(X_sensor, Y_sensor)
    R_sensor = np.sqrt(X_sensor**2 + Y_sensor**2)
    R = out_grids.expanded[0]
    THETA = R_sensor * np.pi / 2
    PHI = np.arctan2(Y_sensor, X_sensor)
    THETA = np.tile(THETA[np.newaxis, :, :], [depth_res, 1, 1])
    PHI = np.tile(PHI[np.newaxis, :, :], [depth_res, 1, 1])
    Y_inv = R * np.sin(THETA) * np.sin(PHI)
    X_inv = R * np.sin(THETA) * np.cos(PHI)
    Z_inv = R * np.cos(THETA)

    inv_grids = Grids(Y_inv, X_inv, Z_inv)

    #
    # Randomly replicate rays inside each pixel
    #
    X_sensor = np.tile(X_sensor[:, :, np.newaxis], [1, 1, replicate])
    Y_sensor = np.tile(Y_sensor[:, :, np.newaxis], [1, 1, replicate])
    X_sensor += np.random.rand(*X_sensor.shape)*step
    Y_sensor += np.random.rand(*Y_sensor.shape)*step

    #
    # Calculate rays angles
    # R_sensor is the radius from the center of the image (0, 0) to the
    # pixel. It is used for calculating th ray direction (PHI, THETA)
    # and for filtering pixels outside the image (radius > 1).
    #
    R_sensor = np.sqrt(X_sensor**2 + Y_sensor**2)
    THETA_ray = R_sensor * np.pi / 2
    PHI_ray = np.arctan2(Y_sensor, X_sensor)
    DY_ray = np.sin(THETA_ray) * np.sin(PHI_ray)
    DX_ray = np.sin(THETA_ray) * np.cos(PHI_ray)
    DZ_ray = np.cos(THETA_ray)

    #
    # Loop on all rays
    #
    data = []
    indices = []
    indptr = [0]
    for samples in samples_array:
        for r, dy, dx, dz, r_dither in itertools.izip(
            R_sensor.reshape((-1, replicate)),
            DY_ray.reshape((-1, replicate)),
            DX_ray.reshape((-1, replicate)),
            DZ_ray.reshape((-1, replicate)),
            R_dither.ravel(),
            ):
            if np.all(r > 1):
                indptr.append(indptr[-1])
                continue

            #
            # Filter steps where r > 1
            #
            dy = dy[r<=1]
            dx = dx[r<=1]
            dz = dz[r<=1]

            #
            # Convert the ray samples to volume indices
            #
            Y_ray = (r_dither+samples) * dy
            X_ray = (r_dither+samples) * dx
            Z_ray = (r_dither+samples) * dz

            #
            # Calculate the atmosphere indices
            #
            Y_indices = np.searchsorted(Y, Y_ray.ravel())
            X_indices = np.searchsorted(X, X_ray.ravel())
            Z_indices = np.searchsorted(Z, Z_ray.ravel())

            Y_filter = (Y_indices > 0) * (Y_indices < Y.size)
            X_filter = (X_indices > 0) * (X_indices < X.size)
            Z_filter = (Z_indices > 0) * (Z_indices < Z.size)

            filtered = Y_filter*X_filter*Z_filter
            Y_indices = Y_indices[filtered]-1
            X_indices = X_indices[filtered]-1
            Z_indices = Z_indices[filtered]-1

            #
            # Calculate unique indices
            #
            inds_ray = (Y_indices*centered_grids.shape[1] + X_indices)*centered_grids.shape[2] + Z_indices
            uniq_indices, inv_indices = np.unique(inds_ray, return_inverse=True)

            #
            # Calculate weights
            # Note:
            # The weights are divided by the number of samples in the voxels, this gives the
            # averaged concentration in the voxel.
            #
            weights = []
            for i, ind in enumerate(uniq_indices):
                weights.append((inv_indices == i).sum() / samples.size / replicate)

            #
            # Sum up the indices and weights
            #
            data.append(weights)
            indices.append(uniq_indices)
            indptr.append(indptr[-1]+uniq_indices.size)

    #
    # Create sparse matrix
    #
    data = np.hstack(data)
    indices = np.hstack(indices)

    H = sps.csr_matrix(
        (data, indices, indptr),
        shape=(sensor_res[0]*sensor_res[1]*depth_res, centered_grids.size)
    )

    return BaseTransform(
        H=H,
        in_grids=in_grids,
        out_grids=out_grids,
        inv_grids=inv_grids
    )


def fisheyeTransform(
    in_grids,
    sensor_center,
    sensor_res,
    samples_num=1000,
    dither_noise=10,
    replicate=10
    ):
    """
    Transofrmation of a linear fisheye (to a x, y). In this version the camera is assumed to point up.
    Implements ray tracing algorithm.

    Parameters:
    -----------
    in_grids: Grids object
        Grids of the 3D space.
    sensor_center: array like
        Center of the camera/sensor
    sensor_res : two tuple.
        resolution of sensor as a tuple of two ints.
    samples_num : int
        Number of samples along the R axis
    dither_noise : int
        Noise in the samples along the R axis (used for avoiding aliasing).
    replicate : int
        Number of replications at each pixel.
    """
    
    #
    # Center the grids
    #
    centered_grids = in_grids.translate(-np.array(sensor_center))
    Y, X, Z = centered_grids.closed

    #
    # Convert image pixels to ray direction
    # The image is assumed the [-1, 1]x[-1, 1] square.
    #
    Y_sensor, step = np.linspace(-1.0, 1.0, sensor_res[0], endpoint=False, retstep=True)
    X_sensor = np.linspace(-1.0, 1.0, sensor_res[1], endpoint=False)

    #
    # Calculate sample steps along ray
    #
    R_max = np.max(np.sqrt(centered_grids.expanded[0]**2 + centered_grids.expanded[1]**2 + centered_grids.expanded[2]**2))
    R_samples, R_step = np.linspace(0.0, R_max, samples_num, retstep=True)
    R_samples = R_samples[1:]
    R_dither = np.random.rand(*sensor_res) * R_step * dither_noise

    #
    # Create the output grids
    #
    out_grids = Grids(Y_sensor, X_sensor)

    #
    # Randomly replicate rays inside each pixel
    #
    X_sensor, Y_sensor = np.meshgrid(X_sensor, Y_sensor)
    X_sensor = np.tile(X_sensor[:, :, np.newaxis], [1, 1, replicate])
    Y_sensor = np.tile(Y_sensor[:, :, np.newaxis], [1, 1, replicate])
    X_sensor += np.random.rand(*X_sensor.shape)*step
    Y_sensor += np.random.rand(*Y_sensor.shape)*step

    #
    # Calculate rays angles
    # R_sensor is the radius from the center of the image (0, 0) to the
    # pixel. It is used for calculating th ray direction (PHI, THETA)
    # and for filtering pixels outside the image (radius > 1).
    #
    R_sensor = np.sqrt(X_sensor**2 + Y_sensor**2)
    THETA_ray = R_sensor * np.pi / 2
    PHI_ray = np.arctan2(Y_sensor, X_sensor)
    DY_ray = np.sin(THETA_ray) * np.sin(PHI_ray)
    DX_ray = np.sin(THETA_ray) * np.cos(PHI_ray)
    DZ_ray = np.cos(THETA_ray)

    #
    # Loop on all rays
    #
    data = []
    indices = []
    indptr = [0]
    for r, dy, dx, dz, r_dither in itertools.izip(
        R_sensor.reshape((-1, replicate)),
        DY_ray.reshape((-1, replicate)),
        DX_ray.reshape((-1, replicate)),
        DZ_ray.reshape((-1, replicate)),
        R_dither.ravel(),
        ):
        if np.all(r > 1):
            indptr.append(indptr[-1])
            continue

        #
        # Filter steps where r > 1
        #
        dy = dy[r<=1]
        dx = dx[r<=1]
        dz = dz[r<=1]

        #
        # Convert the ray samples to volume indices
        #
        Y_ray = (r_dither+R_samples) * dy
        X_ray = (r_dither+R_samples) * dx
        Z_ray = (r_dither+R_samples) * dz

        #
        # Calculate the atmosphere indices
        #
        Y_indices = np.searchsorted(Y, Y_ray.ravel())
        X_indices = np.searchsorted(X, X_ray.ravel())
        Z_indices = np.searchsorted(Z, Z_ray.ravel())

        Y_filter = (Y_indices > 0) * (Y_indices < Y.size)
        X_filter = (X_indices > 0) * (X_indices < X.size)
        Z_filter = (Z_indices > 0) * (Z_indices < Z.size)

        filtered = Y_filter*X_filter*Z_filter
        Y_indices = Y_indices[filtered]-1
        X_indices = X_indices[filtered]-1
        Z_indices = Z_indices[filtered]-1

        #
        # Calculate unique indices
        #
        inds_ray = (Y_indices*centered_grids.shape[1] + X_indices)*centered_grids.shape[2] + Z_indices
        uniq_indices, inv_indices = np.unique(inds_ray, return_inverse=True)

        #
        # Calculate weights
        # Note:
        # The weights are divided by the number of samples in the voxels, this gives the
        # averaged concentration in the voxel.
        #
        weights = []
        for i, ind in enumerate(uniq_indices):
            weights.append((inv_indices == i).sum() / R_samples.size / replicate)

        #
        # Sum up the indices and weights
        #
        data.append(weights)
        indices.append(uniq_indices)
        indptr.append(indptr[-1]+uniq_indices.size)

    #
    # Create sparse matrix
    #
    data = np.hstack(data)
    indices = np.hstack(indices)

    H = sps.csr_matrix(
        (data, indices, indptr),
        shape=(sensor_res[0]*sensor_res[1], centered_grids.size)
    )

    return BaseTransform(
        H=H,
        in_grids=in_grids,
        out_grids=out_grids,
        inv_grids=None
    )


def sensorTransformK(
    in_grids,
    cameraMatrix,
    distCoeffs,
    T,
    sensor_res,
    depth_res,
    samples_num=1000,
    dither_noise=10,
    replicate=10
    ):
    """
    Calculate Cartesian to polar transform.
    The camera is defined internally using the cameraMatrix and distCoeffs as returned from the opencv calibration functions.
    The camera is defined externally using the T transformation between camera coordinates and the outer world (atmosphere coords).
    NOTE:
    Currently it doesn't support fisheye cameras. This is because we don't check 
    """
    
    #
    # Input grids
    #
    Y, X, Z = in_grids.closed

    Y_sensor = np.arange(sensor_res[0])
    X_sensor = np.arange(sensor_res[1])

    #
    # Calculate sample steps along ray
    #
    R_max = np.max(np.sqrt(in_grids.expanded[0]**2 + in_grids.expanded[1]**2 + in_grids.expanded[2]**2))
    R_samples, R_step = np.linspace(0.0, R_max, samples_num, retstep=True)
    R_samples = R_samples[1:]
    R_dither = np.random.rand(*sensor_res) * R_step * dither_noise

    #
    # Calculate depth bins
    #
    #depth_bins = np.logspace(np.log10(R_samples[0]), np.log10(R_samples[-1]+R_step), depth_res+1)
    temp = np.linspace(0, 1, depth_res+1)
    temp = np.cumsum(temp)
    depth_bins = temp / temp[-1] * R_samples[-1]
    samples_bin = np.digitize(R_samples, depth_bins)
    samples_array = []
    for i in range(1, depth_res+1):
        samples_array.append(R_samples[samples_bin==i].reshape((-1, 1)))

    #
    # Create the output grids
    #
    out_grids = Grids(depth_bins[:-1], Y_sensor, X_sensor)

    #
    # Randomly replicate rays inside each pixel
    # step is the unit of pixels.
    #
    step = 1
    X_sensor = np.tile(X_sensor[:, :, np.newaxis], [1, 1, replicate])
    Y_sensor = np.tile(Y_sensor[:, :, np.newaxis], [1, 1, replicate])
    X_sensor += np.random.rand(*X_sensor.shape)*step
    Y_sensor += np.random.rand(*Y_sensor.shape)*step

    #
    # Calculate rays angles
    # R_sensor is the radius from the center of the image (0, 0) to the
    # pixel. It is used for calculating th ray direction (PHI, THETA)
    # and for filtering pixels outside the image (radius > 1).
    #
    cameraMatrix_inv = np.linalg.inv(cameraMatrix)
    temp = np.array((X_sensor.ravel(), Y_sensor.ravel(), np.ones(X_sensor.size)))
    DXY = np.dot(cameraMatrix_inv, temp)
    DX_ray = DXY[0, :].reshape(X_sensor.shape)
    DY_ray = DXY[1, :].reshape(X_sensor.shape)
    DZ_ray = np.sqrt(np.ones_like(DX_ray) - DX_ray**2 - DY_ray**2)

    #
    # Calculate inverse grid
    # TODO:
    # The inverse grids need to take into account the transformation of the camera.
    #
    R = out_grids[0]
    Y_ray = (r_dither+samples) * dy
    X_ray = (r_dither+samples) * dx
    Z_ray = (r_dither+samples) * dz
    temp = np.array((X_ray.ravel(), Y_ray.ravel(), Z_ray.ravel(), np.ones(X_ray.size)))
    XYZ_inv = np.dot(T, temp)
    X_inv = XYZ[0, :].reshape(X_ray.shape)
    Y_inv = XYZ[1, :].reshape(X_ray.shape)
    Z_inv = XYZ[2, :].reshape(X_ray.shape)
    inv_grids = Grids(Y_inv, X_inv, Z_inv)

    #
    # Loop on all rays
    #
    data = []
    indices = []
    indptr = [0]
    for samples in samples_array:
        for dy, dx, dz, r_dither in itertools.izip(
            DY_ray.reshape((-1, replicate)),
            DX_ray.reshape((-1, replicate)),
            DZ_ray.reshape((-1, replicate)),
            R_dither.ravel(),
            ):
            #
            # Calculate the samples in camera coords
            #
            Y_ray = (r_dither+samples) * dy
            X_ray = (r_dither+samples) * dx
            Z_ray = (r_dither+samples) * dz
            
            #
            # Calculate the samples in world coords
            #
            temp = np.array((X_ray.ravel(), Y_ray.ravel(), Z_ray.ravel(), np.ones(X_ray.size)))
            XYZ = np.dot(T, temp)
            X_world_ray = XYZ[0, :].reshape(X_ray.shape)
            Y_world_ray = XYZ[1, :].reshape(X_ray.shape)
            Z_world_ray = XYZ[2, :].reshape(X_ray.shape)
            
            #
            # Calculate the atmosphere indices
            #
            Y_indices = np.searchsorted(Y, Y_world_ray.ravel())
            X_indices = np.searchsorted(X, X_world_ray.ravel())
            Z_indices = np.searchsorted(Z, Z_world_ray.ravel())

            Y_filter = (Y_indices > 0) * (Y_indices < Y.size)
            X_filter = (X_indices > 0) * (X_indices < X.size)
            Z_filter = (Z_indices > 0) * (Z_indices < Z.size)

            filtered = Y_filter*X_filter*Z_filter
            Y_indices = Y_indices[filtered]-1
            X_indices = X_indices[filtered]-1
            Z_indices = Z_indices[filtered]-1

            #
            # Calculate unique indices
            #
            inds_ray = (Y_indices*in_grids.shape[1] + X_indices)*in_grids.shape[2] + Z_indices
            uniq_indices, inv_indices = np.unique(inds_ray, return_inverse=True)

            #
            # Calculate weights
            # Note:
            # The weights are divided by the number of samples in the voxels, this gives the
            # averaged concentration in the voxel.
            #
            weights = []
            for i, ind in enumerate(uniq_indices):
                weights.append((inv_indices == i).sum() / samples.size / replicate)

            #
            # Sum up the indices and weights
            #
            data.append(weights)
            indices.append(uniq_indices)
            indptr.append(indptr[-1]+uniq_indices.size)

    #
    # Create sparse matrix
    #
    data = np.hstack(data)
    indices = np.hstack(indices)

    H = sps.csr_matrix(
        (data, indices, indptr),
        shape=(sensor_res[0]*sensor_res[1]*depth_res, in_grids.size)
    )

    return BaseTransform(
        H=H,
        in_grids=in_grids,
        out_grids=out_grids,
        inv_grids=inv_grids
    )
