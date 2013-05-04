"""
"""

from __future__ import division
import numpy as np
import scipy.sparse as sps
from .base import *
from cytransforms import point2grids, direction2grids
import itertools

__all__ = ['directionTransform', 'integralTransform', 'sensorTransform', 'cumsumTransform']


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

    #
    # Center the grids
    #
    centered_grids = in_grids.translate(-np.array(sensor_center))
    Y, X, Z = centered_grids.closed

    #
    # Convert image pixels to ray direction
    # The image is assumed the [-1, 1]x[-1, 1] square.
    #
    Y_sensor, step = np.linspace(-1.0, 1.0, sensor_res, endpoint=False, retstep=True)
    X_sensor = np.linspace(-1.0, 1.0, sensor_res, endpoint=False)

    #
    # Calculate sample steps along ray
    #
    R_max = np.max(np.sqrt(centered_grids.expanded[0]**2 + centered_grids.expanded[1]**2 + centered_grids.expanded[2]**2))
    R_samples, R_step = np.linspace(0.0, R_max, samples_num, retstep=True)
    R_samples = R_samples[1:]
    R_dither = np.random.rand(sensor_res, sensor_res) * R_step * dither_noise

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
            # The weights are multiplied by the meter/sample ratio and averaged over the replicates.
            # This gives an averaged concentration per meter.
            #
            weights = []
            for i, ind in enumerate(uniq_indices):
                weights.append((inv_indices == i).sum() * R_max / samples_num / replicate)

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
        shape=(sensor_res*sensor_res*depth_res, centered_grids.size)
    )

    return BaseTransform(
        H=H,
        in_grids=in_grids,
        out_grids=out_grids,
        inv_grids=inv_grids
    )


def main():
    """Main doc """

    pass


if __name__ == '__main__':
    main()
