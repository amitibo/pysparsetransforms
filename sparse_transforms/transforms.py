"""
"""

from __future__ import division
import numpy as np
from .base import *
import itertools
import scipy.sparse as sps
from cytransforms import point2grids, direction2grids

__all__ = ['DirectionTransform', 'SensorTransform']


class DirectionTransform(BaseTransform):
    
    def __init__(
        self,
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
        
        super(DirectionTransform, self).__init__(
            H=H,
            in_grids=in_grids,
            out_grids=in_grids
        )


class SensorTransform(BaseTransform):
    
    def __init__(
        self,
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
        depth_bins = np.logspace(np.log10(R_samples[0]), np.log10(R_samples[-1]+R_step), depth_res+1)
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
        #
        X_sensor, Y_sensor = np.meshgrid(X_sensor, Y_sensor)
        X_sensor = np.tile(X_sensor[:, :, np.newaxis], [1, 1, replicate])
        Y_sensor = np.tile(Y_sensor[:, :, np.newaxis], [1, 1, replicate])
        X_sensor += np.random.rand(*X_sensor.shape)*step
        Y_sensor += np.random.rand(*Y_sensor.shape)*step
        
        #
        # Calculate rays angles
        # R_img is the radius from the center of the image (0, 0) to the
        # pixel. It is used for calculating th ray direction (PHI, THETA)
        # and for filtering pixels outside the image (radius > 1).
        #
        R_img = np.sqrt(X_sensor**2 + Y_sensor**2)
        THETA_ray = R_img * np.pi / 2
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
                R_img.reshape((-1, replicate)),
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
                #
                weights = []
                for i, ind in enumerate(uniq_indices):
                    weights.append((inv_indices == i).sum())
                
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

        super(SensorTransform, self).__init__(
            H=H,
            in_grids=in_grids,
            out_grids=out_grids
        )


def main():
    """Main doc """
    
    pass

    
if __name__ == '__main__':
    main()

    
    