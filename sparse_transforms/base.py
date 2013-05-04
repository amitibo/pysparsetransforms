"""
"""

from __future__ import division
import numpy as np
import scipy.sparse as sps
import copy
import types
from .transformation_matrices import euler_matrix
import os
import pickle
import scipy.io as sio

__all__ = ['BaseTransform', 'Grids', 'calcTransformMatrix', 'loadTransform']


class Grids(object):
    """
    A class that eases the use of grids. This class is meant for use
    with open grids.
    """
    
    def __init__(self, *grids):
        
        open_grids = []
        self._ndim = len(grids)
        
        for i, grid in enumerate(grids):
            array = np.array(grid).copy()
            
            if array.ndim == 1:
                #
                # 1D arrays (like the arrays created by linspace) are reshaped to ndim arrays.
                #
                inds = [1] * self._ndim
                inds[i] = -1
                array.shape = inds
                
            open_grids.append(array)
        
        self._grids = open_grids
        
    def __getitem__(self, key):
        
        if not isinstance(key, int):
            raise TypeError('list indices must be integers, not str')
        
        return self._grids[key]

    def __iter__(self):
        return iter(self._grids)
    
    def __len__(self):
        return self._ndim
    
    def copy(self):
        return copy.deepcopy(self)
    
    def save(self, path):
        """Save a copy of the grid"""
        
        np.savez(path, *self._grids)

    @classmethod
    def load(cls, path):
        """Load a copy of the grid"""
        
        path = os.path.abspath(path)
        
        base_path, ext = os.path.splitext(path)
        if ext == '':
            path = base_path + '.npz'
            
        l = np.load(path)
        grids = [l[key] for key in sorted(l.files)]
    
        return cls(*grids)

    @property
    def shape(self):
        return tuple([grid.shape[i] for i, grid in enumerate(self._grids)])
    
    @property
    def size(self):
        return np.prod(self.shape)

    @property
    def ndim(self):
        return self._ndim
    
    @property
    def expanded(self):
        """Expand the open grids to full grids"""
        
        if self.isExpanded:
            return self._grids
        
        expanded_grids = []
        for i, grid in enumerate(self._grids):
            tiles = list(self.shape)
            tiles[i] = 1
            expanded_grids.append(np.tile(grid, tiles))
        
        return expanded_grids
    
    @property
    def isExpanded(self):
        """Check if the grids are expanded"""
        
        return self.shape == self._grids[0].shape
    
    @property
    def closed(self):
        """Retrun a list of closed grids (i.e. include the upper border of the grid)"""
        
        if self.isExpanded:
            raise NotImplementedError('Not implemented for expanded grids')
        
        closed_grids = []
        for grid in self._grids:
            closed_grid = np.hstack((grid.ravel(), 2*grid.ravel()[-1]-grid.ravel()[-2]))
            closed_grids.append(np.ascontiguousarray(closed_grid))

        return closed_grids

    @property
    def derivatives(self):
        """
        Calculate first order partial derivatives. Returns a list of 1D arrays.
        """
    
        derivatives = []
        for i, grid in enumerate(self._grids):
            inds1 = [slice(None)] * self.ndim
            inds1[i] = slice(1, None)
            inds2 = [slice(None)] * self.ndim
            inds2[i] = slice(None, -1)
            
            derivative = np.abs(grid[inds1] - grid[inds2])
            derivative = self._duplicateLastValue(derivative, axis=i)
            derivatives.append(np.ascontiguousarray(derivative))
    
        return derivatives

    def _duplicateLastValue(self, grid, axis):
        """Duplicate the last value of a grid in direction detrmined by axis."""
        
        inds = [slice(None)] * self.ndim
        inds[axis] = -1
        new_shape = list(grid.shape)
        new_shape[axis] = -1
        
        return np.concatenate((grid, grid[inds].reshape(new_shape)), axis=axis)
        
    def translate(self, translation):
        """
        Return a Grid object which is a translation of the original grid.
        """
        translation = np.array(translation)
        assert translation.size == self.ndim, 'translation must have the same dimension of grids'
        
        translated_grids = []
        for grid, delta in zip(self._grids, translation):
            translated_grids.append(grid + delta)

        return Grids(*translated_grids)
            
    def rotate(self, ai, aj, ak):
        """
        Return a Grid object which is a rotation of the original grid.
        Rotation is given in euler angles (axis used is 'sxyz')
        """
        
        assert self.ndim == 3, 'Rotation supports only 3D grids'
        
        if not self.isExpanded:
            self = Grids(self.expanded)
        
        H_rot = euler_matrix(ai, aj, ak)
        
        XYZ = np.vstack([grid.ravel() for grid in self._grids] + [np.ones(self.size)])
        XYZ_rotated = np.dot(H_rot, XYZ)
    
        rotated_grids = []
        for i in range(self.ndim):
            rotated_grids.append(XYZ_rotated[i, :].reshape(self.shape))
            
        return Grids(*rotated_grids)

  
def patchUnaryMethods(cls, method_name):
    """ Add unary methods to a class."""
    
    def wrapper(self):
        #
        # Call the underlying matrix method
        #
        H = getattr(self.H, method_name)()
        
        obj = copy.copy(self)
            
        obj.H = H
        
        return obj
    
    setattr(cls, method_name, wrapper)
    

def patchNumericMethods(cls, method_name, inplace=False):
    """Add methods (__add__ etc) to a class."""
    
    def wrapper(self, other):
        
        if isinstance(other, BaseTransform):
            other = other.H
        
        #
        # Call the underlying matrix method
        #
        H = getattr(self.H, method_name)(other)
        
        if inplace:
            #
            # Change the object inplace
            #
            obj = self
        else:
            #
            # Return a copy
            #
            obj = copy.copy(self)
            
        obj.H = H
        
        return obj
    
    setattr(cls, method_name, wrapper)


def patchMulMethods(cls, method_name):
    """Add method __mul__ to a class."""
    
    def wrapper(self, other):
        
        other_is_transform = False
        other_reshaped = False
        
        if isinstance(other, BaseTransform):
            other_is_transform = True
            other = other.H
        elif not sps.issparse(other):
            if isinstance(other, np.ndarray) and (other.ndim != 2 or other.shape[1] != 1):
                other_reshaped = True
                other = other.reshape((-1, 1))
        
        #
        # Call the underlying matrix method
        #
        H = getattr(self.H, method_name)(other)
        
        if other_is_transform:
            #
            # Return a copy
            #
            obj = copy.copy(self)
            
            obj.H = H
        else:
            #
            # Return the result of the underlying matrix
            #
            obj = H
            
            if other_reshaped:
                obj = obj.reshape(self.out_grids.shape)
            
        return obj
    
    setattr(cls, method_name, wrapper)


class BaseTransform(object):
    """
    Base class for transforms

    Attributes
    ----------
    H : sparse matrix.
        The underlying sparse matrix.
    shape : (int, int)
        The shape of the transform.
    in_grids : Grids object
        The set of grids over which the input is defined.
    out_grids : Grids object
        The set of grids over which the output is defined.
    inv_grids : Grids object
        The out_grids projected back into the in_grids.
    T : type(self)
        The transpose of the transform.
        
    Methods
    -------
    """

    def __new__(cls, *args, **kwargs):

        #
        # Delegate numeric methods to the encapsulated
        # sparse matrix. The reason to do it in the __new__
        # method is that python looks for this methods (__add__ etc)
        # in the class and not the object. So it is not possible
        # to overwrite __getattr__ for these methods.
        #
        NUMERIC_METHODS = ('add', 'sub')
        UNARY_METHODS = ('__neg__', '__pos__')
        binary_methods = ['__{method}__'.format(method=method) for method in NUMERIC_METHODS]
        binary_methods += ['__r{method}__'.format(method=method) for method in NUMERIC_METHODS]
        augmented_methods = ['__i{method}__'.format(method=method) for method in NUMERIC_METHODS]
        
        for method_name in binary_methods: 
            patchNumericMethods(cls, method_name, inplace=False)
            
        for method_name in augmented_methods:
            patchNumericMethods(cls, method_name, inplace=True)
        
        for method_name in ('__mul__', '__rmul__'): 
            patchMulMethods(cls, method_name)
            
        for method_name in UNARY_METHODS: 
            patchUnaryMethods(cls, method_name)
            
        return object.__new__(cls, *args, **kwargs)

        
    def __init__(self, H, in_grids=None, out_grids=None, inv_grids=None):
        """
        Parameters
        ----------
        H : sparse matrix
            The matrix the represents the transform.
        in_grids : tuple of open grids
            The set of grids over which the input is defined.
        out_grids : tuple of open grids
            The set of grids over which the output is defined.
        inv_grids : Grids object
            The out_grids projected back into the in_grids.
        """
        
        self.H = H
        self.in_grids = in_grids
        self.out_grids = out_grids
        self.inv_grids = inv_grids
    
    def save(self, path):
        """Save a copy of the transform"""
        
        path = os.path.abspath(path)
        
        base_path, ext = os.path.splitext(path)
        if ext == '':
            ext = '.pkl'
        
        H_path = base_path + '_H.mat'
        if self.in_grids:
            in_path = base_path + '_in'
            self.in_grids.save(in_path)
        else:
            in_path = None
            
        if self.out_grids:
            out_path = base_path + '_out'
            self.out_grids.save(out_path)
        else:
            out_path = None
            
        if self.inv_grids:
            inv_path = base_path + '_inv'
            self.inv_grids.save(inv_path)
        else:
            inv_path = None
        
        sio.savemat(
            H_path,
            {
                'H': self.H,
            },
            do_compression=True
        )
            
        paths = {
            'H': H_path,
            'in': in_path,
            'out': out_path,
            'inv': inv_path
            }
        
        with open(base_path+ext, 'w') as f:
            pickle.dump(
                paths,
                f
            )
    
    @staticmethod
    def load(path):
        
        path = os.path.abspath(path)
        
        base_path, ext = os.path.splitext(path)
        if ext == '':
            ext = '.pkl'
        
        with open(base_path+ext, 'r') as f:
            paths = pickle.load(f)
        
        H = sio.loadmat(paths['H'])['H']
        if paths['in']:
            in_grids = Grids.load(paths['in'])
        else:
            in_grids = None
        if paths['out']:
            out_grids = Grids.load(paths['out'])
        else:
            out_grids = None
        if paths['inv']:
            inv_grids = Grids.load(paths['inv'])
        else:
            inv_grids = None
            
        return BaseTransform(H, in_grids=in_grids, out_grids=out_grids, inv_grids=inv_grids)
        
    @property
    def shape(self):
        """The shape of the transform.
        """
        return self.H.shape
        
    @property
    def T(self):
        """The transpose of the transform.
        """
        import copy

        new_obj = self.__class__(self.H.T, in_grids=self.out_grids, out_grids=self.in_grids)

        return new_obj


def loadTransform(path):
    
    return BaseTransform.load(path)

    
def spdiag(X):
    """
    Return a sparse diagonal matrix. The elements of the diagonal are made of 
    the elements of the vector X.

    Parameters
    ----------
    X : array
        1D array to be placed on the diagonal.
        
    Returns
    -------
    H : sparse matrix
        Sparse diagonal matrix, in dia format.
"""

    import scipy.sparse as sps

    return sps.dia_matrix((X.ravel(), 0), (X.size, X.size)).tocsr()


def processGrids(grids):
    """Calculate open grids and centered grids"""

    open_grids = []
    centered_grids = []
    
    for dim, grid in enumerate(grids):
        sli = [0] * len(grid.shape)
        sli[dim] = Ellipsis
        open_grid = grid[sli].ravel()
        open_grid = np.hstack((open_grid, 2*open_grid[-1]-open_grid[-2]))
        open_grids.append(np.ascontiguousarray(open_grid))
        
        vec_shape = [1] * len(grid.shape)
        vec_shape[dim] = -1
        centered_grid = grid + (open_grid[1:] - open_grid[:-1]).reshape(vec_shape) / 2
        centered_grids.append(np.ascontiguousarray(centered_grid))
        
    return centered_grids, open_grids


def count_unique(keys, dists):
    """count frequency of unique values in array. non positive values are ignored."""
    
    nnzs = keys>0
    filtered_keys = keys[nnzs]
    filtered_dists = dists[nnzs]
    if filtered_keys.size == 0:
        return filtered_keys, filtered_dists
    
    uniq_keys, inv_indices = np.unique(filtered_keys, return_inverse=True)
    uniq_dists = np.empty(uniq_keys.size)
    
    for i in range(uniq_keys.size):
        d = filtered_dists[inv_indices == i]
        uniq_dists[i] = d.sum()
        
    return uniq_keys, uniq_dists


def calcTransformMatrix(src_grids, dst_coords):
    """
    Calculate a sparse transformation matrix. The transform
    is represented as a mapping from the src_coords to the dst_coords.
    
    Parameters
    ----------
    src_grids : list of arrays
        Array of source grids.
        
    dst_coords : list of arrays
        Array of destination grids as points in the source grids.
        
    Returns
    -------
    H : parse matrix
        Sparse matrix, in csr format, representing the transform.
"""
    
    import numpy as np
    import scipy.sparse as sps
    import itertools

    #
    # Shape of grid
    #
    src_shape = src_grids[0].shape
    src_size = np.prod(np.array(src_shape))
    dst_shape = dst_coords[0].shape
    dst_size = np.prod(np.array(dst_shape))
    dims = len(src_shape)
    
    #
    # Calculate grid indices of coords.
    #
    indices, src_grids_slim = coords2Indices(src_grids, dst_coords)

    #
    # Filter out coords outside of the grids.
    #
    nnz = np.ones(indices[0].shape, dtype=np.bool_)
    for ind, dim in zip(indices, src_shape):
        nnz *= (ind > 0) * (ind < dim)

    dst_indices = np.arange(dst_size)[nnz]
    nnz_indices = []
    nnz_coords = []
    for ind, coord in zip(indices, dst_coords):
        nnz_indices.append(ind[nnz])
        nnz_coords.append(coord.ravel()[nnz])
    
    #
    # Calculate the transform matrix.
    #
    diffs = []
    indices = []
    for grid, coord, ind in zip(src_grids_slim, nnz_coords, nnz_indices):
        diffs.append([grid[ind] - coord, coord - grid[ind-1]])
        indices.append([ind-1, ind])

    diffs = np.array(diffs)
    diffs /= np.sum(diffs, axis=1).reshape((dims, 1, -1))
    indices = np.array(indices)

    dims_range = np.arange(dims)
    strides = np.array(src_grids[0].strides).reshape((-1, 1))
    strides /= strides[-1]
    I, J, VALUES = [], [], []
    for sli in itertools.product(*[[0, 1]]*dims):
        i = np.array(sli)
        c = indices[dims_range, sli, Ellipsis]
        v = diffs[dims_range, sli, Ellipsis]
        I.append(dst_indices)
        J.append(np.sum(c*strides, axis=0))
        VALUES.append(np.prod(v, axis=0))
        
    H = sps.coo_matrix(
        (np.array(VALUES).ravel(), np.array((np.array(I).ravel(), np.array(J).ravel()))),
        shape=(dst_size, src_size)
        ).tocsr()

    return H


def coords2Indices(grids, coords):
    """
    """

    import numpy as np

    inds = []
    slim_grids = []
    for dim, (grid, coord) in enumerate(zip(grids, coords)):
        sli = [0] * len(grid.shape)
        sli[dim] = Ellipsis
        grid = grid[sli]
        slim_grids.append(grid)
        inds.append(np.searchsorted(grid, coord.ravel()))

    return inds, slim_grids


def main():
    """Main doc """
    
    pass


    
if __name__ == '__main__':
    main()

    
    