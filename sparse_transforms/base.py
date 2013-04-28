"""
"""

from __future__ import division
import numpy as np
import copy

__all__ = ['TransformBase', 'Grids']


class Grids(object):
    
    def __init__(self, *grids):
        
        open_grids = []
        self._ndim = len(grids)
        
        for i, grid in enumerate(grids):
            inds = [1] * self._ndim
            inds[i] = -1
            open_grids.append(np.array(grid).copy().reshape(inds))
        
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
    
    @property
    def shape(self):
        return tuple([grid.size for grid in self._grids])
    
    @property
    def size(self):
        return np.prod(self.shape)

    @property
    def ndim(self):
        return self._ndim
    
    def translate(self, translation):
        translation = np.array(translation)
        assert translation.size == self.ndim, 'translation must have the same dimension of grids'
        
        translated_grids = []
        for grid, delta in zip(self._grids, translation):
            translated_grids.append(grid + delta)

        return Grids(*translated_grids)
            
    def rotate(self, rotation):
        raise NotImplemented('Not implemented yet')


class TransformBase(object):
    """
    Base class for transforms

    Attributes
    ----------
    name : string
        Name of transform.
    shape : (int, int)
        The shape of the transform.
    in_grids : tuple of open grids
        The set of grids over which the input is defined.
    out_grids : tuple of open grids
        The set of grids over which the output is defined.
    T : type(self)
        The transpose of the transform.
        
    Methods
    -------
    """

    def __init__(self, name, H, in_grids=None, out_grids=None):
        """
        Parameters
        ----------
        name : string
            Name of transform.
        H : sparse matrix
            The matrix the represents the transform.
        in_signal_shape : tuple of integers, optional (default=None)
            The shape of the input signal. The product of `in_signal_shape` should
            be equal to `shape[1]`. If `None`, then it is set to (shape[1], 1).
        out_signal_shape : tuple of ints
            The shape of the output signal. The product of `out_signal_shape` should
            be equal to `shape[1]`. If `in_signal_shape=None`, then it is set to
            (shape[0], 1). If `out_signal_shape=None` and `shape[0]=shape[1]` then
            `out_signal_shape=in_signal_shape`.
        """
        
        #if in_signal_shape==None:
            #in_signal_shape = (H.shape[1], 1)
            #out_signal_shape = (H.shape[0], 1)            
        #elif out_signal_shape==None:
            #if H.shape[0]==H.shape[1]:
                #out_signal_shape = in_signal_shape
            #else:
                #out_signal_shape = (H.shape[0], 1)
            
        #assert np.prod(in_signal_shape)==shape[1], 'Input signal shape does not conform to the shape of the transform'
        #assert np.prod(out_signal_shape)==shape[0], 'Output signal shape does not conform to the shape of the transform'
            
        self._name = name
        self._H = H
        self._in_grids = in_grids
        self._out_grids = out_grids
        self._conj = False
            
    @property
    def name(self):
        """Name of transform.
        """
        return self._name
        
    @property
    def shape(self):
        """The shape of the transform.
        """
        if self._conj:
            return self._H.shape[::-1]
        else:
            return self._H.shape
        
    @property
    def in_shape(self):
        """The shape of the input signal for the transform.
        """
        if self._conj:
            return self._out_grids.shape
        else:
            return self._in_grids.shape
    
    @property
    def out_shape(self):
        """The shape of the output signal for the transform.
        """
        if self._conj:
            return self._in_grids.shape
        else:
            return self._out_grids.shape
        
    @property
    def T(self):
        """The transpose of the transform.
        """
        import copy

        new_copy = copy.copy(self)
        new_copy._conj = True
        return new_copy

    def _checkDimensions(self, x):
        """Check that the size of the input signal is correct.
        This function is called by the `__call__` method.
        
        Parameters
        ==========
        x : array
            Input signal in columnstack order.
        """

        if x.shape == (1, 1) and self._shape != (1, 1):
            raise Exception('transform-scalar multiplication not yet supported')

        if x.shape[0] != self.shape[1]:
            raise Exception('Incompatible dimensions')

        if x.shape[1] != 1:
            raise Exception('transform-matrix multiplication not yet supported')
    
    def _apply(self, x):
        """Apply the transform on the input signal. Should be overwritten by the transform.
        This function is called by the `__call__` method.
        
        Parameters
        ==========
        x : array
            Input signal in columnstack order.
        """
        
        if self._conj:
            y = np.dot(self._H.T, x)
        else:
            y = np.dot(self._H, x)

        return y
        
    def __call__(self, x):
        
        x = x.reshape((-1, 1))
        
        self._checkDimensions(x)

        return self._apply(x).reshape(self._out_grids.shape)
    
    
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


def limitDGrids(DGrid, Grid, lower_limit, upper_limit):
    ratio = np.ones_like(DGrid)
    
    Ll = (Grid + DGrid) < lower_limit
    if np.any(Ll):
        ratio[Ll] = (lower_limit - Grid[Ll]) / DGrid[Ll]
        
    Lh = (Grid + DGrid) > upper_limit
    if np.any(Lh):
        ratio[Lh] = (upper_limit - Grid[Lh]) / DGrid[Lh]
    
    return ratio, Ll + Lh


def main():
    """Main doc """
    
    pass


    
if __name__ == '__main__':
    main()

    
    