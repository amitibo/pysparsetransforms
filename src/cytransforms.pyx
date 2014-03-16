"""
"""

import scipy.sparse as sps
import numpy as np
cimport numpy as np
import cython
from cpython cimport bool
from libc.math cimport sqrt
from .base import processGrids

DTYPEd = np.double
ctypedef np.double_t DTYPEd_t
DTYPEi32 = np.int32
ctypedef np.int32_t DTYPEi32_t
DTYPEi = np.int
ctypedef np.int_t DTYPEi_t

DEF eps = 1e-10

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline np.intp_t local_argsearch_left(double [:] grid, double key):
    """
    Find the index in the grid where a point resides.
    The calculation uses a mid point search.
    The algorithm returns the index left to the point. Will return 0 if below lowest
    grid index.
    
    Parameters:
    -----------
    grid : array
         Closed ended 1D grid.
    key : double
         The coordinate of the point.
    """
    
    cdef np.intp_t imin = 0
    cdef np.intp_t imax = grid.size
    cdef np.intp_t imid
    
    while imin < imax:
        imid = imin + ((imax - imin) >> 1)
        
        if grid[imid] < key:
            imin = imid + 1
        else:
            imax = imid

    return imin

    
def test_local_argsearch_left(grid, key):
    """
    Allow for testing local_argsearch_left
    """
    
    return local_argsearch_left(grid, key)
    
    
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline bool interpolatePoints(
    double[:] grid,
    np.intp_t i0,
    np.intp_t i1,
    np.intp_t si,
    np.intp_t grid_dim,
    double[:] p0,
    double[:] d,
    int[:, ::1] I,
    double[:, ::1] P
    ):
    """
    Interpolate coordinates of the intersections points
    between a line and a grid.
    This function should be run on each dimension separately.
    
    Parameters:
    -----------
    grid:
         Closed ended 1D grid. The function should be run separately for each dimension.
    i0, i1:
         First and last intersection indices along the grid.
    si: int
        The starting index (in output arrays I and P) where to place the coordinates of the intersection.
    grid_dim: int
        Dimension index of the grid
    p0 : 3 tuple
        coordinates of the starting point
    d : 3 tuple
        distance along each dimension between starting and ending point.
    I: array 3xPointNum [output]
        This array marks the advances of the grid indices. It adds up only to the current dimension.
    P: array 3xPointNum [output]
        This array marks the coordinates of the intersection point.
        
    Retruns:
    --------
    True if any points were interpolated, False other wise.
    """
    
    cdef np.intp_t i, dim, k
    cdef np.intp_t dt
    
    if i0 == i1:
        return False

    if i0 > i1:
        i0 -= 1
        i1 -= 1
        dt = -1
    else:
        dt = 1
    
    #
    # Loop on all dimensions
    #
    for dim in xrange(3):
        #
        # Loop on all the intersections of a dimension
        #
        i = si
        for k in xrange(i0, i1, dt):
            #
            # Calculate the indices of the points
            #
            if dim == grid_dim:
                I[dim, i] = dt

            #
            # Interpolate the value at the point
            #
            P[dim, i] = d[dim]/d[grid_dim] * (grid[k]-p0[grid_dim]) + p0[dim]

            i += 1

    return True


def test_interpolatePoints(grid, i0, i1, si, grid_dim, p0, d, I, P):
    """
    Allow for testing interpolatePoints
    """
    
    return interpolatePoints(grid, i0, i1, si, grid_dim, p0, d, I, P)
    

@cython.boundscheck(False)
@cython.wraparound(False)
cdef calcCrossings(
    double[:] Y,
    double[:] X,
    double[:] Z,
    double[:] p0,
    double[:] p1
    ):
    """
    Calculate corssing points of a line between two points
    and a grid.
    
    Parameters:
    -----------
    Y, X, Z : arrays
        Closed ended open (not expanded) grids.
    p0, p1 : 3 tuple
        Coordinates of the p0 and p1 point. Note that the points are supposed to reside
        inside the grid.
    """
    
    #
    # Collect the inter indices (grid crossings)
    #
    cdef np.intp_t i, j, k, sort_index
    cdef np.intp_t points_num
    cdef np.intp_t x_i0, x_i1, y_i0, y_i1, z_i0, z_i1
    cdef np.intp_t dimx = X.size - 1
    cdef np.intp_t dimz = Z.size - 1
    cdef double tmpd
    cdef int tmpi

    #
    # Calculate the indices of the voxel where the
    # point resides (note that the indices start at 1)
    #
    y_i0 = local_argsearch_left(Y, p0[0])
    y_i1 = local_argsearch_left(Y, p1[0])
    x_i0 = local_argsearch_left(X, p0[1])
    x_i1 = local_argsearch_left(X, p1[1])
    z_i0 = local_argsearch_left(Z, p0[2])
    z_i1 = local_argsearch_left(Z, p1[2])
    
    #
    # Calculate the number of inter points (grid crossings)
    #
    cdef double[:] d = np.empty(3)
    d[0] = p1[0] - p0[0]
    d[1] = p1[1] - p0[1]
    d[2] = p1[2] - p0[2]
    points_num = abs(y_i1 - y_i0) + abs(x_i1 - x_i0) + abs(z_i1 - z_i0)

    #
    # Note:
    # The np_r, np_indices arrays are declared separately
    # as these are values that are returned from the function.
    #
    np_r = np.empty(points_num+1)
    np_indices = np.empty(points_num+1, dtype=DTYPEi32)
    cdef double[:] r = np_r
    cdef int[:] indices = np_indices
    
    #
    # Check whether the start and end points are in the same voxel
    #
    if points_num == 0:
        tmpd = 0
        for i in range(3):
            tmpd += d[i]**2
        r[0] = sqrt(tmpd)
        indices[0] = dimz*(dimx*(y_i0-1) + x_i0-1) + z_i0-1
        return np_r, np_indices
    
    np_I = np.zeros((3, points_num), dtype=DTYPEi32)
    np_P = np.empty((3, points_num))
    cdef int[:, ::1] I = np_I
    cdef double[:, ::1] P = np_P
    
    if interpolatePoints(Y, y_i0, y_i1, 0, 0, p0, d, I, P):
        sort_index = 0
    if interpolatePoints(X, x_i0, x_i1, abs(y_i1 - y_i0), 1, p0, d, I, P):
        sort_index = 1
    if interpolatePoints(Z, z_i0, z_i1, abs(y_i1 - y_i0)+abs(x_i1 - x_i0), 2, p0, d, I, P):
        sort_index = 2

    #
    # Sort points according to their spatial order
    #
    np_order = np.argsort(P[sort_index, :]).astype(DTYPEi32)
    cdef int[:] order = np_order

    np_SI = np.empty((3, points_num+2), dtype=DTYPEi32)
    np_SP = np.empty((3, points_num+2))
    cdef int[:, ::1] SI = np_SI
    cdef double[:, ::1] SP = np_SP
    
    SI[0, 0] = max(y_i0-1, 0)
    SI[1, 0] = max(x_i0-1, 0)
    SI[2, 0] = max(z_i0-1, 0)
    SI[0, points_num+1] = max(y_i1-1, 0)
    SI[1, points_num+1] = max(x_i1-1, 0)
    SI[2, points_num+1] = max(z_i1-1, 0)
    SP[0, 0] = p0[0]
    SP[1, 0] = p0[1]
    SP[2, 0] = p0[2]
    SP[0, points_num+1] = p1[0]
    SP[1, points_num+1] = p1[1]
    SP[2, points_num+1] = p1[2]
    
    if p0[sort_index] > p1[sort_index]:
        for i in range(3):        
            for j in range(points_num):
                 SI[i, j+1] = SI[i, j] + I[i, order[points_num-1-j]]
                 SP[i, j+1] = P[i, order[points_num-1-j]]
    else:
        for i in range(3):        
            for j in range(points_num):
                 SI[i, j+1] = SI[i, j] + I[i, order[j]]
                 SP[i, j+1] = P[i, order[j]]

    #
    # Calculate path segments length
    #
    for j in range(points_num+1):
        tmpd = 0
        for i in range(3):
            tmpd += (SP[i, j+1] - SP[i, j])**2
        r[j] = sqrt(tmpd)
        indices[j] = dimz*(dimx*SI[0, j] + SI[1, j]) + SI[2, j]
    
    #
    # Order the indices
    #
    if indices[0] > indices[points_num]:
        for j in range(points_num+1):
            tmpd = r[j]
            r[j] = r[points_num-j]
            r[points_num-j] = tmpd
            tmpi = indices[j]
            indices[j] = indices[points_num-j]
            indices[points_num-j] = tmpi

    return np_r, np_indices


def limitDGrids(DGrid, Grid, lower_limit, upper_limit):
    """
    Check wheter the addition of some DGrid to Grid
    will cause it to cross a lower_ or upper_ limit.
    
    Retruns:
    ratio - ratio of the crossing
    L - indices that cross (when incremented by DGrid.
    """
    
    ratio = np.ones_like(DGrid)
    
    Ll = (Grid + DGrid) < lower_limit
    if np.any(Ll):
        ratio[Ll] = (lower_limit - Grid[Ll]) / DGrid[Ll]
        
    Lh = (Grid + DGrid) > upper_limit
    if np.any(Lh):
        ratio[Lh] = (upper_limit - Grid[Lh]) / DGrid[Lh]
    
    return ratio, Ll + Lh


@cython.boundscheck(False)
def point2grids(point, Y, X, Z):
    """
    Calculate the contribution of each voxel to the path from a point to each voxel.
    NOTE the current implementation assumes that the point is directly below the grid!
    
    Parameters:
    -----------
    point: 3 tuple
        y, x, z coords of the point. NOTE the current implementation assumes that the point is directly below the grid!
    Y, X, Z: expanded grids
    
    """
    
    #
    # Calculate open and centered grids
    #
    (Y, X, Z), (Y_open, X_open, Z_open) = processGrids((Y, X, Z))

    cdef DTYPEd_t [:] p_Y = Y.ravel()
    cdef DTYPEd_t [:] p_X = X.ravel()
    cdef DTYPEd_t [:] p_Z = Z.ravel()

    p1 = np.array(point, order='C').ravel()
    np_p2 = np.empty(3)    
    cdef double[:] p2 = np_p2
    np_px = np.empty(3)    
    cdef double[:] px = np_px
    
    #
    # Calculate the intersection with the BOA (Bottom Of Atmosphere)
    #
    boa = np.min(Z_open)
    DZ = Z - boa
    ratio = DZ / (Z-p1[2])
    DX = (X-p1[1]) * ratio
    DY = (Y-p1[0]) * ratio
    
    cdef DTYPEd_t [:] p_DY = DY.ravel()
    cdef DTYPEd_t [:] p_DX = DX.ravel()
    cdef DTYPEd_t [:] p_DZ = DZ.ravel()

    data = []
    indices = []
    indptr = [0]
    cdef int grid_size = Y.size
    cdef int i = 0
    for i in xrange(grid_size):
        #
        # Process next voxel
        #
        p2[0] = p_Y[i]
        p2[1] = p_X[i]
        p2[2] = p_Z[i]

        #
        # Intersection of the ray with the TOA
        #
        px[0] = p_Y[i] - p_DY[i]
        px[1] = p_X[i] - p_DX[i]
        px[2] = boa
        
        #
        # Calculate crossings for line between p1 and p2
        #
        r, ind = calcCrossings(Y_open, X_open, Z_open, px, p2)

        #
        # Accomulate the crossings for the sparse matrix
        # Note:
        # I remove values lower than some epsilon value.
        # This way I filter out numerical inacurracies and
        # negative values.
        #
        zr = r > eps
        r = r[zr]
        data.append(r)
        indices.append(ind[zr])
        indptr.append(indptr[-1]+r.size)

    #
    # Form the sparse transform matrix
    #
    data = np.hstack(data)
    indices = np.hstack(indices)
    
    H_dist = sps.csr_matrix(
        (data, indices, indptr),
        shape=(Y.size, Y.size)
    )
    
    return H_dist
    

@cython.boundscheck(False)
def direction2grids(phi, theta, Y, X, Z):
    
    #
    # Calculate open and centered grids
    #
    (Y, X, Z), (Y_open, X_open, Z_open) = processGrids((Y, X, Z))

    cdef DTYPEd_t [:] p_Y = Y.ravel()
    cdef DTYPEd_t [:] p_X = X.ravel()
    cdef DTYPEd_t [:] p_Z = Z.ravel()

    #
    # Calculate the intersection with the TOA (Top Of Atmosphere)
    #
    toa = np.max(Z_open)
    DZ = toa - Z
    DX = DZ * np.cos(phi) * np.tan(theta)
    DY = DZ * np.sin(phi) * np.tan(theta)

    #
    # Check crossing with any of the sides
    #
    ratio, L = limitDGrids(DY, Y, np.min(Y_open), np.max(Y_open))
    if np.any(L):
        DY[L] *= ratio[L]
        DX[L] *= ratio[L]
        DZ[L] *= ratio[L]
    ratio, L = limitDGrids(DX, X, np.min(X_open), np.max(X_open))
    if np.any(L):
        DY[L] *= ratio[L]
        DX[L] *= ratio[L]
        DZ[L] *= ratio[L]
    ratio, L = limitDGrids(DZ, Z, np.min(Z_open), np.max(Z_open))
    if np.any(L):
        DY[L] *= ratio[L]
        DX[L] *= ratio[L]
        DZ[L] *= ratio[L]

    cdef DTYPEd_t [:] p_DY = DY.ravel()
    cdef DTYPEd_t [:] p_DX = DX.ravel()
    cdef DTYPEd_t [:] p_DZ = DZ.ravel()

    cdef double[:] p1 = np.empty(3)
    cdef double[:] p2 = np.empty(3)
    
    data = []
    indices = []
    indptr = [0]
    cdef int grid_size = Y.size
    cdef int i = 0
    for i in xrange(grid_size):
        #
        # Center of each voxel
        #
        p1[0] = p_Y[i]
        p1[1] = p_X[i]
        p1[2] = p_Z[i]
        
        #
        # Intersection of the ray with the TOA
        #
        p2[0] = p1[0] + p_DY[i]
        p2[1] = p1[1] + p_DX[i]
        p2[2] = p1[2] + p_DZ[i]
        
        #
        # Calculate crossings for line between p1 and p2
        #
        r, ind = calcCrossings(Y_open, X_open, Z_open, p1, p2)
        
        #
        # Accomulate the crossings for the sparse matrix
        # Note:
        # I remove values lower than some epsilon value.
        # This way I filter out numerical inacurracies and
        # negative values.
        #
        zr = r > eps
        r = r[zr]
        data.append(r)
        indices.append(ind[zr])
        indptr.append(indptr[-1]+r.size)

    #
    # Form the sparse transform matrix
    #
    data = np.hstack(data)
    indices = np.hstack(indices)
    
    H_dist = sps.csr_matrix(
        (data, indices, indptr),
        shape=(Y.size, Y.size)
    )
    
    return H_dist
