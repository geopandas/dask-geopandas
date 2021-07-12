import numpy as np
from numba import jit
ngjit = jit(nopython=True, nogil=True)

# Based on: https://github.com/holoviz/spatialpandas/blob/
# 9252a7aba5f8bc7a435fffa2c31018af8d92942c/spatialpandas/dask.py

@ngjit
def _hilbert_distance(gdf, total_bounds, p):

    """
    Calculate the hilbert distance for a GeoDataFrame based on the mid-point of
    the bounds for each geom and total bounds of the collection of geoms

    Parameters
    ----------
    gdf : GeoDataFrame

    total_bounds : Total bounds of GeoDataFrame

    p : Hilbert curve parameter

    Returns
    ---------
    Array of hilbert distances for each geom
    """

    if total_bounds is None:
        total_bounds = gdf.total_bounds

    # Calculate bounds of each geom
    bounds = gdf.bounds.to_numpy()

    # Hilbert Side len
    side_length = 2 ** p

    # Calculate x and y range of total bound coords - returns array
    geom_ranges = [
        (total_bounds[0], total_bounds[2]),
        (total_bounds[1], total_bounds[3]),
    ]
    # Calculate mid points for x and y bound coords - returns array
    geom_mids = [
        ((bounds[:, 0] + bounds[:, 2]) / 2.0),
        ((bounds[:, 1] + bounds[:, 3]) / 2.0),
    ]

    # Empty coord array
    coords = np.zeros((bounds.shape[0], 2), dtype=np.int64)
    # Transform continuous int to discrete int for each dimension
    coords[:, 0] = _continuous_int_to_discrete_int(
        geom_mids[0], geom_ranges[0], side_length
    )
    coords[:, 1] = _continuous_int_to_discrete_int(
        geom_mids[1], geom_ranges[1], side_length
    )

    # Calculate hilbert distance
    hilbert_distances = _distances_from_coordinates(p, coords)

    return hilbert_distances


@ngjit
def _continuous_int_to_discrete_int(vals, val_range, n):

    """
    Convert an array of values from continuous data coordinates to discrete
    int coordinates

    Parameters
    ----------
    vals : Array of continuous coordinates to be
            ([([val_1, val_2,..., val_n]), array([val_1, val_2,..., val_n])])

    val_range : Ranges of x and y values ([(xmin, xmax), (ymin, ymax)])

    n : Number of discrete coords (int)

    Returns
    ---------
    Array of discrete int coords
    """

    x_width = val_range[1] - val_range[0]
    res = ((vals - val_range[0]) * (n / x_width)).astype(np.int64)

    # clip
    res[res < 0] = 0
    res[res > n - 1] = n - 1
    return res


@ngjit
def _distances_from_coordinates(p, coords):

    """
    Calculate hilbert distance for a set of coords

    Parameters
    ----------
    p : Hilbert curve param

    coords : Array of coordinates

    Returns
    ---------
    Array of hilbert distances for each geom
    """

    # Create empty coord list
    # coords = np.atleast_2d(coords).copy()
    result = np.zeros(coords.shape[0], dtype=np.int64)
    # For each coord calculate hilbert distance
    for i in range(coords.shape[0]):
        coord = coords[i, :]
        result[i] = _distance_from_coordinate(p, coord)
    return result


@ngjit
def _distance_from_coordinate(p, coord):

    """
    Calculate hilbert distance for a single coord

    Parameters
    ----------
    p : Hilbert curve param

    coord : Array of coordinates

    Returns
    ---------
    Array of hilbert distances for a single coord
    """

    n = len(coord)
    M = 1 << (p - 1)
    Q = M
    while Q > 1:
        P = Q - 1
        for i in range(n):
            if coord[i] & Q:
                coord[0] ^= P
            else:
                t = (coord[0] ^ coord[i]) & P
                coord[0] ^= t
                coord[i] ^= t
        Q >>= 1
    # Gray encode
    for i in range(1, n):
        coord[i] ^= coord[i - 1]
    t = 0
    Q = M
    while Q > 1:
        if coord[n - 1] & Q:
            t ^= Q - 1
        Q >>= 1
    for i in range(n):
        coord[i] ^= t
    h = _transpose_to_hilbert_integer(p, coord)
    return h


@ngjit
def _transpose_to_hilbert_integer(p, coord):

    """
    Calculate hilbert distance for a single coord

    Parameters
    ----------
    p : Hilbert curve param

    coord : Array of coordinates

    Returns
    ---------
    Array of hilbert distances for a single coord
    """

    n = len(coord)
    bins = [_int_2_binary(v, p) for v in coord]
    concat = np.zeros(n * p, dtype=np.uint8)
    for i in range(p):
        for j in range(n):
            concat[n * i + j] = bins[j][i]

    h = _binary_2_int(concat)
    return h


@ngjit
def _int_2_binary(v, width):

    """
    Convert an array of values from discrete int coordinates to binary byte

    Parameters
    ----------
    p : Hilbert curve param

    coord : Array of coordinates

    Returns
    ---------
    Binary byte
    """

    res = np.zeros(width, dtype=np.uint8)
    for i in range(width):
        res[width - i - 1] = v % 2  # zero-passed to width
        v = v >> 1
    return res


@ngjit
def _binary_2_int(bin_vec):

    """
    Convert binary byte to int

    Parameters
    ----------
    p : Hilbert curve param

    coord : Array of coordinates

    Returns
    ---------
    Discrete int
    """

    res = 0
    next_val = 1
    width = len(bin_vec)
    for i in range(width):
        res += next_val * bin_vec[width - i - 1]
        next_val <<= 1
    return res