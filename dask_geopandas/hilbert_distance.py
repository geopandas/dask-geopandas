import numpy as np
import pandas as pd
from numba import jit

ngjit = jit(nopython=True, nogil=True)

# Based on: https://github.com/holoviz/spatialpandas/blob/
# 9252a7aba5f8bc7a435fffa2c31018af8d92942c/spatialpandas/dask.py


def _hilbert_distance(gdf, total_bounds, p):

    """
    Calculate hilbert distance for a GeoDataFrame
    int coordinates

    Parameters
    ----------
    gdf : GeoDataFrame

    total_bounds : Total bounds of geometries - array

    p : The number of iterations used in constructing the Hilbert curve

    Returns
    ---------
    Pandas Series containing hilbert distances
    """

    # Compute bounds as array
    bounds = gdf.bounds.to_numpy()
    # Compute hilbert distances
    coords = _continuous_to_discrete_coords(total_bounds, bounds, p)
    distances = _distances_from_coordinates(p, coords)

    return pd.Series(distances, index=gdf.index, name="hilbert_distance")


@ngjit
def _continuous_to_discrete_coords(total_bounds, bounds, p):

    """
    Calculates mid points & ranges of geoms and returns
    as discrete coords

    Parameters
    ----------

    total_bounds : Total bounds of geometries - array

    bounds : Bounds of each geometry - array

    p : The number of iterations used in constructing the Hilbert curve

    Returns
    ---------
    Discrete two-dimensional numpy array
    Two-dimensional array Array of hilbert distances for each geom
    """

    # Hilbert Side len
    side_length = 2 ** p

    # Calculate x and y range of total bound coords - returns array
    xmin, ymin, xmax, ymax = total_bounds

    # Calculate mid points for x and y bound coords - returns array
    x_mids = (bounds[:, 0] + bounds[:, 2]) / 2.0
    y_mids = (bounds[:, 1] + bounds[:, 3]) / 2.0

    # Transform continuous int to discrete int for each dimension
    x_int = _continuous_to_discrete(x_mids, (xmin, xmax), side_length)
    y_int = _continuous_to_discrete(y_mids, (ymin, ymax), side_length)
    # Stack x and y discrete ints
    coords = np.stack((x_int, y_int), axis=1)

    return coords


@ngjit
def _continuous_to_discrete(vals, val_range, n):

    """
    Convert a continuous one-dimensional array to discrete int
    based on values and their ranges

    Parameters
    ----------
    vals : Array of continuous values

    val_range : Tuple containing range of continuous values

    n : Number of discrete values

    Returns
    ---------
    One-dimensional array of discrete ints
    """

    width = val_range[1] - val_range[0]
    res = ((vals - val_range[0]) * (n / width)).astype(np.int64)

    # TO DO: When numba 0.54 releases - used.clip(res, min=0, max=n, out=res)
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
    p : The number of iterations used in constructing the Hilbert curve.

    coords : Array of coordinates

    Returns
    ---------
    Array of hilbert distances for each geom
    """

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
    p : The number of iterations used in constructing the Hilbert curve

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
    p : The number of iterations used in constructing the Hilbert curve

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
    p : The number of iterations used in constructing the Hilbert curve

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
    p : The number of iterations used in constructing the Hilbert curve

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
