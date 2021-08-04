import numpy as np
import pandas as pd
from dask_geopandas.hilbert_distance import _continuous_to_discrete_coords

# Based on #https://code.activestate.com/recipes/
# 577558-interleave-bits-aka-morton-ize-aka-z-order-curve/


def _morton_distance(gdf, total_bounds, p):
    """
    Calculate

    Parameters
    ----------

    gdf : GeoDataFrame

    total_bounds : Total bounds

    p : number of iterations

    """
    bounds = gdf.bounds.to_numpy()
    coords = _continuous_to_discrete_coords(total_bounds, bounds, p)

    distances = _distances_from_coordinates(coords)

    return pd.Series(distances, index=gdf.index, name="morton_distance")


def _distances_from_coordinates(coords):

    """
    Calculate Morton distances from coordinates

    Parameters
    ----------

    coords : Coordinates in np.array format
    """

    result = np.zeros(coords.shape[0], dtype=np.int64)
    # For each coord calculate hilbert distance
    for i in range(coords.shape[0]):
        result[i] = encode_morton(coords[i][0], coords[i][1])

    return result


def encode_morton(x, y):

    """
    Encode x and y values from coordinates

    Parameters
    ----------

    x : X coordinate

    y : Y coordinate
    """

    return part1by1(x) | (part1by1(y) << 1)


def part1by1(n):

    """Interleave bits by Binary Magic Numbers"""

    n &= 0x0000FFFF
    n = (n | (n << 8)) & 0x00FF00FF
    n = (n | (n << 4)) & 0x0F0F0F0F
    n = (n | (n << 2)) & 0x33333333
    n = (n | (n << 1)) & 0x55555555

    return n
