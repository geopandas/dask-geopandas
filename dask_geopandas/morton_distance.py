import warnings

import pandas as pd
from dask_geopandas.hilbert_distance import _continuous_to_discrete_coords


def _morton_distance(gdf, total_bounds, level):
    """
    Calculate distance of geometries along Morton curve

    The Morton curve is also known as Z-order https://en.wikipedia.org/wiki/Z-order_curve

    Parameters
    ----------
    gdf : GeoDataFrame
    total_bounds : array_like
        array containing xmin, ymin, xmax, ymax
    level : int (1 - 16)
        Determines the precision of the Morton curve.

    Returns
    -------
    type : pandas.Series
        Series containing distances from Morton curve

    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", "GeoSeries.isna() previously returned True", UserWarning
        )
        if gdf.is_empty.any() | gdf.geometry.isna().any():
            raise ValueError(
                "Morton distance cannot be computed on a GeoSeries with empty or "
                "missing geometries.",
            )
    # Calculate bounds as numpy array
    bounds = gdf.bounds.to_numpy()
    # Calculate discrete coords based on total bounds and bounds
    x_int, y_int = _continuous_to_discrete_coords(bounds, level, total_bounds)
    # Calculate distance from morton curve
    distances = _distances_from_coordinates(x_int, y_int)

    return pd.Series(distances, index=gdf.index, name="morton_distance")


def _distances_from_coordinates(x, y):
    """
    Calculate distances from geometry mid-points along Morton curve

    Parameters
    ----------
    x, y : array_like
        x, y coordinate pairs based on mid-points of geoms

    Returns
    -------
    type : int
        Integer distances from Morton curve
    """

    return _part1by1(x) | (_part1by1(y) << 1)


def _part1by1(n):
    """
    Interleave bits by ninary magic numbers

    Based on #http://graphics.stanford.edu/~seander/bithacks.html#InterleaveBMN

    Parameters
    ----------
    n : np.array
        X or Y coordinates

    Returns
    -------
    n : int
        Interleaved bits
    """
    n &= 0x0000FFFF
    n = (n | (n << 8)) & 0x00FF00FF
    n = (n | (n << 4)) & 0x0F0F0F0F
    n = (n | (n << 2)) & 0x33333333
    n = (n | (n << 1)) & 0x55555555

    return n
